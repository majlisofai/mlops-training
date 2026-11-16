"""Start the lesson 07 incident on the running f1-points-finish service.

TRAINEES: do not read this file before you have written your diagnosis (step 4).
It holds the answer. Your trainer runs it, or tells you when to run it.

For the trainer, what it does, in order:
  1. Checks that the MLflow server answers, that @champion exists and that the API container
     is up. It changes nothing if one of them is missing.
  2. Plays a colleague's "weekend hotfix" for the 2026 DNFs: trains a forest on 2021-2025
     plus all eight 2026 races with their PROVISIONAL labels, and adds laps_completed as an
     input (known only after the race), and registers it as a new version.
  3. Moves @champion to that version directly, without @candidate, gates or approval. It tags the
     version like promote.py does (previous_champion, promoted_by), so promote.py --rollback works.
  4. Calls POST /admin/reload. The API refuses the new version and keeps serving the old one.
  5. Restarts the container, like a nightly restart. The API now loads @champion at start-up,
     cannot load it, and the container crash-loops: the service is down.
Recovery: promote.py --rollback, then the container comes up again on its next restart.

Usage:
    python exercises/recovery/incident.py
    python exercises/recovery/incident.py --container f1-api --api http://127.0.0.1:8000
"""

import argparse
import contextlib
import io
import json
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

sys.path.insert(0, "exercises/quality-gates")
from train_candidate import TRUSTED_TYPES, build_model  # noqa: E402

MODEL_NAME = "f1-points-finish"
NUMERIC = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob"]
HOTFIX_INPUT = "laps_completed"  # known only after the race
HOTFIX = {"model": "forest", "hyperparameters": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5}}


def call(api: str, method: str, path: str) -> tuple[int | None, str]:
    request = urllib.request.Request(api + path, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()
    except (urllib.error.URLError, OSError) as err:
        return None, str(err)


def stop(message: str):
    sys.exit(f"incident.py: nothing was changed. {message}")


def preflight(client: MlflowClient, api: str, container: str):
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException as err:
        stop(f"{MODEL_NAME} has no @champion, or MLflow does not answer ({str(err).splitlines()[0][:120]}). Do step 1.")
    state = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", container], capture_output=True, text=True)
    if state.stdout.strip() != "running":
        stop(f"The container {container} is not running. Start it as in step 1.")
    status, body = call(api, "GET", "/health")
    if status != 200:
        stop(f"The API at {api} does not answer /health ({body[:120]}). Start it as in step 1.")
    return champion


def train_hotfix() -> tuple:
    races = pd.read_csv("f1/data/races.csv")
    season = pd.read_csv("f1/data/season_2026_provisional.csv")  # provisional labels, not the final ones
    train = pd.concat([races, season], ignore_index=True)
    numeric = NUMERIC + [HOTFIX_INPUT]
    features = numeric + ["circuit_type"]
    model = build_model(HOTFIX, numeric).fit(train[features], train["points_finish"])
    accuracy = float((model.predict(season[features]) == season["points_finish"]).mean())
    return model, features, season, accuracy


def register_hotfix(model, features, season, accuracy) -> str:
    mlflow.set_experiment(MODEL_NAME)
    with mlflow.start_run(run_name="hotfix-2026-dnf"):
        mlflow.set_tags({
            "proposed_by": "weekend-hotfix",
            "mlflow.note.content": "Quick fix for the 2026 DNFs: retrained with 2026 and laps_completed. "
                                   "Gates skipped, they test on 2025 and 2025 is stale.",
        })
        mlflow.log_params({
            "model_type": "forest",
            "features": ",".join(features),
            "train_data": "races.csv 2021-2025 + season_2026_provisional.csv rounds 1-8",
            "labels_2026": "provisional",
            **HOTFIX["hyperparameters"],
        })
        mlflow.log_metrics({"accuracy_2026": accuracy})
        info = mlflow.sklearn.log_model(model, name="model", input_example=season[features].head(5),
                                        skops_trusted_types=TRUSTED_TYPES)
    return mlflow.register_model(info.model_uri, MODEL_NAME).version


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", default=os.environ.get("CONTAINER", "f1-api"))
    parser.add_argument("--api", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    api = args.api.rstrip("/")
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    champion = preflight(client, api, args.container)
    print("Sunday, 22:40. Someone is working on the f1-points-finish service ...", flush=True)

    # The hotfix: quiet, like a script that nobody watched.
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        version = register_hotfix(*train_hotfix())
        client.set_model_version_tag(MODEL_NAME, version, "proposed_by", "weekend-hotfix")
        client.set_model_version_tag(MODEL_NAME, version, "run_name", "hotfix-2026-dnf")
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        tags = {
            "promoted_by": "hotfix-script",
            "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "previous_champion": champion.version,
            "gates": "not run",
        }
        for key, value in tags.items():
            client.set_model_version_tag(MODEL_NAME, version, key, value)

    call(api, "POST", "/admin/reload")
    print("Monday, 03:00. The nightly restart of the API container ...", flush=True)
    subprocess.run(["docker", "restart", args.container], capture_output=True, text=True)

    print(json.dumps({
        "pager": "f1-api health check FAILING",
        "since": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": "Race engineers: the dashboard shows no predictions for the next race.",
        "next": "Follow exercises/recovery/RUNBOOK.md. Collect evidence before you change anything.",
    }, indent=2))


if __name__ == "__main__":
    main()
