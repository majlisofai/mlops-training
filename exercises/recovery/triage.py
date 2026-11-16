"""Collect the evidence for an incident on the f1-points-finish service. It reads, it changes nothing.

Each part answers one question of the runbook (RUNBOOK.md):
    health    Is the API up? Which model version does it serve? What do the container and its log say?
    registry  Which version is @champion? Who moved it, when, and did the gates run? What inputs does it need?
    inputs    Do the 2026 inputs keep the API contract? How far have they moved from the training seasons?
    labels    How right is the served model on the 2026 races whose final labels have arrived?
    all       All four, in that order.

Usage:
    python exercises/recovery/triage.py health
    python exercises/recovery/triage.py labels
    python exercises/recovery/triage.py labels --model-uri models:/f1-points-finish/1   # offline, one version
    python exercises/recovery/triage.py all --out incident_evidence.md

Environment: API_URL (default http://127.0.0.1:8000), CONTAINER (default f1-api),
MLFLOW_TRACKING_URI (default http://127.0.0.1:5000).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

# The API contract (lesson 05): the same Entry model that app.py uses to check every request.
sys.path.insert(0, "exercises/serve-model")
from app import INPUTS, Entry  # noqa: E402

MODEL_NAME = "f1-points-finish"
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
CONTAINER = os.environ.get("CONTAINER", "f1-api")
RACES = Path("f1/data/races.csv")
PROVISIONAL = Path("f1/data/season_2026_provisional.csv")
FINAL = Path("f1/data/season_2026_final.csv")
FORBIDDEN = ["finish_position", "points", "status", "laps_completed", "fastest_lap", "actual_rain", "points_finish"]
NUMERIC = [c for c in INPUTS if c != "circuit_type"]

LINES: list[str] = []  # everything printed, for --out


def say(text: str = ""):
    print(text)
    LINES.append(text)


def section(title: str):
    say()
    say(f"== {title}  ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")


def http(method: str, path: str, body=None, timeout: float = 10):
    """Return (status, parsed JSON or text). status None means no answer at all."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API_URL + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode(errors="replace")[:500]
    except (urllib.error.URLError, OSError) as err:
        return None, str(getattr(err, "reason", err))


# ---------------------------------------------------------------------------- health

def health():
    section(f"1. Health: {API_URL}/health")
    status, body = http("GET", "/health", timeout=3)
    if status == 200:
        say(f"UP: model {body['model_name']} version {body['model_version']} (alias {body['alias']}), "
            f"loaded at {body['loaded_at']}")
    else:
        say(f"DOWN: {'no answer: ' + body if status is None else f'status {status}: {body}'}")

    if not shutil.which("docker"):
        say("(docker not found: no container state or log)")
        return
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}|{{.RestartCount}}|{{.State.StartedAt}}|{{.State.ExitCode}}",
         CONTAINER], capture_output=True, text=True)
    if inspect.returncode != 0:
        say(f"Container {CONTAINER}: not found. Is the API running as a container (step 1)?")
        return
    state, restarts, started, exit_code = inspect.stdout.strip().split("|")
    say(f"Container {CONTAINER}: {state}, {restarts} restarts, last start {started[:19]}Z, last exit code {exit_code}")
    logs = subprocess.run(["docker", "logs", "--timestamps", "--tail", "400", CONTAINER],
                          capture_output=True, text=True)
    lines = (logs.stdout + logs.stderr).splitlines()
    lines.sort()  # stdout and stderr arrive separately; the timestamps put them back in order
    keys = ("Serving", "Reloaded", "Error:", "ERROR:", "/admin/reload")
    seen: dict[str, list] = {}  # message -> [first time, last time, count], in order of first appearance
    for line in lines:
        stamp, _, message = line.partition(" ")
        message = message.strip()
        if any(k in message for k in keys) and "Traceback" not in message:
            first, _, count = seen.get(message, [stamp, stamp, 0])
            seen[message] = [first, stamp, count + 1]
    say("Log lines that matter (last 400 lines, repeats counted):")
    for message, (first, last, count) in seen.items():
        when = f"{first[:19]}Z" + (f" .. {last[11:19]}Z, {count} times" if count > 1 else "")
        say(f"  {when}: {message[:300]}")


# ---------------------------------------------------------------------------- registry

def registry():
    section(f"2. Registry: {MODEL_NAME} at {mlflow.get_tracking_uri()}")
    client = MlflowClient()
    try:
        aliases = client.get_registered_model(MODEL_NAME).aliases
    except Exception as err:  # noqa: BLE001
        say(f"Cannot read the registry: {str(err).splitlines()[0]}")
        return
    by_version: dict[str, list[str]] = {}
    for alias, version in aliases.items():
        by_version.setdefault(str(version), []).append("@" + alias)
    versions = sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version))
    for v in versions:
        created = datetime.fromtimestamp(v.creation_timestamp / 1000, timezone.utc).isoformat(timespec="seconds")
        say(f"version {v.version} {' '.join(sorted(by_version.get(v.version, [])))}".rstrip())
        say(f"  created {created}, run {v.tags.get('run_name', '?')}, proposed by {v.tags.get('proposed_by', '?')}")
        if "promoted_by" in v.tags:
            say(f"  promoted by {v.tags['promoted_by']} at {v.tags.get('promoted_at', '?')}, "
                f"previous champion {v.tags.get('previous_champion', '?')}, gates: {v.tags.get('gates', 'none recorded')}")
        if "rolled_back_by" in v.tags:
            say(f"  rolled back by {v.tags['rolled_back_by']} at {v.tags.get('rolled_back_at', '?')}")
        try:
            inputs = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{v.version}").signature.inputs.input_names()
        except Exception as err:  # noqa: BLE001
            say(f"  inputs: cannot read the model signature: {str(err).splitlines()[0]}")
            continue
        extra = [c for c in inputs if c not in INPUTS]
        forbidden = [c for c in inputs if c in FORBIDDEN]
        verdict = "the API v1 inputs" if sorted(inputs) == sorted(INPUTS) else (
            f"NOT the API v1 inputs: extra {extra}" + (f", after-race {forbidden}" if forbidden else ""))
        say(f"  inputs: {len(inputs)}, {verdict}")
        metrics = client.get_run(v.run_id).data.metrics
        say("  run metrics: " + (", ".join(f"{k} {val:.3f}" for k, val in sorted(metrics.items())) or "none"))


# ---------------------------------------------------------------------------- inputs

def season_2026() -> pd.DataFrame:
    """The 2026 entries: pre-race inputs and provisional results, with the final label added."""
    final = pd.read_csv(FINAL)[["race_id", "driver_code", "points_finish_final"]]
    return pd.read_csv(PROVISIONAL).merge(final, on=["race_id", "driver_code"], how="left", validate="one_to_one")


def inputs():
    section("3. Inputs: the 2026 requests (pre-race inputs of rounds 1-8) against the contract and the history")
    rows = season_2026()
    history = pd.read_csv(RACES)
    invalid = []
    for record in rows[INPUTS].to_dict(orient="records"):
        try:
            Entry.model_validate(record)
        except Exception as err:  # noqa: BLE001
            invalid.append(str(err).splitlines()[1])
    say(f"Contract (API v1, app.py Entry): {len(rows) - len(invalid)} of {len(rows)} entries valid"
        + (f"; first problem: {invalid[0]}" if invalid else ""))
    say("Distribution, history 2021-2025 vs 2026 (shift = difference of the means in history standard deviations):")
    say(f"  {'input':<14} {'history mean':>12} {'2026 mean':>10} {'history median':>15} {'2026 median':>12} {'shift':>6}")
    for col in NUMERIC:
        h, n = history[col], rows[col]
        shift = (n.mean() - h.mean()) / h.std()
        flag = "  MOVED" if abs(shift) >= 0.5 else ""
        say(f"  {col:<14} {h.mean():>12.3f} {n.mean():>10.3f} {h.median():>15.3f} {n.median():>12.3f} {shift:>+6.2f}{flag}")
    shares = lambda s: ", ".join(f"{k} {v:.2f}" for k, v in s.value_counts(normalize=True).sort_index().items())  # noqa: E731
    say(f"  circuit_type   history {shares(history['circuit_type'])} | 2026 {shares(rows['circuit_type'])}")
    say("Outcomes (known after the race):")
    say(f"  did not finish: history {(history['status'] == 'DNF').mean():.1%}, 2026 {(rows['status'] == 'DNF').mean():.1%}")
    say(f"  final labels arrived for {rows['points_finish_final'].notna().sum()} of {len(rows)} 2026 entries; "
        f"{int((rows['points_finish'] != rows['points_finish_final']).sum())} changed from the provisional result")


# ---------------------------------------------------------------------------- labels

def predict_api(frame: pd.DataFrame):
    """Ask the live API for one race at a time, like the pit wall does. Return (predictions, version)."""
    predictions, versions = [], set()
    for _, race in frame.groupby("race_id", sort=False):
        status, body = http("POST", "/v1/predict", {"entries": race[INPUTS].to_dict(orient="records")})
        if status != 200:
            raise RuntimeError(f"POST /v1/predict: {'no answer: ' + body if status is None else f'status {status}: {body}'}")
        predictions += body["predictions"]
        versions.add(body["model_version"])
    return pd.Series(predictions, index=frame.index), ",".join(sorted(versions))


def predict_offline(model, version: str):
    def run(frame: pd.DataFrame):
        return pd.Series((model.predict_proba(frame[INPUTS])[:, 1] > 0.5).astype(int), index=frame.index), version
    return run


def labels(model_uri: str | None):
    rows = season_2026()
    rows = rows[rows["points_finish_final"].notna()]
    history = pd.read_csv(RACES)
    reference = history[history["season"] == 2025]
    if model_uri:
        section(f"4. Delayed labels: {model_uri}, offline")
        model = mlflow.sklearn.load_model(model_uri)
        needs = list(getattr(model, "feature_names_in_", []))
        if sorted(needs) != sorted(INPUTS):
            say(f"Cannot evaluate {model_uri}: it needs the inputs {needs}. Before a race only the API v1 inputs exist: "
                f"{INPUTS}. Missing: {sorted(set(needs) - set(INPUTS))}.")
            return
        predict = predict_offline(model, model_uri.rsplit("/", 1)[-1])
    else:
        section(f"4. Delayed labels: what {API_URL} serves now")
        status, _ = http("GET", "/health", timeout=3)
        if status != 200:
            say("The API does not answer, so nobody can measure what it serves. Measure one version offline:")
            say("  python exercises/recovery/triage.py labels --model-uri models:/f1-points-finish/<version>")
            return
        predict = predict_api
    try:
        pred_ref, version = predict(reference)
        pred, version = predict(rows)
    except RuntimeError as err:
        say(f"Cannot measure: {err}")
        return
    correct = pred == rows["points_finish_final"]
    say(f"Model version {version}. For reference: accuracy {(pred_ref == reference['points_finish']).mean():.3f} "
        f"on the 2025 holdout ({len(reference)} entries).")
    say(f"  {'race':<8} {'accuracy':>8} {'predicted points':>17} {'scored points':>14} {'DNF':>4}")
    for race, group in rows.groupby("race_id"):
        say(f"  {race:<8} {correct[group.index].mean():>8.2f} {pred[group.index].mean():>17.2f} "
            f"{group['points_finish_final'].mean():>14.2f} {(group['status'] == 'DNF').sum():>4}")
    say(f"2026 overall: accuracy {correct.mean():.3f} on {len(rows)} entries with final labels "
        f"(on the provisional labels it would read {(pred == rows['points_finish']).mean():.3f}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["health", "registry", "inputs", "labels", "all"])
    parser.add_argument("--model-uri", help="labels: evaluate this model version offline instead of the live API")
    parser.add_argument("--out", type=Path, help="also write the evidence to this Markdown file")
    args = parser.parse_args()

    if args.part in ("health", "all"):
        health()
    if args.part in ("registry", "all"):
        registry()
    if args.part in ("inputs", "all"):
        inputs()
    if args.part in ("labels", "all"):
        labels(args.model_uri)
    if args.out:
        args.out.write_text("# Incident evidence (triage.py)\n\n```\n" + "\n".join(LINES).strip() + "\n```\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
