"""Train a challenger on newer data, in time order, and register it as f1-points-finish@challenger.

Your task: complete TODO 1 to TODO 3. The rest is done for you. Until a TODO is done,
the script stops at it and says which one. See exercises/recovery/README.md, step 7.

The challenger is controlled: it uses the lesson 04 model code (build_model in
train_candidate.py) with the hyperparameters in challenger.yaml, the same as the champion's.
Only the training data changes:
  - the history seasons from races.csv (challenger.yaml, data.history_seasons),
  - plus the 2026 races before the test window, with their FINAL labels (data.train_2026_rounds).
compare.py then measures champion and challenger on the test window in comparison.yaml
(2026 rounds 5 to 8), which the challenger must not see in training.

Usage:
    python exercises/recovery/train_challenger.py
    python exercises/recovery/train_challenger.py --challenger exercises/recovery/examples/all_history.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, roc_auc_score

# The lesson 04 model code: the same pipeline as the champion.
sys.path.insert(0, "exercises/quality-gates")
from train_candidate import TRUSTED_TYPES, build_model, data_version, git_commit  # noqa: E402

MODEL_NAME = "f1-points-finish"
EXPERIMENT = "f1-points-finish"
TARGET = "points_finish"
NUMERIC = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob"]
CATEGORICAL = ["circuit_type"]
FEATURES = NUMERIC + CATEGORICAL
# Columns known before the race: identity, subgroups and the seven model inputs.
PRE_RACE = ["race_id", "season", "round", "circuit", "driver_code", "driver_name", "team", "team_tier", "rookie"]


def labelled_2026(provisional: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """The 2026 entries with their final labels.

    The inputs come from the provisional file (they are known before the race). The label comes
    from the final file: points_finish_final, after the stewards' decisions. The provisional
    points_finish is the result on the evening of the race, and some of it changed later.
    """
    # TODO 1: keep the columns PRE_RACE + FEATURES of `provisional` (the inputs known before the race),
    #         and merge in points_finish_final from `final` on race_id and driver_code. Use
    #         how="inner", so that a race whose final labels have not arrived is left out, and
    #         validate="one_to_one". Rename points_finish_final to TARGET and return the result.
    #         Do not use points_finish from the provisional file: 6 of its 160 labels changed.
    #         Then delete the raise line.
    raise NotImplementedError("TODO 1: labelled_2026() is not written yet")


def training_data(races: pd.DataFrame, rows_2026: pd.DataFrame, data: dict) -> pd.DataFrame:
    """The history seasons from races.csv, plus the 2026 races listed in data['train_2026_rounds']."""
    # TODO 2: take the rows of `races` whose season is in data["history_seasons"], and the rows
    #         of `rows_2026` whose round is in data["train_2026_rounds"]. Keep the columns
    #         PRE_RACE + FEATURES + [TARGET] of the history rows, and return both together
    #         (pd.concat with ignore_index=True). Take only the rounds that the file lists:
    #         compare.py checks that none of them is in the test window.
    #         Then delete the raise line.
    raise NotImplementedError("TODO 2: training_data() is not written yet")


def sample_weights(train: pd.DataFrame, weight_2026: float) -> np.ndarray:
    """One weight per training entry: weight_2026 for a 2026 entry, 1 for a history entry."""
    # TODO 3: return a numpy array with one number per row of `train`: weight_2026 where the
    #         season is 2026, 1.0 everywhere else. Hint: np.where(condition, a, b).
    #         Then delete the raise line.
    raise NotImplementedError("TODO 3: sample_weights() is not written yet")


def evaluate(model, test: pd.DataFrame) -> dict:
    """Metrics on the test window, for the MLflow UI. compare.py measures them again."""
    proba = model.predict_proba(test[FEATURES])[:, 1]
    return {
        "test_accuracy": float(accuracy_score(test[TARGET], proba > 0.5)),
        "test_roc_auc": float(roc_auc_score(test[TARGET], proba)),
    }


def span(values: list) -> str:
    """[2023, 2024, 2025] -> '2023-2025'; [1, 2, 3, 4] -> '1-4'; [1, 3] -> '1,3'."""
    values = sorted(values)
    if values == list(range(values[0], values[-1] + 1)) and len(values) > 1:
        return f"{values[0]}-{values[-1]}"
    return ",".join(str(v) for v in values)


def run_name(data: dict) -> str:
    rounds = f"+2026r{span(data['train_2026_rounds'])}" if data["train_2026_rounds"] else ""
    return f"challenger-{span(data['history_seasons'])}{rounds}-w{data['weight_2026']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--challenger", type=Path, default=Path("exercises/recovery/challenger.yaml"))
    parser.add_argument("--comparison", type=Path, default=Path("exercises/recovery/comparison.yaml"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    challenger = yaml.safe_load(args.challenger.read_text())
    comparison = yaml.safe_load(args.comparison.read_text())
    data, paths = challenger["data"], comparison["data"]
    races = pd.read_csv(paths["races"])
    provisional = pd.read_csv(paths["provisional_2026"])
    final = pd.read_csv(paths["final_2026"])

    try:
        rows_2026 = labelled_2026(provisional, final)
        train = training_data(races, rows_2026, data)
        weights = sample_weights(train, data["weight_2026"])
    except NotImplementedError as err:
        sys.exit(f"STOPPED at {err}. Nothing was trained. See exercises/recovery/README.md, step 7.")
    test = rows_2026[rows_2026["round"].isin(comparison["test_rounds_2026"])]

    model = build_model(challenger, NUMERIC)
    model.fit(train[FEATURES], train[TARGET], model__sample_weight=weights)
    metrics = evaluate(model, test)
    train_2026 = train[train["season"] == 2026]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name(data)) as run:
        mlflow.set_tags({
            "role": "challenger",
            "data.version": data_version(Path(paths["races"])),
            "data.version_2026_final": data_version(Path(paths["final_2026"])),
            "git.commit": git_commit(),
            "challenger.file": str(args.challenger),
            "proposed_by": str(challenger.get("proposed_by", "unknown")),
        })
        mlflow.log_params({
            "model_type": challenger["model"],
            "features": ",".join(FEATURES),
            "history_seasons": span(data["history_seasons"]),
            "train_2026_rounds": ",".join(str(r) for r in sorted(data["train_2026_rounds"])),
            "train_2026_races": ",".join(sorted(train_2026["race_id"].unique())),
            "labels_2026": "final",
            "weight_2026": data["weight_2026"],
            "test_rounds_2026": ",".join(str(r) for r in comparison["test_rounds_2026"]),
            **(challenger.get("hyperparameters") or {}),
        })
        mlflow.log_metrics({**metrics, "train_entries": len(train), "train_entries_2026": len(train_2026)})
        mlflow.log_artifact(str(args.challenger), artifact_path="challenger")
        info = mlflow.sklearn.log_model(
            model, name="model", input_example=test[FEATURES].head(5), skops_trusted_types=TRUSTED_TYPES
        )

    version = mlflow.register_model(info.model_uri, MODEL_NAME).version
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    client.set_model_version_tag(MODEL_NAME, version, "proposed_by", str(challenger.get("proposed_by", "unknown")))
    client.set_model_version_tag(MODEL_NAME, version, "run_name", run_name(data))

    print(
        f"{run_name(data)}: trained on {len(train)} entries ({len(train_2026)} from 2026, final labels), "
        f"history {span(data['history_seasons'])}, 2026 rounds {span(data['train_2026_rounds']) or 'none'}"
    )
    print(
        f"On the test window (2026 rounds {span(comparison['test_rounds_2026'])}, {len(test)} entries): "
        f"accuracy {metrics['test_accuracy']:.3f}, roc_auc {metrics['test_roc_auc']:.3f}"
    )
    print(f"Registered {MODEL_NAME} version {version} as @challenger (run {run.info.run_id})")


if __name__ == "__main__":
    main()
