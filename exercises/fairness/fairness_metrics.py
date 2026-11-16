"""Shared helpers for lesson 08. You do not need to change this file.

The fairness gates (conftest.py), mitigate.py and model_card.py use these helpers, so that
every script measures the same thing in the same way.
"""

import os
import re
from pathlib import Path

# The same MLflow server as lessons 02 to 05. Fail in seconds, not minutes, when it is not running.
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from fairlearn.metrics import false_negative_rate, false_positive_rate, selection_rate  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

HERE = Path(__file__).parent
TARGET = "points_finish"
DEFAULT_MODEL_URI = "models:/f1-points-finish@champion"

# The four metrics of the lesson, as fairlearn and scikit-learn functions of (y_true, y_pred).
METRICS = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,       # share of entries predicted to score points
    "false_positive_rate": false_positive_rate,   # predicted points, did not score / did not score
    "false_negative_rate": false_negative_rate,   # predicted no points, scored / scored
}
SHORT = {"accuracy": "acc", "selection_rate": "sel", "false_positive_rate": "fpr", "false_negative_rate": "fnr"}


def load_criteria() -> dict:
    return yaml.safe_load((HERE / "fairness.yaml").read_text())


def model_uri() -> str:
    return os.environ.get("MODEL_URI", DEFAULT_MODEL_URI)


def resolve_version(uri: str):
    """Return the registry entry (ModelVersion) behind models:/<name>@<alias> or models:/<name>/<version>."""
    from mlflow import MlflowClient

    match = re.fullmatch(r"models:/([^/@]+)(?:@([\w-]+)|/(\d+))", uri)
    if not match:
        raise SystemExit(f"MODEL_URI must look like models:/<name>@<alias> or models:/<name>/<version>, not {uri!r}")
    name, alias, version = match.groups()
    client = MlflowClient()
    return client.get_model_version_by_alias(name, alias) if alias else client.get_model_version(name, version)


def load_holdout(criteria: dict) -> pd.DataFrame:
    races = pd.read_csv(criteria["data"])
    return races[races["season"] == criteria["test_season"]].reset_index(drop=True)


def rates(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """The four metrics, plus the counts each rate rests on."""
    out = {name: float(fn(y_true, y_pred)) for name, fn in METRICS.items()}
    out.update(entries=int(len(y_true)), scorers=int((y_true == 1).sum()), non_scorers=int((y_true == 0).sum()))
    return out


def rookie_gaps(holdout: pd.DataFrame, y_pred: pd.Series, column: str = "rookie") -> dict:
    """Rates for rookies (True) and everyone else (False), and the signed gaps.

    gap = rookie rate - rate of everyone else. For the error rates, positive = worse for rookies.
    For accuracy the gap is the other way round (everyone else - rookies), so that positive
    means worse for rookies there too.
    """
    y_true, y_pred = holdout[TARGET], pd.Series(y_pred, index=holdout.index)
    mask = holdout[column].astype(bool)
    group, rest = rates(y_true[mask], y_pred[mask]), rates(y_true[~mask], y_pred[~mask])
    return {
        "group": group,
        "rest": rest,
        "drivers": int(holdout.loc[mask, "driver_code"].nunique()),
        "accuracy_gap": rest["accuracy"] - group["accuracy"],
        "selection_rate_gap": group["selection_rate"] - rest["selection_rate"],
        "fpr_gap": group["false_positive_rate"] - rest["false_positive_rate"],
        "fnr_gap": group["false_negative_rate"] - rest["false_negative_rate"],
    }


def gate_verdicts(gaps: dict, criteria: dict) -> dict:
    """True/False per fairness requirement. The gates assert the same rules."""
    f = criteria["fairness"]
    return {
        "group_size": gaps["group"]["entries"] >= f["min_group_entries"],
        "fpr_gap": gaps["fpr_gap"] <= f["max_fpr_gap"],
        "fnr_gap": gaps["fnr_gap"] <= f["max_fnr_gap"],
    }


def grid_rule(holdout: pd.DataFrame) -> pd.Series:
    """The rule anyone can apply without a model: start in the top ten, score points."""
    return (holdout["grid"] <= 10).astype(int)
