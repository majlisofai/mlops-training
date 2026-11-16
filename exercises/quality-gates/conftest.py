"""Shared pytest setup for the quality gates. You do not need to change this file.

It loads criteria.yaml, the 2025 holdout and the model f1-points-finish@candidate,
and gives them to the gates as fixtures. When the gates finish, it writes the gate
report that promote.py reads: gate_report.json, plus gate_report.md for people.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml
from mlflow import MlflowClient

HERE = Path(__file__).parent
TARGET = "points_finish"
REPORT = Path(os.environ.get("GATE_REPORT", "gate_report.json"))
# The alias of the version to check. Lesson 09's acceptance suite checks @champion again.
ALIAS = os.environ.get("GATE_ALIAS", "candidate")

GATES = []      # one entry per gate, filled while the tests run
CANDIDATE = {}  # which model version the gates checked


@pytest.fixture(scope="session")
def criteria() -> dict:
    return yaml.safe_load((HERE / "criteria.yaml").read_text())


@pytest.fixture(scope="session")
def races(criteria) -> pd.DataFrame:
    """All of races.csv: training seasons and the holdout."""
    return pd.read_csv(criteria["data"]["path"])


@pytest.fixture(scope="session")
def holdout(races, criteria) -> pd.DataFrame:
    """The holdout season (2025): 480 entries the model never saw in training."""
    return races[races["season"] == criteria["test_season"]]


@pytest.fixture(scope="session")
def candidate(criteria):
    """The model version that the alias @candidate points to (or GATE_ALIAS, if set)."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    version = MlflowClient().get_model_version_by_alias(criteria["model_name"], ALIAS)
    CANDIDATE.update(model=criteria["model_name"], version=version.version, run_id=version.run_id)
    return version


@pytest.fixture(scope="session")
def model_uri(candidate, criteria) -> str:
    return f"models:/{criteria['model_name']}/{candidate.version}"


@pytest.fixture(scope="session")
def model(model_uri):
    """The fitted scikit-learn pipeline, loaded from the Model Registry.

    The model was saved with skops. MLflow reads the trusted types from the model itself.
    """
    return mlflow.sklearn.load_model(model_uri)


@pytest.fixture(scope="session")
def input_columns(model_uri) -> list[str]:
    """The input columns in the model signature: what the model needs to make a prediction."""
    return mlflow.models.get_model_info(model_uri).signature.inputs.input_names()


@pytest.fixture(scope="session")
def predictions(model, holdout, input_columns) -> pd.Series:
    """The model's 0/1 predictions for every 2025 entry, aligned with `holdout`."""
    return pd.Series(model.predict(holdout[input_columns]), index=holdout.index)


@pytest.fixture(scope="session")
def probabilities(model, holdout, input_columns) -> pd.Series:
    """The model's probability of a points finish for every 2025 entry."""
    return pd.Series(model.predict_proba(holdout[input_columns])[:, 1], index=holdout.index)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Record each gate's outcome, its description and the values it measured."""
    report = yield
    if report.when == "call" or report.outcome != "passed":
        outcome = report.outcome if report.when == "call" else ("skipped" if report.skipped else "error")
        doc = (item.function.__doc__ or "").strip()
        GATES.append({
            "gate": item.name.removeprefix("test_"),
            "description": doc.splitlines()[0] if doc else "",
            "outcome": outcome,
            "measured": {k: v for k, v in report.user_properties},
            "detail": _detail(report),
        })
    return report


def _detail(report) -> str:
    if report.passed:
        return ""
    if report.skipped:
        return str(report.longrepr[2]).removeprefix("Skipped: ")
    crash = getattr(report.longrepr, "reprcrash", None)
    return (crash.message if crash else str(report.longrepr)).splitlines()[0]


def pytest_sessionfinish(session):
    counts = {o: sum(g["outcome"] == o for g in GATES) for o in ("passed", "failed", "skipped", "error")}
    report = {
        **CANDIDATE,
        "gates_file": ",".join(str(a) for a in session.config.args),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        # Skipped gates are not checked. They do not block a release, and they protect nothing.
        "all_passed": bool(CANDIDATE) and counts["passed"] > 0 and counts["failed"] + counts["error"] == 0,
        "gates": GATES,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    REPORT.with_suffix(".md").write_text(_markdown(report))


def _markdown(report: dict) -> str:
    icons = {"passed": "PASS", "failed": "FAIL", "skipped": "NOT CHECKED", "error": "ERROR"}
    skipped = report["counts"]["skipped"]
    verdict = "BLOCKED" if not report["all_passed"] else "passed" + (f", {skipped} gates NOT CHECKED" if skipped else "")
    lines = [
        f"### Quality gates: {report.get('model', '?')} version {report.get('version', '?')}: {verdict}",
        "",
        "| Gate | Result | Measured | Detail |",
        "|---|---|---|---|",
    ]
    for g in report["gates"]:
        measured = ", ".join(f"{k} {v}" for k, v in g["measured"].items())
        detail = g["detail"].replace("|", "\\|")
        lines.append(f"| {g['gate']} | {icons[g['outcome']]} | {measured} | {detail} |")
    return "\n".join(lines) + "\n"
