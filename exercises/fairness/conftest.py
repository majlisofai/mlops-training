"""Shared pytest setup for the fairness and privacy gates. You do not need to change this file.

It loads fairness.yaml, the 2025 holdout and the model behind MODEL_URI (default
models:/f1-points-finish@champion), and gives them to the gates as fixtures. When the gates
finish, it writes the gate report: fairness_gate_report.json, plus fairness_gate_report.md
for people. The same idea as the lesson 04 gate report, with one more column: warnings.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from fairness_metrics import grid_rule, load_criteria, load_holdout, model_uri, resolve_version, rookie_gaps  # noqa: E402

REPORT = Path(os.environ.get("FAIRNESS_GATE_REPORT", "fairness_gate_report.json"))

GATES = []   # one entry per gate, filled while the tests run
MODEL = {}   # which model version the gates checked


@pytest.fixture(scope="session")
def criteria() -> dict:
    return load_criteria()


@pytest.fixture(scope="session")
def holdout(criteria) -> pd.DataFrame:
    """The 2025 holdout: 480 entries, with the subgroup columns and the identifiers."""
    return load_holdout(criteria)


@pytest.fixture(scope="session")
def version():
    """The registry entry behind MODEL_URI."""
    uri = model_uri()
    try:
        found = resolve_version(uri)
    except Exception as err:
        pytest.exit(f"Could not find {uri} in the MLflow registry: {err}\n"
                    "Is the MLflow server running, and does the alias exist?", returncode=1)
    MODEL.update(model=found.name, version=found.version, run_id=found.run_id, model_uri=uri)
    return found


@pytest.fixture(scope="session")
def model(version):
    return mlflow.sklearn.load_model(f"models:/{version.name}/{version.version}")


@pytest.fixture(scope="session")
def input_columns(version) -> list[str]:
    """The input columns in the model signature: everything the model needs to make a prediction."""
    info = mlflow.models.get_model_info(f"models:/{version.name}/{version.version}")
    return info.signature.inputs.input_names()


@pytest.fixture(scope="session")
def predictions(model, holdout, input_columns) -> pd.Series:
    return pd.Series(model.predict(holdout[input_columns]), index=holdout.index)


@pytest.fixture(scope="session")
def gaps(holdout, predictions, criteria) -> dict:
    """Rookie rates, everyone else's rates and the signed gaps (see fairness_metrics.rookie_gaps)."""
    return rookie_gaps(holdout, predictions, criteria["fairness"]["gated_subgroup"])


@pytest.fixture(scope="session")
def grid_gaps(holdout, criteria) -> dict:
    """The same gaps for the grid rule, for comparison."""
    return rookie_gaps(holdout, grid_rule(holdout), criteria["fairness"]["gated_subgroup"])


@pytest.fixture
def warn(record_property):
    """Record a warning: the gate still passes or fails, but the report shows the warning."""
    found = []

    def _warn(message: str):
        found.append(message)
        record_property("warning", " / ".join(found))
        warnings.warn(message, UserWarning, stacklevel=2)

    return _warn


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Record each gate's outcome, its description, the values it measured and its warnings."""
    report = yield
    if report.when == "call" or report.outcome != "passed":
        outcome = report.outcome if report.when == "call" else ("skipped" if report.skipped else "error")
        doc = (item.function.__doc__ or "").strip()
        props = dict(report.user_properties)
        GATES.append({
            "gate": item.name.removeprefix("test_"),
            "file": item.path.name,
            "description": doc.splitlines()[0] if doc else "",
            "outcome": outcome,
            "measured": {k: v for k, v in props.items() if k != "warning"},
            "warning": props.get("warning", ""),
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
        **MODEL,
        "gates_file": ",".join(str(a) for a in session.config.args),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "warnings": sum(bool(g["warning"]) for g in GATES),
        # Skipped gates are not checked. They do not block a release, and they protect nothing.
        "all_passed": bool(MODEL) and counts["passed"] > 0 and counts["failed"] + counts["error"] == 0,
        "gates": GATES,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    REPORT.with_suffix(".md").write_text(_markdown(report))


def _markdown(report: dict) -> str:
    icons = {"passed": "PASS", "failed": "FAIL", "skipped": "NOT CHECKED", "error": "ERROR"}
    skipped = report["counts"]["skipped"]
    verdict = "BLOCKED" if not report["all_passed"] else "passed" + (f", {skipped} gates NOT CHECKED" if skipped else "")
    lines = [
        f"### Fairness and privacy gates: {report.get('model', '?')} version {report.get('version', '?')}: {verdict}",
        "",
        "| Gate | Result | Measured | Detail | Warning |",
        "|---|---|---|---|---|",
    ]
    for g in report["gates"]:
        measured = ", ".join(f"{k} {v}" for k, v in g["measured"].items())
        cells = [g["gate"], icons[g["outcome"]], measured, g["detail"], g["warning"]]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"
