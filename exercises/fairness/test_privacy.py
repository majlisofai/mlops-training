"""Privacy gates: data minimisation for f1-points-finish. You do not need to change this file.

The model may use only what it needs to predict a race, and the prediction log may keep only
what the model used and what it answered. The rules are in fairness.yaml, under privacy.

Usage:
    pytest exercises/fairness/test_fairness.py exercises/fairness/test_privacy.py
    PREDICTION_LOG=exercises/fairness/examples/prediction_log_v0.jsonl pytest exercises/fairness/test_privacy.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))


def test_no_personal_identifiers_in_model_inputs(input_columns, criteria, record_property):
    """The model needs no input that identifies a person (driver_code, driver_name)."""
    found = sorted(set(input_columns) & set(criteria["privacy"]["personal_identifiers"]))
    record_property("inputs", len(input_columns))
    assert not found, f"the model uses personal identifiers as inputs: {', '.join(found)}"


def test_no_reporting_only_columns_in_model_inputs(input_columns, criteria, record_property):
    """The subgroup columns are used to measure the model, not as its inputs (rookie, team_tier)."""
    found = sorted(set(input_columns) & set(criteria["privacy"]["reporting_only"]))
    record_property("checked", ",".join(criteria["privacy"]["reporting_only"]))
    assert not found, f"the model uses reporting-only columns as inputs: {', '.join(found)}"


def _walk(value, path=""):
    """Yield (key path, key, value) for every field of a JSON record, nested or not."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield f"{path}.{key}".lstrip("."), key, inner
            yield from _walk(inner, f"{path}.{key}".lstrip("."))
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            yield from _walk(inner, f"{path}[{i}]")


def test_no_personal_data_in_prediction_log(criteria, record_property):
    """The prediction log holds no personal identifier, as a field name or as a value."""
    if not PREDICTION_LOG.exists():
        pytest.skip(f"no prediction log at {PREDICTION_LOG}. Make one with logged_api.py and send_season.py.")
    races = pd.read_csv(criteria["data"])
    identifiers = set(criteria["privacy"]["personal_identifiers"])
    names = set(races["driver_name"])
    codes = set(races["driver_code"])

    problems, records = [], 0
    for number, line in enumerate(PREDICTION_LOG.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        records += 1
        for where, key, value in _walk(json.loads(line)):
            if key in identifiers:
                problems.append(f"line {number}: field {where}")
            elif isinstance(value, str) and (value in codes or any(name in value for name in names)):
                problems.append(f"line {number}: {where} = {value!r}")
    record_property("log", str(PREDICTION_LOG))
    record_property("records", records)
    record_property("problems", len(problems))
    assert records > 0, f"{PREDICTION_LOG} is empty"
    assert not problems, (
        f"{len(problems)} personal identifiers in {PREDICTION_LOG}, first: {'; '.join(problems[:3])}"
    )
