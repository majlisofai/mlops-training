"""Shared pytest setup for the API contract tests. You do not need to change this file.

It starts the API in the test process with FastAPI's TestClient. Starting it runs the
lifespan handler in app.py, so the tests use the real model behind MODEL_URI
(default models:/f1-points-finish@champion), loaded from the MLflow server.

Lesson 09: with API_URL set, the same tests call a running API over HTTP instead, and send
the key in API_KEY as the X-API-Key header. The acceptance suite tests the container this way.
"""

import csv
import os
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from app import MODEL_URI, app

RACES = Path(__file__).parents[2] / "f1" / "data" / "races.csv"
INPUTS = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob", "circuit_type"]


@pytest.fixture(scope="session")
def client():
    """A client for the API. The model is loaded once for all the tests."""
    if os.environ.get("API_URL"):  # lesson 09: test a running API, with a client key
        headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
        with httpx2.Client(base_url=os.environ["API_URL"], headers=headers, timeout=60) as live:
            yield live
        return
    test_client = TestClient(app)
    try:
        test_client.__enter__()  # runs the lifespan handler: loads the model
    except Exception as err:
        pytest.exit(
            f"The API could not load {MODEL_URI}: {err}\n"
            "Is the MLflow server running (step 1), and does @champion exist (step 2)?",
            returncode=1,
        )
    yield test_client
    test_client.__exit__(None, None, None)


@pytest.fixture(scope="session")
def season_2025() -> list[dict]:
    """The model inputs of all 480 entries of 2025, as the API expects them."""
    with RACES.open() as f:
        rows = [r for r in csv.DictReader(f) if r["season"] == "2025"]
    types = {"grid": int, "quali_gap_ms": int, "circuit_type": str}
    return [{c: types.get(c, float)(r[c]) for c in INPUTS} for r in rows]


@pytest.fixture
def entry(season_2025) -> dict:
    """One valid entry: pole position at the first race of 2025. A fresh copy for each test."""
    return dict(season_2025[0])
