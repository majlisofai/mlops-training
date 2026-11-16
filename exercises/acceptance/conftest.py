"""Shared pytest setup for the access tests. You do not need to change this file.

Two ways to run the tests:

- In the test process (default). The fixture `client` starts the API with FastAPI's
  TestClient, from the app.py next to the test file. If API_CLIENT_KEY and API_ADMIN_KEY
  are not set, it makes throwaway keys for this test run.
- Against a running API. Set API_URL, for example API_URL=http://127.0.0.1:8000, and load
  the same keys that the API uses (set -a; source .secrets/api.env; set +a). The tests then
  call the container over HTTP, the way a client would. accept.py runs them this way.
"""

import csv
import importlib.util
import os
import secrets
import sys
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RACES = ROOT / "f1" / "data" / "races.csv"
INPUTS = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob", "circuit_type"]
API_URL = os.environ.get("API_URL")

if not API_URL:
    # The API in the test process needs keys too. Make throwaway ones if there are none.
    for variable in ("API_CLIENT_KEY", "API_ADMIN_KEY"):
        os.environ.setdefault(variable, secrets.token_urlsafe(32))
sys.path.insert(0, str(HERE))  # app.py imports security.py from this folder


def load_app(app_file: Path):
    """Import app.py from a path (the starter or solution/app.py) and return its FastAPI app."""
    name = "app_under_test_" + str(abs(hash(str(app_file))))
    spec = importlib.util.spec_from_file_location(name, app_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


@pytest.fixture(scope="module")
def client(request):
    """A client for the API: the running API at API_URL, or app.py in the test process."""
    if API_URL:
        with httpx.Client(base_url=API_URL, timeout=60) as live:
            try:
                live.get("/health").raise_for_status()
            except httpx.HTTPError as err:
                pytest.exit(f"The API at {API_URL} does not answer /health: {err}", returncode=1)
            yield live
        return
    from fastapi.testclient import TestClient

    app_file = Path(getattr(request.module, "APP_FILE", HERE / "app.py"))
    test_client = TestClient(load_app(app_file))
    try:
        test_client.__enter__()  # runs the lifespan handler: reads the keys, loads the model
    except Exception as err:
        pytest.exit(
            f"The API in {app_file} could not start: {err}\nIs the MLflow server running, and does @champion exist?",
            returncode=1,
        )
    yield test_client
    test_client.__exit__(None, None, None)


@pytest.fixture(scope="session")
def client_key() -> str:
    return os.environ["API_CLIENT_KEY"]


@pytest.fixture(scope="session")
def admin_key() -> str:
    return os.environ["API_ADMIN_KEY"]


@pytest.fixture(scope="session")
def race() -> dict:
    """A valid request body: the 20 cars of the first race of 2025."""
    with RACES.open() as f:
        rows = [r for r in csv.DictReader(f) if r["season"] == "2025"][:20]
    types = {"grid": int, "quali_gap_ms": int, "circuit_type": str}
    return {"entries": [{c: types.get(c, float)(r[c]) for c in INPUTS} for r in rows]}
