"""Serve f1-points-finish as a versioned HTTP API, with access control (lesson 09).

Reference solution. Try exercises/acceptance/app.py first.

The lesson 05 API, plus API keys with two roles (security.py):

    GET  /health         public: status, model name, version and alias
    POST /v1/predict     needs a client key in the X-API-Key header
    POST /admin/reload   needs an admin key in the X-API-Key header
    GET  /docs           public: the API contract, with an "Authorize" button for the key

The keys come from the environment variables API_CLIENT_KEY and API_ADMIN_KEY.
Without them the API does not start.

Usage:
    python exercises/acceptance/keys.py                  # once: writes .secrets/api.env
    set -a; source .secrets/api.env; set +a
    uvicorn app:app --app-dir exercises/acceptance/solution --port 8000
"""

import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum

# Fail in seconds, not minutes, when the MLflow server is not running.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

# security.py is in the folder above solution/. (In the Docker image both files are in /app.)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from security import Caller, load_keys, require_role  # noqa: E402

MODEL_URI = os.environ.get("MODEL_URI", "models:/f1-points-finish@champion")
# The seven model inputs, all known before the race. See f1/data/README.md.
INPUTS = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob", "circuit_type"]
MAX_BATCH = 1000


class CircuitType(str, Enum):
    street = "street"
    permanent = "permanent"
    high_speed = "high_speed"


class Entry(BaseModel):
    """One car before one race. The types and ranges are the lesson 02 schema."""

    # An unknown field is an error, not ignored: a client that sends finish_position has a bug.
    model_config = ConfigDict(extra="forbid")

    grid: int = Field(ge=1, le=20, description="Starting position after grid penalties")
    quali_gap_ms: int = Field(ge=0, description="Gap to pole position in qualifying, milliseconds")
    driver_form: float = Field(ge=1, le=20, description="Mean finishing position over the last 3 races")
    team_form: float = Field(ge=0, description="Mean team points per race over the last 3 races")
    team_dnf_rate: float = Field(ge=0, le=1, description="Share of team cars that did not finish, last 6 races")
    rain_prob: float = Field(ge=0, le=1, description="Rain forecast for the race")
    circuit_type: CircuitType


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[Entry] = Field(min_length=1, max_length=MAX_BATCH)


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: list[int] = Field(description="1 = the model predicts a points finish (top 10)")
    probabilities: list[float] = Field(description="Probability of a points finish, one per entry")


class Health(BaseModel):
    status: str
    model_name: str
    model_version: str
    alias: str | None
    model_uri: str
    loaded_at: str


def load_model(uri: str) -> dict:
    """Find the version behind the alias, then load exactly that version.

    The version in /health and in every answer is then the version that really answers,
    even when someone moves the alias while the model loads.
    """
    match = re.fullmatch(r"models:/([^/@]+)(?:@([\w-]+)|/(\d+))", uri)
    if not match:
        raise ValueError(f"MODEL_URI must look like models:/<name>@<alias> or models:/<name>/<version>, not {uri!r}")
    name, alias, version = match.groups()
    if alias:
        version = MlflowClient().get_model_version_by_alias(name, alias).version
    # MLflow saved the model with skops. It reads the trusted types from the model itself.
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")

    # The API accepts INPUTS. A model that needs other columns cannot answer this contract.
    needs = list(getattr(model, "feature_names_in_", []))
    if sorted(needs) != sorted(INPUTS):
        raise RuntimeError(f"{name} version {version} needs the inputs {needs}, but API v1 accepts {INPUTS}")
    return {
        "model": model,
        "columns": needs,
        "model_name": name,
        "model_version": str(version),
        "alias": alias,
        "model_uri": uri,
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Read the keys first. Without them the API does not start: it fails closed.
    app.state.keys = load_keys()
    # Load the model once, before the first request. If it fails, the API does not start.
    served = app.state.served = load_model(MODEL_URI)
    print(f"Serving {served['model_name']} version {served['model_version']} ({MODEL_URI})", flush=True)
    yield


app = FastAPI(title="f1-points-finish", version="1", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    """Answer 422 with one readable line per problem, for example 'entries.0.grid: Input should be ...'."""
    problems = [f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": problems})


@app.get("/health", response_model=Health)
def health():
    served = app.state.served
    return Health(status="ok", **{k: served[k] for k in Health.model_fields if k != "status"})


@app.post("/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest, caller: Caller = Depends(require_role("client"))):
    # Read the model once: the whole batch is answered by one version, even during a reload.
    served = app.state.served
    model = served["model"]
    X = pd.DataFrame([e.model_dump(mode="json") for e in request.entries])[served["columns"]]
    proba = model.predict_proba(X)
    # The same rule as model.predict(), without a second pass through the model.
    predictions = model.classes_.take(proba.argmax(axis=1))
    return PredictResponse(
        model_name=served["model_name"],
        model_version=served["model_version"],
        predictions=[int(p) for p in predictions],
        probabilities=[float(p) for p in proba[:, 1]],
    )


@app.post("/admin/reload")
def reload(caller: Caller = Depends(require_role("admin"))):
    """Load MODEL_URI again. Until the new model is loaded, the old one answers."""
    previous = app.state.served
    try:
        served = load_model(MODEL_URI)
    except Exception as err:  # keep serving the old model
        return JSONResponse(
            status_code=500,
            content={"detail": f"reload failed, still serving version {previous['model_version']}: {err}"},
        )
    app.state.served = served
    print(
        f"Reloaded {MODEL_URI}: version {previous['model_version']} -> {served['model_version']} "
        f"by {caller.role} key {caller.fingerprint}",
        flush=True,
    )
    return {
        "model_name": served["model_name"],
        "previous_version": previous["model_version"],
        "model_version": served["model_version"],
        "alias": served["alias"],
        "loaded_at": served["loaded_at"],
    }
