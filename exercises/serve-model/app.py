"""Serve f1-points-finish as a versioned HTTP API.

The model is loaded once, at start-up, from MODEL_URI (default models:/f1-points-finish@champion).
POST /admin/reload loads it again, for example after promote.py moved @champion.
When PREDICTION_LOG is set (lesson 06), every request to /v1/predict is also written to that file.

Endpoints:
    GET  /health         status, model name, version and alias
    POST /v1/predict     predictions for a batch of entries (version 1 of the API contract)
    POST /admin/reload   load MODEL_URI again; the old model answers until the new one is ready
    GET  /docs           the API contract as an interactive page (made by FastAPI)

Usage:
    uvicorn app:app --app-dir exercises/serve-model --port 8000
    MODEL_URI=models:/f1-points-finish/1 uvicorn app:app --app-dir exercises/serve-model --port 8000
    PREDICTION_LOG=logs/predictions.jsonl uvicorn app:app --app-dir exercises/serve-model --port 8000
"""

import json
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum

# Fail in seconds, not minutes, when the MLflow server is not running.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

MODEL_URI = os.environ.get("MODEL_URI", "models:/f1-points-finish@champion")
# The seven model inputs, all known before the race. See f1/data/README.md.
INPUTS = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob", "circuit_type"]
MAX_BATCH = 1000
# Lesson 06: the prediction log, one JSON line per entry. Off when PREDICTION_LOG is not set.
PREDICTION_LOG = os.environ.get("PREDICTION_LOG")
_log_lock = threading.Lock()


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


def log_request(request: Request, status: int, entries: list, served: dict | None = None,
                predictions=None, probabilities=None, errors: dict | None = None):
    """Append one line per entry to PREDICTION_LOG: the inputs, the answer and the model version.

    Rejected requests (422) are logged too, with the reason for each entry: the monitor needs them.
    The client can send X-Request-ID to find its request in the log later.
    """
    if not PREDICTION_LOG:
        return
    common = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request_id": request.headers.get("x-request-id") or uuid.uuid4().hex,
        "status": status,
        "model_name": served["model_name"] if served else None,
        "model_version": served["model_version"] if served else None,
    }
    lines = []
    for i, inputs in enumerate(entries):
        line = {**common, "entry": i, "inputs": inputs}
        if status == 200:
            line |= {"prediction": int(predictions[i]), "probability": round(float(probabilities[i]), 6)}
        else:
            line["errors"] = (errors or {}).get(i, [])
        lines.append(json.dumps(line))
    os.makedirs(os.path.dirname(PREDICTION_LOG) or ".", exist_ok=True)
    with _log_lock, open(PREDICTION_LOG, "a") as f:
        f.write("\n".join(lines) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, before the first request. If it fails, the API does not start.
    served = app.state.served = load_model(MODEL_URI)
    print(f"Serving {served['model_name']} version {served['model_version']} ({MODEL_URI})", flush=True)
    yield


app = FastAPI(title="f1-points-finish", version="1", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    """Answer 422 with one readable line per problem, for example 'entries.0.grid: Input should be ...'."""
    problems = [f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors()]
    if PREDICTION_LOG and request.url.path == "/v1/predict":
        body = exc.body if isinstance(exc.body, dict) else {}
        entries = body.get("entries") if isinstance(body.get("entries"), list) else [body]
        errors = {}  # entry number -> its problems, for example ["grid: Input should be ..."]
        for e in exc.errors():
            loc = [str(p) for p in e["loc"][1:]]
            if len(loc) >= 2 and loc[0] == "entries" and loc[1].isdigit():
                errors.setdefault(int(loc[1]), []).append(f"{'.'.join(loc[2:]) or 'entry'}: {e['msg']}")
            else:
                errors.setdefault(0, []).append(f"{'.'.join(loc) or 'body'}: {e['msg']}")
        log_request(request, 422, entries, served=getattr(app.state, "served", None), errors=errors)
    return JSONResponse(status_code=422, content={"detail": problems})


@app.get("/health", response_model=Health)
def health():
    served = app.state.served
    return Health(status="ok", **{k: served[k] for k in Health.model_fields if k != "status"})


@app.post("/v1/predict", response_model=PredictResponse)
def predict(request: PredictRequest, http: Request):
    # Read the model once: the whole batch is answered by one version, even during a reload.
    served = app.state.served
    model = served["model"]
    entries = [e.model_dump(mode="json") for e in request.entries]
    X = pd.DataFrame(entries)[served["columns"]]
    proba = model.predict_proba(X)
    # The same rule as model.predict(), without a second pass through the model.
    predictions = model.classes_.take(proba.argmax(axis=1))
    log_request(http, 200, entries, served, predictions, proba[:, 1])
    return PredictResponse(
        model_name=served["model_name"],
        model_version=served["model_version"],
        predictions=[int(p) for p in predictions],
        probabilities=[float(p) for p in proba[:, 1]],
    )


@app.post("/admin/reload")
def reload():
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
    print(f"Reloaded {MODEL_URI}: version {previous['model_version']} -> {served['model_version']}", flush=True)
    return {
        "model_name": served["model_name"],
        "previous_version": previous["model_version"],
        "model_version": served["model_version"],
        "alias": served["alias"],
        "loaded_at": served["loaded_at"],
    }
