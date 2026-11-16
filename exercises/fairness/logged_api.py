"""The lesson 05 API, plus a prediction log. You do not need to change this file.

It imports the API from exercises/serve-model/app.py unchanged and adds one thing: every
successful POST /v1/predict is written as one JSON line to PREDICTION_LOG
(default logs/predictions.jsonl). Monitoring needs this log later. The log is also a place
where personal data can leak, so it keeps only an allowlist of fields:

    logged_at, model_name, model_version, the seven model inputs per entry, predictions, probabilities

It does not log the client address, headers or any field outside the allowlist. The lesson 05
contract already refuses unknown fields such as driver_code (422), so they never reach the model
or the log. The allowlist is the second line of defence.

Usage:
    uvicorn logged_api:app --app-dir exercises/fairness --port 8000
    PREDICTION_LOG=logs/other.jsonl uvicorn logged_api:app --app-dir exercises/fairness --port 8000
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request, Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve-model"))
from app import INPUTS, app  # noqa: E402  the lesson 05 API, unchanged

PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))


def log_record(request_body: bytes, response_body: bytes) -> dict:
    """Build the log record from the allowlist only."""
    sent, answer = json.loads(request_body), json.loads(response_body)
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": answer["model_name"],
        "model_version": answer["model_version"],
        "entries": [{k: entry[k] for k in INPUTS} for entry in sent["entries"]],
        "predictions": answer["predictions"],
        "probabilities": [round(p, 4) for p in answer["probabilities"]],
    }


@app.middleware("http")
async def log_predictions(request: Request, call_next):
    if request.method != "POST" or request.url.path != "/v1/predict":
        return await call_next(request)
    request_body = await request.body()
    response = await call_next(request)
    response_body = b"".join([chunk async for chunk in response.body_iterator])
    if response.status_code == 200:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTION_LOG.open("a") as f:
            f.write(json.dumps(log_record(request_body, response_body)) + "\n")
    return Response(content=response_body, status_code=response.status_code,
                    headers=dict(response.headers), media_type=response.media_type)
