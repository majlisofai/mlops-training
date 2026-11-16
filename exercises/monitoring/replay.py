"""Send a monitoring window to the running API, the way the pit wall would during a race weekend.

Each race is one request of 20 entries, with an X-Request-ID header, so the monitor can find the
requests in the prediction log. replay.py also writes logs/replay_manifest.json: which requests
belong to this window, a key for each entry (race and driver, when known) and the features that
the training pipeline computes for the same entries ("recomputed"). The monitor reads it.

Windows:
    reference     the last four races of 2025 (f1/data/scenarios/reference_window.json)
    wet_race      a storm forecast at one race (f1/data/scenarios/wet_race_window.json)
    serving_skew  a new timing feed (f1/data/scenarios/serving_skew_window.json)
    season_2026   the first eight races of 2026 (f1/data/season_2026_provisional.csv)

Usage:
    python exercises/monitoring/replay.py reference
    python exercises/monitoring/replay.py wet_race --api http://127.0.0.1:8001
"""

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "f1" / "data"
INPUTS = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob", "circuit_type"]
CARS_PER_RACE = 20
MANIFEST = Path("logs/replay_manifest.json")


def races_from_csv(path: Path, keep=lambda row: True) -> list[dict]:
    """One request per race: the model inputs of its 20 cars, keyed by race_id/driver_code."""
    types = {"grid": int, "quali_gap_ms": int, "circuit_type": str}
    races: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if not keep(row):
                continue
            race = races.setdefault(row["race_id"], {"entries": [], "keys": []})
            race["entries"].append({c: types.get(c, float)(row[c]) for c in INPUTS})
            race["keys"].append(f"{row['race_id']}/{row['driver_code']}")
    for race in races.values():
        race["recomputed"] = [dict(e) for e in race["entries"]]
    return list(races.values())


def races_from_scenario(path: Path, served: str = "rows", recomputed: str = "rows") -> list[dict]:
    """Cut a scenario file into races of 20 cars. The scenario rows have no race or driver names."""
    scenario = json.loads(path.read_text())
    sent, truth = scenario[served], scenario[recomputed]
    requests = []
    for start in range(0, len(sent), CARS_PER_RACE):
        stop = start + CARS_PER_RACE
        requests.append({
            "entries": sent[start:stop],
            "keys": [f"{path.stem}#{i}" for i in range(start, min(stop, len(sent)))],
            "recomputed": [dict(row) for row in truth[start:stop]],  # a copy: inject.py changes entries
        })
    return requests


def load_window(name: str) -> list[dict]:
    if name == "reference":
        return races_from_scenario(DATA / "scenarios" / "reference_window.json")
    if name == "wet_race":
        return races_from_scenario(DATA / "scenarios" / "wet_race_window.json")
    if name == "serving_skew":
        # served = what the new feed sends; recomputed = what the training pipeline computes.
        return races_from_scenario(DATA / "scenarios" / "serving_skew_window.json", "served", "recomputed")
    if name == "season_2026":
        return races_from_csv(DATA / "season_2026_provisional.csv")
    raise SystemExit(f"unknown window {name!r}. Choose one of: reference, wet_race, serving_skew, season_2026")


def post(api: str, path: str, payload: dict | None, request_id: str | None = None) -> tuple[int, dict]:
    """POST (or GET when payload is None) and return the status and the JSON answer, also for 4xx."""
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["X-Request-ID"] = request_id
    req = urllib.request.Request(api + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def send_window(window: str, requests: list[dict], api: str, labels: str | None = None,
                manifest: Path = MANIFEST, quiet: bool = False) -> dict:
    """Send each request to /v1/predict, print what came back and write the manifest."""
    try:
        _, health = post(api, "/health", None)
    except OSError as err:
        raise SystemExit(f"The API does not answer at {api}: {err}\nStart it first (step 3).")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    say = (lambda *a: None) if quiet else print
    say(f"Replaying {window} to {api} (serving {health['model_name']} version {health['model_version']})")

    answered = rejected = 0
    for n, req in enumerate(requests, start=1):
        req["request_id"] = f"{window}-{stamp}-{n}"
        status, body = post(api, "/v1/predict", {"entries": req["entries"]}, req["request_id"])
        size = len(req["entries"])
        if status == 200:
            answered += size
            say(f"  {req['request_id']}: {size} entries -> 200, version {body['model_version']}, "
                f"{sum(body['predictions'])} predicted in the points")
        else:
            rejected += size
            detail = body.get("detail", body)
            first = detail[0] if isinstance(detail, list) and detail else detail
            more = f" (and {len(detail) - 1} more problems)" if isinstance(detail, list) and len(detail) > 1 else ""
            say(f"  {req['request_id']}: {size} entries -> {status}: {first}{more}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "window": window,
        "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api": api,
        "labels": labels,
        "requests": [{k: r[k] for k in ("request_id", "keys", "recomputed")} for r in requests],
    }, indent=1))
    total = answered + rejected
    say(f"Replayed {window}: {len(requests)} requests, {total} entries: {answered} answered, {rejected} rejected.")
    say(f"Manifest: {manifest}")
    return {"answered": answered, "rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("window", help="reference, wet_race, serving_skew or season_2026")
    parser.add_argument("--api", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    send_window(args.window, load_window(args.window), args.api.rstrip("/"), manifest=args.manifest)


if __name__ == "__main__":
    sys.exit(main())
