"""Load test for the f1-points-finish API.

Each simulated user asks for the predictions of one race at a time: the 20 cars of a
real 2025 race in one POST /v1/predict. It waits 0.5 to 1.5 seconds between requests,
so 10 users send about 10 requests per second. Now and then it calls GET /health.

Usage:
    locust -f exercises/serve-model/locustfile.py --host http://127.0.0.1:8000
        (web UI on port 8089)
    locust -f exercises/serve-model/locustfile.py --host http://127.0.0.1:8000 \
        --headless --users 10 --spawn-rate 5 --run-time 30s --csv results/locust --only-summary
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

from locust import HttpUser, between, task

RACES = Path(__file__).parents[2] / "f1" / "data" / "races.csv"
INPUTS = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob", "circuit_type"]
TYPES = {"grid": int, "quali_gap_ms": int, "circuit_type": str}


def load_races() -> list[list[dict]]:
    """The 24 races of 2025. Each race is a list of 20 entries with the seven model inputs."""
    races = defaultdict(list)
    with RACES.open() as f:
        for row in csv.DictReader(f):
            if row["season"] == "2025":
                races[row["race_id"]].append({c: TYPES.get(c, float)(row[c]) for c in INPUTS})
    return list(races.values())


RACE_ENTRIES = load_races()


class RaceEngineer(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        # Lesson 09: the secured API needs a client key. Send it when API_KEY is set.
        if os.environ.get("API_KEY"):
            self.client.headers["X-API-Key"] = os.environ["API_KEY"]

    @task(10)
    def predict_a_race(self):
        race = random.choice(RACE_ENTRIES)
        with self.client.post("/v1/predict", json={"entries": race}, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}: {response.text[:200]}")
            elif len(response.json()["predictions"]) != len(race):
                response.failure(f"{len(race)} entries, {len(response.json()['predictions'])} predictions")

    @task(1)
    def health(self):
        self.client.get("/health")
