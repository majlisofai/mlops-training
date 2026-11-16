"""Send the 24 races of 2025 to the API, one race (20 cars) per request, like a race dashboard.

The client sends only the seven model inputs. races.csv also holds driver_code and
driver_name, but the API does not need them, so the client does not send them: data
minimisation starts at the client. --with-driver-code sends driver_code too, to show that
the API refuses it.

Usage:
    python exercises/fairness/send_season.py
    python exercises/fairness/send_season.py --with-driver-code
    python exercises/fairness/send_season.py --api http://127.0.0.1:8001
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

import pandas as pd

INPUTS = ["grid", "quali_gap_ms", "driver_form", "team_form", "team_dnf_rate", "rain_prob", "circuit_type"]


def post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", default="f1/data/races.csv")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--with-driver-code", action="store_true", help="also send driver_code (the API refuses it)")
    args = parser.parse_args()

    races = pd.read_csv(args.csv)
    season = races[races["season"] == args.season]
    columns = INPUTS + (["driver_code"] if args.with_driver_code else [])
    sent = refused = 0
    for race_id, race in season.groupby("race_id", sort=True):
        entries = json.loads(race[columns].to_json(orient="records"))
        status, answer = post(f"{args.api}/v1/predict", {"entries": entries})
        if status != 200:
            refused += 1
            if refused == 1:
                detail = answer.get("detail", answer)
                print(f"{race_id}: HTTP {status}, {len(detail)} problems, first: {detail[0]}")
            continue
        sent += 1
        version = answer["model_version"]
    if sent:
        print(f"Sent {sent} races ({sent * 20} entries) to {args.api}; model version {version} answered.")
    if refused:
        print(f"{refused} races were refused.")
        sys.exit(1)


if __name__ == "__main__":
    main()
