"""Drift Detective: send a faulty window to the API. The detective names the fault with the monitor.

The trainer (or your partner) runs inject.py. It sends a window with one fault, or none, and
prints only how many entries it sent. The detective then runs monitor.py and reads the log, and
names the fault. The answer key is in solution/ANSWER_KEY.md: do not open it while you play.

Usage:
    python exercises/monitoring/inject.py --list
    python exercises/monitoring/inject.py <fault>
    python exercises/monitoring/inject.py --random      the fault is written to logs/incident_answer.txt
"""

import argparse
import csv
import os
import random
from pathlib import Path

from replay import DATA, load_window, races_from_csv, send_window

LABELS = Path("logs/labels_latest.csv")


def units_change():
    """The timing feed 'fixed' its 422 errors: it now sends quali_gap_ms as whole seconds."""
    requests = load_window("reference")
    for req in requests:
        for e in req["entries"]:
            e["quali_gap_ms"] = round(e["quali_gap_ms"] / 1000)
    return requests, None


def stuck_feature():
    """The form job stopped. The feature store fills driver_form with its default, 10.0."""
    requests = load_window("reference")
    for req in requests:
        for e in req["entries"]:
            e["driver_form"] = 10.0
    return requests, None


def category_renamed():
    """The circuit database renamed high_speed to high-speed."""
    requests = load_window("reference")
    for req in requests:
        for e in req["entries"]:
            if e["circuit_type"] == "high_speed":
                e["circuit_type"] = "high-speed"
    return requests, None


def label_delay():
    """2026 is replayed, but the stewards' feed stalled: races 4 to 8 have no final classification yet."""
    requests = load_window("season_2026")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with (DATA / "season_2026_final.csv").open() as f, LABELS.open("w", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["race_id"][-2:]) >= 4:
                row["final_position"] = row["points_finish_final"] = ""
            writer.writerow(row)
    return requests, str(LABELS)


def normal_weekend():
    """No fault: four ordinary races, 2025 rounds 17 to 20. A good monitor stays quiet."""
    keep = lambda row: row["season"] == "2025" and 17 <= int(row["round"]) <= 20  # noqa: E731
    return races_from_csv(DATA / "races.csv", keep), None


FAULTS = {f.__name__: f for f in [units_change, stuck_feature, category_renamed, label_delay, normal_weekend]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fault", nargs="?", choices=sorted(FAULTS))
    parser.add_argument("--random", action="store_true", help="pick a fault at random and keep it secret")
    parser.add_argument("--list", action="store_true", help="list the faults")
    parser.add_argument("--api", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()

    if args.list:
        for name, fault in FAULTS.items():
            print(f"{name:<18} {fault.__doc__}")
        return
    if args.random:
        args.fault = random.choice(sorted(FAULTS))
        Path("logs").mkdir(exist_ok=True)
        Path("logs/incident_answer.txt").write_text(args.fault + "\n")
    if not args.fault:
        parser.error("name a fault, or use --random or --list")

    # The fault changes what the API receives. The recomputed features stay clean: the training
    # pipeline computes them from the source data, which has no fault.
    requests, labels = FAULTS[args.fault]()
    # The window is called "incident", so the monitor output does not give the fault away.
    result = send_window("incident", requests, args.api.rstrip("/"), labels=labels, quiet=True)
    print(f"Incident sent: {len(requests)} requests, {result['answered'] + result['rejected']} entries. "
          "Detective: run the monitor.")


if __name__ == "__main__":
    main()
