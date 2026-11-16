"""Measure the model on each reporting subgroup of 2025 and write the evidence for the dossier.

The lesson 04 gate checked rookies only. This check measures every subgroup that
f1/data/README.md marks "subgroup for reporting" (rookie and team_tier): accuracy, the share
of entries predicted to score points (selection rate) and, among the entries that did score,
the share the model found (true positive rate). Each subgroup must reach min_accuracy in
release.yaml. The result goes to the fairness_report slot of release.yaml, where lesson 08's
fairness report can take its place: accept.py reads only model_name, model_version, passed
and checks.

Usage:
    python exercises/acceptance/subgroup_check.py
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

HERE = Path(__file__).parent
TARGET = "points_finish"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, default=HERE / "release.yaml")
    parser.add_argument("--csv", type=Path, default=Path("f1/data/races.csv"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    release = yaml.safe_load(args.release.read_text())
    rules = release["subgroups"]

    version = MlflowClient().get_model_version_by_alias(release["model_name"], release["alias"])
    uri = f"models:/{release['model_name']}/{version.version}"
    model = mlflow.sklearn.load_model(uri)
    inputs = mlflow.models.get_model_info(uri).signature.inputs.input_names()
    holdout = pd.read_csv(args.csv).query("season == 2025").copy()
    holdout["pred"] = model.predict(holdout[inputs])

    checks = []
    for column in rules["columns"]:
        for group, rows in holdout.groupby(column):
            scored = rows[rows[TARGET] == 1]
            accuracy = float((rows["pred"] == rows[TARGET]).mean())
            checks.append(
                {
                    "name": f"{column}={str(group).lower()}",
                    "entries": len(rows),
                    "accuracy": round(accuracy, 3),
                    "selection_rate": round(float(rows["pred"].mean()), 3),
                    "true_positive_rate": round(float(scored["pred"].mean()), 3) if len(scored) else None,
                    "threshold": rules["min_accuracy"],
                    "passed": accuracy >= rules["min_accuracy"],
                }
            )
    report = {
        "model_name": release["model_name"],
        "model_version": str(version.version),
        "produced_by": "exercises/acceptance/subgroup_check.py",
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": f"{args.csv}, season 2025 ({len(holdout)} entries)",
        "rule": f"accuracy of every subgroup of {', '.join(rules['columns'])} >= {rules['min_accuracy']}",
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    out = Path(release["evidence"]["fairness_report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Subgroups of {release['model_name']} version {version.version} on 2025:")
    print(f"  {'subgroup':<26} {'entries':>7} {'accuracy':>9} {'selected':>9} {'TPR':>6}")
    for c in checks:
        tpr = "-" if c["true_positive_rate"] is None else f"{c['true_positive_rate']:.3f}"
        print(
            f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']:<20} {c['entries']:>7} {c['accuracy']:>9.3f} "
            f"{c['selection_rate']:>9.3f} {tpr:>6}"
        )
    print(f"{'SUBGROUPS OK' if report['passed'] else 'SUBGROUPS FAILED'}: {report['rule']}. Written to {out}.")


if __name__ == "__main__":
    main()
