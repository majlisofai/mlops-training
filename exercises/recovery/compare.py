"""Compare f1-points-finish@champion with @challenger on the same held-out 2026 races. Exit 1 unless PROMOTE.

Both models predict the test window in comparison.yaml (2026 rounds 5 to 8, final labels).
compare.py measures accuracy, ROC AUC, the rookie and team_tier subgroups, the grid rule and
the accuracy per race, and runs the challenger gates. It recommends PROMOTE only when every
gate passes. Then it writes the gate report for the challenger in the lesson 04 format
(gate_report.json and gate_report.md) and points @candidate at the challenger, so that the
promotion takes the lesson 04 path: promote.py reads the report and needs an approver.

Usage:
    python exercises/recovery/compare.py
    python exercises/recovery/compare.py --comparison exercises/recovery/comparison.yaml
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

TARGET = "points_finish"
REPORT = Path(os.environ.get("GATE_REPORT", "gate_report.json"))
COMPARISON_MD = Path(os.environ.get("COMPARISON_REPORT", "comparison_report.md"))
BOOTSTRAP_SAMPLES = 2000


def load_test_window(comparison: dict) -> pd.DataFrame:
    """The 2026 entries of the test rounds: pre-race inputs, with the final label as the target."""
    paths = comparison["data"]
    provisional = pd.read_csv(paths["provisional_2026"])
    final = pd.read_csv(paths["final_2026"])[["race_id", "driver_code", "points_finish_final"]]
    rows = provisional.drop(columns=[TARGET]).merge(final, on=["race_id", "driver_code"], validate="one_to_one")
    rows = rows.rename(columns={"points_finish_final": TARGET})
    return rows[rows["round"].isin(comparison["test_rounds_2026"])].reset_index(drop=True)


def find(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        sys.exit(f"ERROR: {name} has no @{alias}. " + {
            "champion": "Run the lesson 04 pipeline first (step 1).",
            "challenger": "Run train_challenger.py first (step 7).",
        }[alias])


def measure(name: str, version, test: pd.DataFrame, comparison: dict) -> dict:
    """Load one model version and measure it on the test window."""
    uri = f"models:/{name}/{version.version}"
    model = mlflow.sklearn.load_model(uri)
    inputs = list(getattr(model, "feature_names_in_", []))
    result = {"version": version.version, "inputs": inputs, "run_id": version.run_id}
    missing = sorted(set(inputs) - set(comparison["api_inputs"]))
    if missing:  # needs a column that the pit wall does not send before the race
        result["error"] = f"needs inputs that the API does not accept: {', '.join(missing)}"
        return result
    proba = model.predict_proba(test[inputs])[:, 1]
    pred = (proba > 0.5).astype(int)
    correct = pred == test[TARGET].to_numpy()
    groups = {f"rookie ({int(test['rookie'].sum())})": test["rookie"].to_numpy()}
    for tier in ["top", "midfield", "backmarker"]:
        mask = (test["team_tier"] == tier).to_numpy()
        groups[f"{tier} ({int(mask.sum())})"] = mask
    result.update(
        predictions=pred,
        correct=correct,
        accuracy=float(correct.mean()),
        roc_auc=float(roc_auc_score(test[TARGET], proba)),
        points_share=float(pred.mean()),
        subgroups={g: float(correct[mask].mean()) for g, mask in groups.items()},
        per_race={race: float(correct[(test["race_id"] == race).to_numpy()].mean())
                  for race in sorted(test["race_id"].unique())},
    )
    return result


def bootstrap_gain(champion: dict, challenger: dict) -> tuple[float, float]:
    """90 % interval of the accuracy gain, resampling the test entries (paired: same entries for both)."""
    diff = challenger["correct"].astype(float) - champion["correct"].astype(float)
    rng = np.random.default_rng(2026)
    means = rng.choice(diff, size=(BOOTSTRAP_SAMPLES, len(diff)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def gate(name: str, description: str, passed: bool, measured: dict, detail: str = "") -> dict:
    """One gate in the lesson 04 gate report format."""
    return {"gate": name, "description": description, "outcome": "passed" if passed else "failed",
            "measured": {k: str(v) for k, v in measured.items()}, "detail": "" if passed else detail}


def run_gates(champion: dict, challenger: dict, train_rounds: str | None, grid_rule: float, comparison: dict):
    rules, test_rounds = comparison["rules"], comparison["test_rounds_2026"]
    gates = []

    inputs = challenger["inputs"]
    forbidden = sorted(set(inputs) & set(comparison["forbidden_features"]))
    gates.append(gate("no_forbidden_features", "The challenger needs no input that is known only after the race.",
                      not forbidden, {"inputs": len(inputs)},
                      f"the model uses after-race columns: {', '.join(forbidden)}"))
    gates.append(gate("inputs_match_api_contract", "The challenger needs exactly the seven inputs of API v1.",
                      sorted(inputs) == sorted(comparison["api_inputs"]), {"inputs": ",".join(inputs)},
                      f"inputs {inputs} are not the API v1 inputs {comparison['api_inputs']}"))

    if train_rounds is None:
        gates.append(gate("time_split", "The challenger trained only on 2026 races before the test window.",
                          False, {"train_2026_rounds": "unknown"},
                          "the challenger run has no parameter train_2026_rounds: nobody can tell what it saw"))
    else:
        rounds = [int(r) for r in train_rounds.split(",") if r]
        late = sorted(r for r in rounds if r >= min(test_rounds))
        gates.append(gate("time_split", "The challenger trained only on 2026 races before the test window.",
                          not late, {"train_2026_rounds": train_rounds or "none"},
                          f"trained on 2026 rounds {late}: the test window starts at round {min(test_rounds)}"))

    if "error" in challenger:
        gates.append(gate("predicts_0_or_1", "The challenger predicts 0 or 1 for every test entry.",
                          False, {}, challenger["error"]))
        return gates

    gates.append(gate("predicts_0_or_1", "The challenger predicts 0 or 1 for every test entry.",
                      set(np.unique(challenger["predictions"])) <= {0, 1},
                      {"rows": len(challenger["predictions"])}, "unexpected predictions"))

    margin = challenger["accuracy"] - grid_rule
    gates.append(gate("beats_grid_rule", "Accuracy on the test window beats the grid rule by the lesson 04 margin.",
                      margin >= rules["min_margin_over_grid_rule"] - 1e-9, {"margin": f"{margin:.3f}"},
                      f"accuracy {challenger['accuracy']:.3f} beats the grid rule ({grid_rule:.3f}) "
                      f"by less than {rules['min_margin_over_grid_rule']}"))

    gain = challenger["accuracy"] - champion["accuracy"]
    gates.append(gate("accuracy_gain", "Accuracy beats the champion by at least min_accuracy_gain.",
                      gain >= rules["min_accuracy_gain"] - 1e-9, {"gain": f"{gain:+.3f}"},
                      f"accuracy gain {gain:+.3f} is below {rules['min_accuracy_gain']:+.3f} "
                      f"(champion {champion['accuracy']:.3f}, challenger {challenger['accuracy']:.3f})"))

    drop = champion["roc_auc"] - challenger["roc_auc"]
    gates.append(gate("roc_auc_not_worse", "ROC AUC falls by at most max_roc_auc_drop.",
                      drop <= rules["max_roc_auc_drop"] + 1e-9, {"roc_auc_change": f"{-drop:+.3f}"},
                      f"roc_auc falls by {drop:.3f}, more than {rules['max_roc_auc_drop']} "
                      f"(champion {champion['roc_auc']:.3f}, challenger {challenger['roc_auc']:.3f})"))

    for label in ["rookie", "team_tier"]:
        names = [g for g in champion["subgroups"] if g.startswith("rookie") == (label == "rookie")]
        drops = {g: champion["subgroups"][g] - challenger["subgroups"][g] for g in names}
        worst = max(drops, key=drops.get)
        gates.append(gate(f"{label}_accuracy_not_worse",
                          f"Accuracy per {label} group falls by at most max_subgroup_drop.",
                          drops[worst] <= rules["max_subgroup_drop"] + 1e-9,
                          {"worst_change": f"{worst.split(' ')[0]} {0.0 - drops[worst]:+.3f}"},
                          f"accuracy for {worst} falls by {drops[worst]:.3f}, more than {rules['max_subgroup_drop']}"))
    return gates


def tables(champion: dict, challenger: dict, grid_rule: float, test: pd.DataFrame, comparison: dict) -> list[str]:
    """The comparison as Markdown table lines."""
    c, d = champion, challenger
    rules = comparison["rules"]
    head = f"| Measure | Champion v{c['version']} | Challenger v{d['version']} | Change | Rule |"
    lines = [head, "|---|---:|---:|---:|---|"]
    lines.append(f"| accuracy | {c['accuracy']:.3f} | {d['accuracy']:.3f} | {d['accuracy'] - c['accuracy']:+.3f} "
                 f"| gain at least {rules['min_accuracy_gain']:+.3f} |")
    lines.append(f"| roc_auc | {c['roc_auc']:.3f} | {d['roc_auc']:.3f} | {d['roc_auc'] - c['roc_auc']:+.3f} "
                 f"| drop at most {rules['max_roc_auc_drop']} |")
    lines.append(f"| margin over grid rule ({grid_rule:.3f}) | {c['accuracy'] - grid_rule:.3f} "
                 f"| {d['accuracy'] - grid_rule:.3f} | | at least {rules['min_margin_over_grid_rule']} |")
    for g in c["subgroups"]:
        lines.append(f"| accuracy {g} | {c['subgroups'][g]:.3f} | {d['subgroups'][g]:.3f} "
                     f"| {d['subgroups'][g] - c['subgroups'][g]:+.3f} | drop at most {rules['max_subgroup_drop']} |")
    lines.append(f"| predicted points share (actual {test[TARGET].mean():.2f}) | {c['points_share']:.2f} "
                 f"| {d['points_share']:.2f} | | |")
    races = list(c["per_race"])
    lines += ["", "| Accuracy per race | " + " | ".join(races) + " |", "|---|" + "---:|" * len(races)]
    for label, m in [(f"Champion v{c['version']}", c), (f"Challenger v{d['version']}", d)]:
        lines.append(f"| {label} | " + " | ".join(f"{m['per_race'][r]:.2f}" for r in races) + " |")
    return lines


def print_table(lines: list[str]):
    """Print Markdown table lines as aligned text."""
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines if not line.startswith("|---")]
    blocks, block = [], []
    for row in rows + [[""]]:
        if row == [""]:
            if block:
                blocks.append(block)
            block = []
        else:
            block.append(row)
    for block in blocks:
        widths = [max(len(r[i]) for r in block if i < len(r)) for i in range(max(len(r) for r in block))]
        for r in block:
            print("  " + "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in
                                   enumerate(zip(r, widths))).rstrip())
        print()


def write_reports(gates: list[dict], challenger: dict, name: str, recommendation: str, markdown: list[str]):
    counts = {o: sum(g["outcome"] == o for g in gates) for o in ("passed", "failed", "skipped", "error")}
    report = {
        "model": name,
        "version": challenger["version"],
        "run_id": challenger["run_id"],
        "gates_file": "exercises/recovery/compare.py",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "all_passed": counts["passed"] > 0 and counts["failed"] + counts["error"] == 0,
        "gates": gates,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    verdict = "passed" if report["all_passed"] else "BLOCKED"
    md = [f"### Challenger gates: {name} version {challenger['version']}: {verdict}", "",
          "| Gate | Result | Measured | Detail |", "|---|---|---|---|"]
    md += [f"| {g['gate']} | {'PASS' if g['outcome'] == 'passed' else 'FAIL'} | "
           f"{', '.join(f'{k} {v}' for k, v in g['measured'].items())} | {g['detail'].replace('|', '/')} |"
           for g in gates]
    REPORT.with_suffix(".md").write_text("\n".join(md) + "\n")
    full = markdown + ["", "\n".join(md), "", f"**{recommendation}**", ""]
    COMPARISON_MD.write_text("\n".join(full))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("\n".join(full) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--comparison", type=Path, default=Path("exercises/recovery/comparison.yaml"))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    comparison = yaml.safe_load(args.comparison.read_text())
    name = comparison["model_name"]
    test = load_test_window(comparison)
    grid_rule = float(((test["grid"] <= 10).astype(int) == test[TARGET]).mean())

    champion_v, challenger_v = find(client, name, "champion"), find(client, name, "challenger")
    if champion_v.version == challenger_v.version:
        sys.exit(f"ERROR: @champion and @challenger are both version {champion_v.version}. "
                 "Train a new challenger first (step 7).")
    champion = measure(name, champion_v, test, comparison)
    challenger = measure(name, challenger_v, test, comparison)
    if "error" in champion:
        sys.exit(f"ERROR: the champion, version {champion['version']}, {champion['error']}. Recover the service first.")
    train_rounds = client.get_run(challenger["run_id"]).data.params.get("train_2026_rounds")

    rounds = comparison["test_rounds_2026"]
    title = (f"Champion v{champion['version']} vs challenger v{challenger['version']} on 2026 rounds "
             f"{rounds[0]}-{rounds[-1]} ({len(test)} entries, final labels)")
    print(title + "\n")
    lines = [f"### {title}", ""]
    if "error" not in challenger:
        table = tables(champion, challenger, grid_rule, test, comparison)
        print_table(table)
        low, high = bootstrap_gain(champion, challenger)
        gain = challenger["accuracy"] - champion["accuracy"]
        interval = (f"Accuracy gain {gain:+.3f}, 90 % bootstrap interval [{low:+.3f}, {high:+.3f}] "
                    f"over the {len(test)} entries of {len(rounds)} races.")
        print(interval + "\n")
        lines += table + ["", interval]

    gates = run_gates(champion, challenger, train_rounds, grid_rule, comparison)
    print(f"Gates for the challenger (version {challenger['version']}):")
    for g in gates:
        measured = ", ".join(f"{k} {v}" for k, v in g["measured"].items())
        print(f"  {g['outcome'].upper():<7} {g['gate']:<30} {measured}")
    failed = [g for g in gates if g["outcome"] != "passed"]
    if failed:
        recommendation = (f"RECOMMENDATION: KEEP the champion (version {champion['version']}). "
                          f"The challenger (version {challenger['version']}) failed {len(failed)} of {len(gates)} gates.")
        print("\n" + recommendation)
        for g in failed:
            print(f"- {g['gate']}: {g['detail']}")
    else:
        recommendation = (f"RECOMMENDATION: PROMOTE the challenger (version {challenger['version']}). "
                          f"It passed all {len(gates)} gates.")
        print("\n" + recommendation)

    write_reports(gates, challenger, name, recommendation, lines)
    client.set_registered_model_alias(name, "candidate", challenger["version"])
    print(f"\nWrote {REPORT} and {COMPARISON_MD}. @candidate -> version {challenger['version']}.")
    print("Promotion takes the lesson 04 path: python exercises/quality-gates/promote.py --approver \"<your name>\"")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
