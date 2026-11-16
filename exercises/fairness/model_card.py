"""Generate the model card of f1-points-finish@champion from MLflow, and log it to the model's run.

The card follows the sections of Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019).
What can be measured comes from the machines: the registry (version, aliases, who promoted it,
the lesson 04 gates), the run (parameters, metrics, data version, git commit), the fairness
report that fairness_report.py logged to the run, and the fairness gate report. What people
must write and sign comes from model_card.yaml: intended use, out-of-scope use, owner, approver,
the decision. Placeholders that are still <...> are listed at the top of the card.

Usage:
    python exercises/fairness/model_card.py
    MODEL_URI=models:/f1-points-finish/2 python exercises/fairness/model_card.py
    python exercises/fairness/model_card.py --out fairness_report/model_card.md --no-log
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from fairness_metrics import load_criteria, model_uri, resolve_version  # noqa: E402

METRIC_NAMES = {"accuracy": "Accuracy", "selection_rate": "Selection rate",
                "false_positive_rate": "False positive rate", "false_negative_rate": "False negative rate"}


def placeholders(value, path="") -> list[str]:
    """Every field of model_card.yaml that still holds a <...> placeholder."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in placeholders(v, f"{path}.{k}".lstrip("."))]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in placeholders(v, f"{path}[{i}]")]
    return [path] if isinstance(value, str) and value.startswith("<") and value.endswith(">") else []


def fairness_report(run_id: str) -> dict | None:
    try:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="fairness/fairness_report.json")
    except Exception:
        return None
    return json.loads(Path(path).read_text())


def gate_report(path: Path, version: str) -> tuple[dict | None, str]:
    if not path.exists():
        return None, f"No fairness gate report at `{path}`. Run the fairness and privacy gates."
    report = json.loads(path.read_text())
    if str(report.get("version")) != str(version):
        return None, f"`{path}` is for version {report.get('version')}, not version {version}. Run the gates again."
    return report, ""


def bullets(items) -> list[str]:
    return [f"- {item}" for item in items]


def training_distribution(criteria: dict, train_seasons: str) -> list[str]:
    races = pd.read_csv(criteria["data"])
    first, last = (int(s) for s in train_seasons.split("-"))
    train = races[races["season"].between(first, last)]
    rows = []
    for season, group in train.groupby("season"):
        rookies = group[group["rookie"]]
        rows.append(f"| {season} | {len(group)} | {len(rookies)} | {rookies['driver_code'].nunique()} | "
                    + (f"{rookies['points_finish'].mean():.3f}" if len(rookies) else "-")
                    + f" | {group.loc[~group['rookie'], 'points_finish'].mean():.3f} |")
    lines = ["| Season | Entries | Rookie entries | Rookie drivers | Rookies scoring points | Others scoring points |",
             "|---|---:|---:|---:|---:|---:|"] + rows
    tiers = train["team_tier"].value_counts().to_dict()
    lines += ["", f"Team tiers in training: " + ", ".join(f"{k} {v}" for k, v in sorted(tiers.items())) + " entries."]
    return lines


def card(version, run, human, fair, gates, gates_note, criteria, uri) -> str:
    client = MlflowClient()
    model = client.get_registered_model(version.name)
    aliases = sorted(a for a, v in model.aliases.items() if str(v) == str(version.version))
    p, m, t, vt = run.data.params, run.data.metrics, run.data.tags, version.tags
    acc = human["accountability"]
    todo = placeholders(human)
    trained = datetime.fromtimestamp(run.info.start_time / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    hyper = {k: v for k, v in p.items() if k not in ("model_type", "features", "train_seasons", "test_season")}
    lines = [f"# Model card: {version.name} version {version.version}", ""]
    lines += [f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `exercises/fairness/model_card.py` "
              f"from `{uri}`. Sections follow Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019).", ""]

    # ---- release status first: what a reviewer needs to see
    lab04 = {k.removeprefix("gate."): v for k, v in vt.items() if k.startswith("gate.")}
    fair_status = "MISSING" if gates is None else ("passed" if gates["all_passed"] else "BLOCKED")
    lines += ["## Release status", "",
              "| Check | Result |", "|---|---|",
              f"| Lesson 04 quality gates | {vt.get('gates', 'not run through promote.py')} |",
              f"| Fairness and privacy gates | {fair_status}" + (f", {gates['warnings']} with warnings" if gates and gates["warnings"] else "") + " |",
              f"| Fairness report logged to the run | {'yes' if fair else 'MISSING: run fairness_report.py'} |",
              f"| Human decision | {acc['decision']} |",
              f"| Approver | {acc['approver']}, {acc['decision_date']} |", ""]
    if todo:
        lines += ["> **Not ready for review.** These fields of `model_card.yaml` are still placeholders: "
                  + ", ".join(f"`{x}`" for x in todo) + ".", ""]
    if gates_note:
        lines += [f"> {gates_note}", ""]

    # ---- 1 model details
    lines += ["## Model details", "",
              f"- **Developed by:** {acc['owner']} (proposed by `{t.get('proposed_by', vt.get('proposed_by', '?'))}`)",
              f"- **Model date:** trained {trained}",
              f"- **Model version:** {version.name} version {version.version}"
              + (f", aliases {', '.join('@' + a for a in aliases)}" if aliases else ", no alias")
              + f"; previous champion: {vt.get('previous_champion', 'none recorded')}",
              f"- **Promoted by:** {vt.get('promoted_by', 'not promoted')} at {vt.get('promoted_at', '-')}",
              f"- **Model type:** scikit-learn pipeline, `{p.get('model_type', '?')}`, hyperparameters "
              + (", ".join(f"{k}={v}" for k, v in sorted(hyper.items())) or "none"),
              f"- **Inputs:** {p.get('features', '?').replace(',', ', ')}",
              f"- **Training data:** seasons {p.get('train_seasons', '?')} of `{t.get('data.path', '?')}`, "
              f"data version (md5) `{t.get('data.version', '?')}`",
              f"- **Code:** git commit `{t.get('git.commit', '?')}`, MLflow run `{run.info.run_id}`",
              f"- **License:** {human['license']}",
              f"- **Questions and complaints:** {acc['contact']}", ""]

    # ---- 2 intended use
    iu = human["intended_use"]
    lines += ["## Intended use", "", "**Primary intended uses**", ""] + bullets(iu["primary_uses"])
    lines += ["", "**Primary intended users**", ""] + bullets(iu["primary_users"])
    lines += ["", "**Out-of-scope uses**", ""] + bullets(iu["out_of_scope"]) + [""]

    # ---- 3 factors
    lines += ["## Factors", ""] + bullets(human["factors"]["relevant"]) + ["", human["factors"]["note"], ""]

    # ---- 4 metrics
    lines += ["## Metrics", "",
              "- **Performance measures:** accuracy and ROC AUC on all entries; per subgroup: accuracy, selection rate "
              "(share predicted to score), false positive rate (predicted points, did not score) and false negative rate "
              "(predicted no points, scored).",
              "- **Decision threshold:** a car is predicted to score when the predicted probability is at least 0.5, for every group.",
              "- **Fairness measures:** the rookie gap of the false positive rate and of the false negative rate "
              "(rookie rate minus everyone else's). Equal false negative rates correspond to equality of opportunity; "
              "equal false negative and false positive rates to equalized odds (Mitchell et al., section 4.4).",
              "- **Variation:** bootstrap 95 % intervals per group (500 resamples of entries). They are too narrow for "
              "groups of a few drivers, because entries of the same driver are not independent.", ""]

    # ---- 5 evaluation data, 6 training data
    entries = fair["entries"] if fair else "?"
    lines += ["## Evaluation data", "",
              f"- **Dataset:** season {criteria['test_season']} of `{criteria['data']}`, {entries} entries "
              "(24 races, 20 cars). The model never saw it in training.",
              "- **Motivation:** the most recent complete season: the closest to the next races.",
              "- **Preprocessing:** none outside the model; the pipeline one-hot encodes circuit_type itself.", "",
              "## Training data", "", f"Seasons {p.get('train_seasons', '?')}. Distribution over the factors:", ""]
    lines += training_distribution(criteria, p.get("train_seasons", "2021-2024")) + [""]

    # ---- 7 quantitative analyses
    lines += ["## Quantitative analyses", "", "**Overall** (2025 holdout, from the training run)", "",
              "| Accuracy | F1 | ROC AUC |", "|---:|---:|---:|",
              f"| {m.get('accuracy', float('nan')):.3f} | {m.get('f1', float('nan')):.3f} | {m.get('roc_auc', float('nan')):.3f} |", ""]
    if fair:
        lines += ["**Unitary results** (value, bootstrap 95 % interval)", ""]
        for column, rows in fair["by_group"].items():
            lines += [f"*{column}*", "", "| Group | Entries | Drivers | " + " | ".join(METRIC_NAMES.values()) + " |",
                      "|---|---:|---:|" + "---:|" * len(METRIC_NAMES)]
            for r in rows:
                lines.append(f"| {r['group']} | {int(r['entries'])} | {int(r['drivers'])} | " + " | ".join(
                    f"{r[k]:.3f} ({r[k + '_low']:.2f}-{r[k + '_high']:.2f})" for k in METRIC_NAMES) + " |")
            lines.append("")
        lines += [f"*rookie x team_tier* (cells with fewer than {criteria['privacy']['min_drivers_per_published_cell']} "
                  "drivers are suppressed: a group of one driver is a person)", "",
                  "| rookie | team_tier | Entries | Drivers | Accuracy | False positive rate | False negative rate |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for r in fair["intersectional"]:
            vals = ["suppressed", "", ""] if pd.isna(r["accuracy"]) else [
                f"{r['accuracy']:.3f}", f"{r['false_positive_rate']:.3f}", f"{r['false_negative_rate']:.3f}"]
            lines.append(f"| {r['rookie']} | {r['team_tier']} | {r['entries']} | {r['drivers']} | " + " | ".join(vals) + " |")
        g, grid = fair["rookie_gaps"], fair["grid_rule_rookie_gaps"]
        lines += ["", "**Rookie gaps** (rookie minus everyone else; positive = worse for rookies)", "",
                  "| Gap | This model | Grid rule | Limit (fairness.yaml) |", "|---|---:|---:|---:|",
                  f"| False positive rate | {g['fpr_gap']:+.3f} | {grid['fpr_gap']:+.3f} | {criteria['fairness']['max_fpr_gap']} |",
                  f"| False negative rate | {g['fnr_gap']:+.3f} | {grid['fnr_gap']:+.3f} | {criteria['fairness']['max_fnr_gap']} |",
                  f"| Accuracy | {g['accuracy_gap']:+.3f} | {grid['accuracy_gap']:+.3f} | lesson 04: rookie accuracy >= 0.60 |", ""]
        lines += ["![Metrics per subgroup](fairness_by_group.png)", ""]
    else:
        lines += ["> No fairness report in the run. Run `exercises/fairness/fairness_report.py` first.", ""]

    lines += ["**Gates**", "", "| Gate | Result | Measured | Warning |", "|---|---|---|---|"]
    lines += [f"| lesson 04: {name} | {outcome} | | |" for name, outcome in lab04.items()]
    if gates:
        for gate in gates["gates"]:
            measured = ", ".join(f"{k} {v}" for k, v in gate["measured"].items())
            lines.append(f"| {gate['gate']} | {gate['outcome']} | {measured} | {gate['warning']} |")
    lines.append("")

    # ---- 8 ethical considerations, 9 caveats
    lines += ["## Ethical considerations", ""] + bullets(human["ethical_considerations"]) + [""]
    caveats = list(human["caveats"]) + (fair["warnings"] if fair else [])
    lines += ["## Caveats and recommendations", ""] + bullets(caveats) + [""]

    # ---- accountability (not a Mitchell et al. section: the course adds it for the release dossier)
    lines += ["## Accountability and sign-off", "",
              "| Role | Who |", "|---|---|",
              f"| Model owner | {acc['owner']} |",
              f"| Proposed by | {t.get('proposed_by', '?')} |",
              f"| Promoted to @champion by | {vt.get('promoted_by', 'not promoted')} |",
              f"| Release approver | {acc['approver']} |",
              f"| Decision | {acc['decision']} ({acc['decision_date']}) |", ""]
    # "<...>" placeholders would vanish in rendered Markdown (they look like HTML tags): escape "<".
    return "\n".join(line.replace("<", "&lt;") for line in lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--human", type=Path, default=HERE / "model_card.yaml")
    parser.add_argument("--gate-report", type=Path, default=Path(os.environ.get("FAIRNESS_GATE_REPORT", "fairness_gate_report.json")))
    parser.add_argument("--out", type=Path, default=Path("fairness_report/model_card.md"))
    parser.add_argument("--no-log", action="store_true", help="do not log the card to the model's MLflow run")
    args = parser.parse_args()

    criteria = load_criteria()
    uri = model_uri()
    version = resolve_version(uri)
    run = MlflowClient().get_run(version.run_id)
    human = yaml.safe_load(args.human.read_text())
    fair = fairness_report(version.run_id)
    gates, note = gate_report(args.gate_report, version.version)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(card(version, run, human, fair, gates, note, criteria, uri))
    missing = placeholders(human)
    print(f"Wrote {args.out} for {version.name} version {version.version}.")
    print(f"Fairness and privacy gates: {'MISSING' if gates is None else ('passed' if gates['all_passed'] else 'BLOCKED')}. "
          f"Placeholders still to fill in model_card.yaml: {len(missing)}.")
    if not args.no_log:
        client = MlflowClient()
        client.log_artifact(version.run_id, str(args.out), artifact_path="model_card")
        client.set_model_version_tag(version.name, version.version, "model_card",
                                     f"runs:/{version.run_id}/model_card/{args.out.name}")
        if gates is not None:
            client.set_model_version_tag(version.name, version.version, "fairness_gates",
                                         "passed" if gates["all_passed"] else "BLOCKED")
        print(f"Logged it to run {version.run_id} under model_card/ and tagged version {version.version} with model_card.")


if __name__ == "__main__":
    main()
