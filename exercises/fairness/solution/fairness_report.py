"""Fairness report: per-subgroup metrics of f1-points-finish@champion on the 2025 holdout.

Reference solution. Try exercises/fairness/fairness_report.py first.

For each subgroup column in fairness.yaml (rookie, team_tier, circuit_type) it measures
accuracy, selection rate, false positive rate and false negative rate per group, with
bootstrap 95 % intervals, and the gaps between the groups. It writes a table, a chart and a
JSON file to fairness_report/, and logs them to the model's MLflow run under fairness/.

Usage:
    python exercises/fairness/solution/fairness_report.py
    MODEL_URI=models:/f1-points-finish@candidate python exercises/fairness/solution/fairness_report.py
    python exercises/fairness/solution/fairness_report.py --no-log
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # exercises/fairness

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from fairlearn.metrics import MetricFrame  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from fairness_metrics import (  # noqa: E402
    METRICS, SHORT, TARGET, grid_rule, load_criteria, load_holdout, model_uri, resolve_version, rookie_gaps,
)

N_BOOT = 500          # bootstrap resamples for the 95 % intervals
CI = [0.025, 0.975]


# ---------------------------------------------------------------- the three measurements

def subgroup_frame(holdout: pd.DataFrame, predictions: pd.Series, column: str) -> MetricFrame:
    """A fairlearn MetricFrame: the four METRICS for every group of `column`, with intervals."""
    return MetricFrame(
        metrics=METRICS,
        y_true=holdout[TARGET],
        y_pred=predictions,
        sensitive_features=holdout[column],
        n_boot=N_BOOT,
        ci_quantiles=CI,
        random_state=0,
    )


def between_group_gaps(frame: MetricFrame) -> pd.DataFrame:
    """For every metric: the largest difference between two groups, and smallest / largest."""
    return pd.DataFrame({
        "difference": frame.difference(method="between_groups"),
        "ratio": frame.ratio(method="between_groups"),
    })


def drivers_per_group(holdout: pd.DataFrame, column: str) -> pd.Series:
    """How many different drivers each group holds. driver_code is counted, never used as an input."""
    return holdout.groupby(column)["driver_code"].nunique()


# ---------------------------------------------------------------- the report around them

def group_table(holdout, predictions, column) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = subgroup_frame(holdout, predictions, column)
    table = frame.by_group.copy()
    low, high = frame.by_group_ci
    for metric in METRICS:
        table[f"{metric}_low"], table[f"{metric}_high"] = low[metric], high[metric]
    counts = holdout.groupby(column)[TARGET].agg(entries="size", scorers="sum")
    table = counts.assign(non_scorers=counts.entries - counts.scorers,
                          drivers=drivers_per_group(holdout, column)).join(table)
    table.index = table.index.astype(str)
    return table, between_group_gaps(frame)


def intersectional(holdout, predictions, criteria) -> pd.DataFrame:
    """rookie x team_tier. Cells with too few drivers are suppressed: a group of one is a person."""
    minimum = criteria["privacy"]["min_drivers_per_published_cell"]
    rows = []
    for (rookie, tier), cell in holdout.groupby(["rookie", "team_tier"]):
        drivers = cell["driver_code"].nunique()
        row = {"rookie": rookie, "team_tier": tier, "entries": len(cell), "drivers": drivers}
        if drivers >= minimum:
            row.update({m: float(fn(cell[TARGET], predictions[cell.index])) for m, fn in METRICS.items()})
        else:
            row.update({m: None for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def warnings_for(tables: dict, criteria: dict) -> list[str]:
    f = criteria["fairness"]
    found = []
    for column, table in tables.items():
        for group, row in table.iterrows():
            where = f"{column}={group}"
            if row.drivers < f["warn_group_drivers"]:
                found.append(f"{where}: {int(row.entries)} entries from only {int(row.drivers)} drivers. "
                             "The result describes those drivers, not a population.")
            if row.non_scorers < f["warn_rate_base"]:
                found.append(f"{where}: the false positive rate rests on {int(row.non_scorers)} entries that did not score.")
            if row.scorers < f["warn_rate_base"]:
                found.append(f"{where}: the false negative rate rests on {int(row.scorers)} entries that scored.")
    return found


def chart(tables: dict, overall: dict, title: str, path: Path):
    """Small multiples: one row per subgroup column, one column per metric, 95 % intervals."""
    ink, muted, bar = "#0b0b0b", "#52514e", "#2a78d6"
    columns = list(tables)
    fig, axes = plt.subplots(len(columns), len(METRICS), figsize=(13, 2.0 + 1.25 * len(columns) * 1.6),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for r, column in enumerate(columns):
        table = tables[column]
        labels = [f"{column}={g}\n{int(row.entries)} entries, {int(row.drivers)} drivers" for g, row in table.iterrows()]
        for c, metric in enumerate(METRICS):
            ax = axes[r][c]
            ax.set_facecolor("#fcfcfb")
            values = table[metric].to_numpy()
            err = [values - table[f"{metric}_low"].to_numpy(), table[f"{metric}_high"].to_numpy() - values]
            ys = range(len(table))
            ax.barh(ys, values, height=0.55, color=bar, xerr=err, error_kw={"ecolor": muted, "elinewidth": 1, "capsize": 3})
            ax.axvline(overall[metric], color=muted, linestyle="--", linewidth=1)
            for y, v, high in zip(ys, values, table[f"{metric}_high"].to_numpy()):
                ax.text(max(v, high) + 0.02, y, f"{v:.2f}", va="center", fontsize=8, color=ink)
            ax.set_yticks(list(ys), labels if c == 0 else [""] * len(table), fontsize=8, color=ink)
            ax.set_xlim(0, 1.2)
            ax.invert_yaxis()
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=muted, labelsize=8)
            if r == 0:
                ax.set_title(metric.replace("_", " "), fontsize=10, color=ink)
    fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Bars: value per group. Whiskers: bootstrap 95 % interval (resamples entries, not drivers). "
             "Dashed line: all 2025 entries.", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{v:.3f}"


def markdown(name, version, tables, gaps_by_column, rookie, grid, limits, inter, warnings) -> str:
    lines = [f"### Fairness report: {name} version {version}, 2025 holdout", ""]
    for column, table in tables.items():
        lines += [f"**{column}**", "",
                  "| Group | Entries | Drivers | Accuracy | Selection rate | FPR | FNR |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for g, row in table.iterrows():
            lines.append(f"| {g} | {int(row.entries)} | {int(row.drivers)} | " + " | ".join(
                f"{row[m]:.3f} ({row[m + '_low']:.2f}-{row[m + '_high']:.2f})" for m in METRICS) + " |")
        d = gaps_by_column[column]
        lines.append("| largest difference | | | " + " | ".join(fmt(d.loc[m, "difference"]) for m in METRICS) + " |")
        lines.append("")
    lines += ["**Rookie gaps** (rookie rate minus everyone else; positive = worse for rookies)", "",
              "| Gap | Model | Grid rule | Limit |", "|---|---:|---:|---:|"]
    for key, label in (("fpr_gap", "false positive rate"), ("fnr_gap", "false negative rate"), ("accuracy_gap", "accuracy")):
        lines.append(f"| {label} | {rookie[key]:+.3f} | {grid[key]:+.3f} | {limits.get(key, 'not gated')} |")
    lines += ["", "**rookie x team_tier** (cells with too few drivers are suppressed)", "",
              "| rookie | team_tier | Entries | Drivers | Accuracy | FPR | FNR |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in inter.iterrows():
        cells = ("suppressed: " + str(row.drivers) + " driver", "", "") if pd.isna(row.accuracy) else (
            fmt(row.accuracy), fmt(row.false_positive_rate), fmt(row.false_negative_rate))
        lines.append(f"| {row.rookie} | {row.team_tier} | {int(row.entries)} | {int(row.drivers)} | " + " | ".join(cells) + " |")
    if warnings:
        lines += ["", "**Warnings**", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("fairness_report"))
    parser.add_argument("--no-log", action="store_true", help="do not log the report to the model's MLflow run")
    args = parser.parse_args()

    criteria = load_criteria()
    holdout = load_holdout(criteria)
    uri = model_uri()
    version = resolve_version(uri)
    model = mlflow.sklearn.load_model(f"models:/{version.name}/{version.version}")
    inputs = list(model.feature_names_in_)
    predictions = pd.Series(model.predict(holdout[inputs]), index=holdout.index)

    tables, gaps_by_column = {}, {}
    for column in criteria["report_subgroups"]:
        tables[column], gaps_by_column[column] = group_table(holdout, predictions, column)
    overall = {m: float(fn(holdout[TARGET], predictions)) for m, fn in METRICS.items()}
    rookie = rookie_gaps(holdout, predictions, criteria["fairness"]["gated_subgroup"])
    grid = rookie_gaps(holdout, grid_rule(holdout), criteria["fairness"]["gated_subgroup"])
    limits = {"fpr_gap": criteria["fairness"]["max_fpr_gap"], "fnr_gap": criteria["fairness"]["max_fnr_gap"]}
    inter = intersectional(holdout, predictions, criteria)
    warnings = warnings_for(tables, criteria)

    # ---- print
    print(f"Fairness report: {version.name} version {version.version} ({uri}) on {criteria['test_season']}, "
          f"{len(holdout)} entries. Overall accuracy {overall['accuracy']:.3f}.")
    for column, table in tables.items():
        print(f"\n{column:<14}{'entries':>8}{'drivers':>8}" + "".join(f"{SHORT[m]:>7}" for m in METRICS))
        for g, row in table.iterrows():
            print(f"  {g:<12}{int(row.entries):>8}{int(row.drivers):>8}" + "".join(f"{row[m]:>7.3f}" for m in METRICS))
        d = gaps_by_column[column]
        print(f"  {'difference':<28}" + "".join(f"{d.loc[m, 'difference']:>7.3f}" for m in METRICS))
    print("\nRookie gaps (rookie minus everyone else; positive = worse for rookies):")
    print(f"  {'':<14}{'model':>8}{'grid rule':>11}{'limit':>8}")
    for key in ("fpr_gap", "fnr_gap", "accuracy_gap"):
        print(f"  {key:<14}{rookie[key]:>+8.3f}{grid[key]:>+11.3f}{str(limits.get(key, '-')):>8}")
    for w in warnings:
        print(f"WARNING {w}")

    # ---- write
    args.out.mkdir(parents=True, exist_ok=True)
    title = f"{version.name} version {version.version}: metrics per subgroup, 2025 holdout"
    pd.concat(tables, names=["subgroup", "group"]).to_csv(args.out / "fairness_table.csv")
    inter.to_csv(args.out / "intersectional.csv", index=False)
    (args.out / "fairness_table.md").write_text(
        markdown(version.name, version.version, tables, gaps_by_column, rookie, grid, limits, inter, warnings))
    chart(tables, overall, title, args.out / "fairness_by_group.png")
    report = {
        "model": version.name, "version": version.version, "run_id": version.run_id, "model_uri": uri,
        "test_season": criteria["test_season"], "entries": len(holdout), "overall": overall,
        "rookie_gaps": rookie, "grid_rule_rookie_gaps": grid, "limits": limits,
        "by_group": {c: t.rename_axis("group").reset_index().to_dict(orient="records") for c, t in tables.items()},
        "largest_difference": {c: d["difference"].to_dict() for c, d in gaps_by_column.items()},
        "intersectional": inter.to_dict(orient="records"), "warnings": warnings,
    }
    (args.out / "fairness_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out}/fairness_table.md, fairness_table.csv, fairness_by_group.png, fairness_report.json")

    if not args.no_log:
        MlflowClient().log_artifacts(version.run_id, str(args.out), artifact_path="fairness")
        print(f"Logged them to run {version.run_id} under fairness/ ({os.environ['MLFLOW_TRACKING_URI']})")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as err:
        sys.exit(f"{err}. Complete it in exercises/fairness/fairness_report.py (README step 3).")
