"""Mitigation test: try to close the rookie gap, and measure what it costs.

It trains the candidate model again (default: the lesson 04 candidate.yaml, the champion
forest) and compares it with mitigated versions on the 2025 holdout:

  baseline           the model as it is
  reweight-xW        rookie rows count W times in training (sample weighting).
                     Uses rookie only in training. The model keeps the seven API inputs.
  rookie-threshold-T the same model, but rookies need probability >= T to be predicted to score
                     (everyone else 0.5). Needs rookie at prediction time: not possible on API v1.
                     T is picked by looking at 2025, the holdout: a what-if, not a model choice.
  past-rookies-T     the same, but T is chosen on 2021-2024 only (out-of-fold, one season per
                     fold), to maximise rookie accuracy there. The honest way to pick T.

Every variant is one MLflow run in the experiment f1-points-finish-fairness, nested in one
parent run that holds the comparison table and the trade-off chart.

Usage:
    python exercises/fairness/mitigate.py
    python exercises/fairness/mitigate.py --weights 2 5 10 --thresholds 0.6 0.7 0.8
    python exercises/fairness/mitigate.py --candidate exercises/quality-gates/examples/logreg.yaml
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "quality-gates"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402

from fairness_metrics import TARGET, gate_verdicts, grid_rule, load_criteria, rookie_gaps  # noqa: E402
from train_candidate import CATEGORICAL, NUMERIC, TRAIN_SEASONS, build_model  # noqa: E402

EXPERIMENT = "f1-points-finish-fairness"
FEATURES = NUMERIC + CATEGORICAL
LAB04 = yaml.safe_load((HERE.parent / "quality-gates" / "criteria.yaml").read_text())["metrics"]


def passes_lab04(accuracy: float, auc: float, rookie_accuracy: float, grid_accuracy: float) -> bool:
    """The lesson 04 metric gates, measured the same way."""
    return (accuracy >= LAB04["min_accuracy"] and auc >= LAB04["min_roc_auc"]
            and accuracy >= grid_accuracy + LAB04["min_margin_over_grid_rule"]
            and rookie_accuracy >= LAB04["min_accuracy_rookie"])


def past_rookie_threshold(candidate: dict, train: pd.DataFrame) -> float:
    """Choose the rookie threshold on 2021-2024 only: out-of-fold predictions, one season per fold."""
    oof = cross_val_predict(build_model(candidate, NUMERIC), train[FEATURES], train[TARGET],
                            groups=train["season"], cv=GroupKFold(n_splits=len(TRAIN_SEASONS)),
                            method="predict_proba")[:, 1]
    rookie = train["rookie"].to_numpy()
    y = train[TARGET].to_numpy()[rookie]
    candidates = np.round(np.arange(0.30, 0.91, 0.05), 2)
    accuracy = [((oof[rookie] >= t).astype(int) == y).mean() for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])


def evaluate(name, kind, setting, test, proba, predictions, criteria, grid_accuracy, deployable):
    g = rookie_gaps(test, predictions, criteria["fairness"]["gated_subgroup"])
    accuracy = float((predictions == test[TARGET]).mean())
    auc = float(roc_auc_score(test[TARGET], proba))
    verdicts = gate_verdicts(g, criteria)
    return {
        "run": name, "mitigation": kind, "setting": setting,
        "accuracy": accuracy, "roc_auc": auc,
        "rookie_accuracy": g["group"]["accuracy"], "others_accuracy": g["rest"]["accuracy"],
        "accuracy_gap": g["accuracy_gap"],
        "rookie_fpr": g["group"]["false_positive_rate"], "rookie_fnr": g["group"]["false_negative_rate"],
        "fpr_gap": g["fpr_gap"], "fnr_gap": g["fnr_gap"],
        "passes_lab04": passes_lab04(accuracy, auc, g["group"]["accuracy"], grid_accuracy),
        "passes_fairness": all(verdicts.values()),
        "api_v1": deployable,
    }


def chart(results: pd.DataFrame, limit: float, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    colors = {"none": "#0b0b0b", "reweight": "#2a78d6", "rookie_threshold": "#eb6834", "past_rookies": "#6250d6"}
    for kind, rows in results.groupby("mitigation"):
        ax.scatter(rows["fpr_gap"], rows["accuracy"], s=60, color=colors.get(kind, "#52514e"), label=kind, zorder=3)
    for _, row in results.iterrows():
        ax.annotate(row["run"], (row["fpr_gap"], row["accuracy"]), textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="#52514e")
    ax.axvline(limit, color="#52514e", linestyle="--", linewidth=1)
    ax.text(limit, ax.get_ylim()[0], f" max_fpr_gap {limit}", va="bottom", fontsize=8, color="#52514e")
    ax.set_xlabel("rookie false positive rate gap (lower is better for rookies)")
    ax.set_ylabel("accuracy on 2025 (all entries)")
    ax.set_title("Mitigation test: accuracy against the rookie FPR gap", fontsize=11, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", type=Path, default=Path("exercises/quality-gates/candidate.yaml"))
    parser.add_argument("--weights", type=float, nargs="*", default=[2, 5, 10])
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.6, 0.7, 0.8])
    parser.add_argument("--out", type=Path, default=Path("fairness_report"))
    args = parser.parse_args()

    criteria = load_criteria()
    candidate = yaml.safe_load(args.candidate.read_text())
    races = pd.read_csv(criteria["data"])
    train = races[races["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    test = races[races["season"] == criteria["test_season"]].reset_index(drop=True)
    rookie_test = test["rookie"].to_numpy()
    grid_accuracy = float((grid_rule(test) == test[TARGET]).mean())

    def fit(weights=None):
        model = build_model(candidate, NUMERIC)
        fit_params = {} if weights is None else {"model__sample_weight": weights}
        return model.fit(train[FEATURES], train[TARGET], **fit_params)

    results = []
    base = fit()
    base_proba = base.predict_proba(test[FEATURES])[:, 1]
    results.append(evaluate("baseline", "none", "", test, base_proba, pd.Series((base_proba >= 0.5).astype(int)),
                            criteria, grid_accuracy, True))
    for w in args.weights:
        proba = fit(np.where(train["rookie"], w, 1.0)).predict_proba(test[FEATURES])[:, 1]
        results.append(evaluate(f"reweight-x{w:g}", "reweight", w, test, proba,
                                pd.Series((proba >= 0.5).astype(int)), criteria, grid_accuracy, True))
    past = past_rookie_threshold(candidate, train)
    for t, kind, name in [(t, "rookie_threshold", f"rookie-threshold-{t:g}") for t in args.thresholds] + \
                         [(past, "past_rookies", f"past-rookies-{past:g}")]:
        predictions = pd.Series(np.where(rookie_test, base_proba >= t, base_proba >= 0.5).astype(int))
        results.append(evaluate(name, kind, t, test, base_proba, predictions, criteria, grid_accuracy, False))
    table = pd.DataFrame(results)

    # ---- MLflow: one parent run, one child run per variant
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "mitigation.csv", index=False)
    chart(table, criteria["fairness"]["max_fpr_gap"], args.out / "mitigation_tradeoff.png")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"mitigation-test-{candidate['model']}") as parent:
        mlflow.set_tags({"candidate.file": str(args.candidate), "lesson": "08"})
        mlflow.log_params({"model_type": candidate["model"], **(candidate.get("hyperparameters") or {})})
        mlflow.log_artifact(str(args.out / "mitigation.csv"))
        mlflow.log_artifact(str(args.out / "mitigation_tradeoff.png"))
        for row in results:
            with mlflow.start_run(run_name=row["run"], nested=True):
                mlflow.set_tags({
                    "mitigation": row["mitigation"],
                    "needs_rookie_at_prediction": str(not row["api_v1"]).lower(),
                    "tuned_on_holdout": str(row["mitigation"] == "rookie_threshold").lower(),
                })
                mlflow.log_params({"mitigation": row["mitigation"], "setting": row["setting"]})
                mlflow.log_metrics({k: float(row[k]) for k in (
                    "accuracy", "roc_auc", "rookie_accuracy", "others_accuracy", "accuracy_gap",
                    "rookie_fpr", "rookie_fnr", "fpr_gap", "fnr_gap", "passes_lab04", "passes_fairness")})

    # ---- print
    print(f"Mitigation test for {args.candidate} on {criteria['test_season']} "
          f"(limits: fpr_gap <= {criteria['fairness']['max_fpr_gap']}, fnr_gap <= {criteria['fairness']['max_fnr_gap']})")
    print(f"{'run':<22}{'acc':>7}{'rookie':>8}{'fpr_gap':>9}{'fnr_gap':>9}{'lab04':>7}{'fair':>6}{'API v1':>8}")
    for row in results:
        print(f"{row['run']:<22}{row['accuracy']:>7.3f}{row['rookie_accuracy']:>8.3f}{row['fpr_gap']:>+9.3f}"
              f"{row['fnr_gap']:>+9.3f}{'pass' if row['passes_lab04'] else 'FAIL':>7}"
              f"{'pass' if row['passes_fairness'] else 'FAIL':>6}{'yes' if row['api_v1'] else 'no':>8}")
    print(f"\nLogged {len(results)} runs under the parent run {parent.info.run_id} "
          f"in the experiment {EXPERIMENT}. Chart: {args.out}/mitigation_tradeoff.png")


if __name__ == "__main__":
    main()
