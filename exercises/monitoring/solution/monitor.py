"""Monitor f1-points-finish in production. Reference solution: try the TODOs in ../monitor.py first.

Reads the prediction log that the API writes (PREDICTION_LOG) and the manifest of the last replay
(logs/replay_manifest.json), and runs five checks on that window:

    service health          how many entries the API rejected, and why
    input drift             the inputs the model saw, against the reference window
    training-serving skew   the inputs the API received, against the features the training pipeline recomputes
    delayed labels          accuracy once the race results arrive: provisional and final labels
    prediction drift        the share of cars predicted to score points

Each check passes or raises an alert, with a reason. The thresholds are in monitoring.yaml.
monitor.py writes results/monitoring/alerts.json, summary.md and drift_report.html (Evidently),
and exits 1 when any check raises an alert.

Usage:
    python exercises/monitoring/solution/monitor.py
    python exercises/monitoring/solution/monitor.py --labels f1/data/season_2026_final.csv
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

STATUS_LABEL = {"pass": "PASS", "alert": "ALERT", "skip": "SKIP", "todo": "TODO"}


@dataclass
class Check:
    name: str
    status: str  # pass, alert, skip (nothing to check) or todo (not written yet)
    summary: str  # one line: what was measured and why it passed or alerted
    details: list[str] = field(default_factory=list)
    measured: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# Load the window: the log lines of the requests that replay.py (or inject.py) sent.
# ---------------------------------------------------------------------------------------------

def load_window(log_path: Path, manifest_path: Path) -> tuple[dict, pd.DataFrame]:
    """Return the manifest and one row per logged entry of its requests.

    Columns: request_id, entry, status, model_version, prediction, probability, errors,
    key (race/driver), served (the inputs as logged: a dict) and recomputed (a dict or None).
    """
    if not manifest_path.exists():
        sys.exit(f"{manifest_path} not found. Send a window to the API first: python exercises/monitoring/replay.py reference")
    manifest = json.loads(manifest_path.read_text())
    requests = {r["request_id"]: r for r in manifest["requests"]}
    rows = []
    if log_path.exists():
        with log_path.open() as f:
            for line in f:
                record = json.loads(line)
                request = requests.get(record.get("request_id"))
                if request is None:
                    continue  # another window
                i = record["entry"]
                rows.append({
                    "request_id": record["request_id"],
                    "entry": i,
                    "status": record["status"],
                    "model_version": record.get("model_version"),
                    "prediction": record.get("prediction"),
                    "probability": record.get("probability"),
                    "errors": record.get("errors", []),
                    "key": request["keys"][i] if i < len(request["keys"]) else None,
                    "served": record.get("inputs") or {},
                    "recomputed": request["recomputed"][i] if i < len(request.get("recomputed") or []) else None,
                })
    return manifest, pd.DataFrame(rows, columns=[
        "request_id", "entry", "status", "model_version", "prediction", "probability",
        "errors", "key", "served", "recomputed"])


def answered(window: pd.DataFrame) -> pd.DataFrame:
    """The entries that the model answered (status 200)."""
    return window[window["status"] == 200]


def as_frame(dicts: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Turn a column of dicts into a DataFrame with the given columns (missing values are NaN)."""
    return pd.DataFrame(list(dicts), index=dicts.index).reindex(columns=columns)


# ---------------------------------------------------------------------------------------------
# Check 1: service health
# ---------------------------------------------------------------------------------------------

def check_service_health(window: pd.DataFrame, cfg: dict) -> Check:
    limit = cfg["service_health"]["max_rejected_share"]
    if window.empty:
        return Check("service health", "alert",
                     "the log has no entry of this window: is PREDICTION_LOG set, and did the replay reach this API?")
    rejected = window["status"] != 200
    share = rejected.mean()
    versions = sorted(str(v) for v in window["model_version"].dropna().unique())
    measured = {"entries": len(window), "rejected": int(rejected.sum()), "rejected_share": round(share, 4),
                "model_versions": versions}
    summary = (f"{len(window) - rejected.sum()} of {len(window)} entries answered, {rejected.sum()} rejected "
               f"({share:.1%}, max {limit:.0%})")
    errors = Counter(e for errs in window.loc[rejected, "errors"] for e in errs)
    details = [f"model version(s) in the log: {', '.join(versions) or 'none'}"]
    details += [f"{n} x {error}" for error, n in errors.most_common(3)]
    return Check("service health", "alert" if share > limit else "pass", summary, details, measured)


# ---------------------------------------------------------------------------------------------
# Check 2: input drift
# ---------------------------------------------------------------------------------------------

def numeric_drift(name: str, current: pd.Series, reference: pd.Series, cfg: dict, per_race: bool):
    """Compare one numeric feature with the reference.

    Return (measured, reasons): measured = {"mean", "reference_mean", "mean_shift", "spread_ratio"},
    reasons = a list of alert reasons, empty when the feature looks normal.
    """
    t = cfg["input_drift"]
    ref_mean, ref_sd = reference.mean(), reference.std()
    reasons = []

    # TODO 1
    mean_shift = abs(current.mean() - ref_mean) / ref_sd
    spread_ratio = current.std() / ref_sd if len(current) > 1 else float("nan")
    if mean_shift > t["max_mean_shift"]:
        reasons.append(f"{name}: mean {current.mean():.4g} vs {ref_mean:.4g} in the reference, "
                       f"shift {mean_shift:.2f} sd (max {t['max_mean_shift']})")
    if not per_race and not t["min_spread_ratio"] <= spread_ratio <= t["max_spread_ratio"]:
        reasons.append(f"{name}: spread x{spread_ratio:.3g} of the reference "
                       f"(allowed x{t['min_spread_ratio']} to x{t['max_spread_ratio']})")

    measured = {"mean": round(current.mean(), 4), "reference_mean": round(ref_mean, 4),
                "mean_shift": round(mean_shift, 2), "spread_ratio": round(spread_ratio, 4)}
    return measured, reasons


def evidently_report(current: pd.DataFrame, reference: pd.DataFrame, cfg: dict, html: Path) -> list[str]:
    """Run Evidently's data drift preset, save the HTML report and return the columns it flags."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=cfg["numeric_features"],
                                categorical_columns=cfg["categorical_features"])
    columns = cfg["numeric_features"] + cfg["categorical_features"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snapshot = Report([DataDriftPreset()]).run(
            Dataset.from_pandas(current[columns].reset_index(drop=True), data_definition=definition),
            Dataset.from_pandas(reference[columns].reset_index(drop=True), data_definition=definition),
        )
    html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html))
    flagged = []
    for metric in snapshot.dict()["metrics"]:
        config = metric.get("config", {})
        if "ValueDrift" not in config.get("type", ""):
            continue
        value, threshold = float(metric["value"]), float(config["threshold"])
        # A p-value drifts when it is small; a distance drifts when it is large.
        drifted = value < threshold if "p_value" in config["method"] else value >= threshold
        if drifted:
            flagged.append(config["column"])
    return flagged


def check_input_drift(window: pd.DataFrame, reference: pd.DataFrame, cfg: dict, out: Path) -> Check:
    seen = answered(window)
    if seen.empty:
        return Check("input drift", "skip", "no answered entries: no input reached the model")
    num, cat = cfg["numeric_features"], cfg["categorical_features"]
    current = as_frame(seen["served"], num + cat)
    current[num] = current[num].astype(float)

    flagged = evidently_report(current, reference, cfg, out / "drift_report.html")
    measured, reasons, details = {"evidently_flags": flagged}, [], []
    for name in num:
        per_race = name in cfg["input_drift"]["per_race_features"]
        m, why = numeric_drift(name, current[name], reference[name], cfg, per_race)
        measured[name], reasons = m, reasons + why
        spread = "per race, not checked" if per_race else f"x{m['spread_ratio']:.3g}"
        details.append(f"{name:<14} mean {m['mean']:>9.4g}  reference {m['reference_mean']:>9.4g}  "
                       f"shift {m['mean_shift']:>5.2f} sd  spread {spread}")
    for name in cat:
        unseen = sorted(set(current[name].dropna()) - set(reference[name]))
        shares = current[name].value_counts(normalize=True).round(2).to_dict()
        measured[name] = {"shares": shares, "unseen": unseen}
        details.append(f"{name:<14} {shares}")
        if unseen:
            reasons.append(f"{name}: values not in the reference: {', '.join(unseen)}")

    details.append(f"Evidently flags (p < 0.05): {', '.join(flagged) or 'none'}")
    if reasons:
        return Check("input drift", "alert", "; ".join(reasons), details, measured)
    return Check("input drift", "pass", f"{len(num) + len(cat)} features within the thresholds "
                 f"({len(seen)} entries)", details, measured)


# ---------------------------------------------------------------------------------------------
# Check 3: training-serving skew
# ---------------------------------------------------------------------------------------------

def skew_mismatches(served: pd.Series, recomputed: pd.Series, numeric: bool, cfg: dict) -> pd.Series:
    """Return True for each entry where the served value differs from the recomputed one."""
    # TODO 2
    if numeric:
        s, r = pd.to_numeric(served, errors="coerce"), recomputed.astype(float)
        return s.isna() | ((s - r).abs() > cfg["skew"]["relative_tolerance"] * r.abs())
    return served.astype(str) != recomputed.astype(str)


def describe_skew(name: str, served: pd.Series, recomputed: pd.Series, numeric: bool) -> str:
    """A hint at what went wrong: a ratio for numbers, the most common swap for categories."""
    if served.dropna().nunique() == 1 and recomputed.nunique() > 1:
        value = served.dropna().iloc[0]
        return f"served is always {value!r}" if isinstance(value, str) else f"served is always {value:g}"
    if numeric:
        s, r = pd.to_numeric(served, errors="coerce"), recomputed.astype(float)
        ratio = (s / r)[(r != 0) & s.notna()]
        return f"served / recomputed: median x{ratio.median():.3g}" if not ratio.empty else "served is missing"
    pair, n = Counter(zip(served.astype(str), recomputed.astype(str))).most_common(1)[0]
    return f"served {pair[0]!r} where recomputed is {pair[1]!r} ({n} entries)"


def check_skew(window: pd.DataFrame, cfg: dict) -> Check:
    window = window[window["recomputed"].notna()]
    if window.empty:
        return Check("training-serving skew", "skip", "no recomputed features for this window")
    num, cat = cfg["numeric_features"], cfg["categorical_features"]
    served, recomputed = as_frame(window["served"], num + cat), as_frame(window["recomputed"], num + cat)
    limit = cfg["skew"]["max_mismatch_share"]

    measured, reasons, details = {}, [], []
    for name in num + cat:
        numeric = name in num
        mismatch = skew_mismatches(served[name], recomputed[name], numeric, cfg)
        share = mismatch.mean()
        measured[name] = {"mismatches": int(mismatch.sum()), "share": round(share, 4)}
        if not mismatch.any():
            continue
        line = (f"{name}: {mismatch.sum()} of {len(mismatch)} served values differ from the recomputed ones; "
                f"{describe_skew(name, served[name][mismatch], recomputed[name][mismatch], numeric)}")
        details.append(line)
        if share > limit:
            reasons.append(f"{line} ({share:.0%}, max {limit:.0%})")
    if reasons:
        return Check("training-serving skew", "alert", "; ".join(reasons), [], measured)
    summary = (f"served = recomputed for all {len(num) + len(cat)} features ({len(window)} entries)" if not details
               else f"differences below the threshold of {limit:.0%} ({len(window)} entries)")
    return Check("training-serving skew", "pass", summary, details, measured)


# ---------------------------------------------------------------------------------------------
# Check 4: delayed labels
# ---------------------------------------------------------------------------------------------

def read_labels(path: Path) -> pd.DataFrame:
    """Return key, provisional and final labels. season_2026_provisional.csv has no final labels."""
    labels = pd.read_csv(path)
    labels["key"] = labels["race_id"] + "/" + labels["driver_code"]
    if "points_finish_final" in labels:
        labels = labels.rename(columns={"points_finish_provisional": "provisional", "points_finish_final": "final"})
    else:
        labels = labels.rename(columns={"points_finish": "provisional"}).assign(final=float("nan"))
    return labels[["key", "provisional", "final"]]


def label_performance(joined: pd.DataFrame, cfg: dict):
    """Measure the champion against the labels that have arrived.

    joined has one row per answered entry: prediction, provisional and final (NaN = not arrived yet).
    Return (measured, reasons).
    """
    t = cfg["delayed_labels"]
    has_prov, has_final = joined["provisional"].notna(), joined["final"].notna()
    coverage = has_final.mean()
    reasons = []
    if coverage < t["min_final_label_coverage"]:
        reasons.append(f"labels late: {has_final.sum()} of {len(joined)} predictions have a final label "
                       f"({coverage:.0%}, need {t['min_final_label_coverage']:.0%})")

    # TODO 3
    acc_prov = (joined["prediction"] == joined["provisional"])[has_prov].mean()
    acc_final = (joined["prediction"] == joined["final"])[has_final].mean()
    floor = t["baseline_accuracy"] - t["max_accuracy_drop"]
    if has_final.any() and acc_final < floor:
        reasons.append(f"accuracy {acc_final:.3f} on the final labels, baseline {t['baseline_accuracy']:.3f} "
                       f"(alert below {floor:.3f})")

    both = has_prov & has_final
    changed = int((joined["provisional"] != joined["final"])[both].sum())
    right_prov = joined["prediction"] == joined["provisional"]
    right_final = joined["prediction"] == joined["final"]
    measured = {"predictions": len(joined), "provisional_labels": int(has_prov.sum()),
                "final_labels": int(has_final.sum()), "final_label_coverage": round(coverage, 4),
                "accuracy_provisional": round(acc_prov, 4), "accuracy_final": round(acc_final, 4),
                "labels_changed": changed, "right_to_wrong": int((right_prov & ~right_final)[both].sum()),
                "wrong_to_right": int((~right_prov & right_final)[both].sum()),
                "baseline_accuracy": t["baseline_accuracy"]}
    return measured, reasons


def check_delayed_labels(window: pd.DataFrame, labels_path: Path | None, cfg: dict) -> Check:
    if labels_path is None:
        return Check("delayed labels", "skip", "no labels for this window yet: the results are not known")
    seen = answered(window)
    if seen.empty:
        return Check("delayed labels", "skip", "no answered entries to compare with the labels")
    joined = seen.merge(read_labels(labels_path), on="key", how="left")
    measured, reasons = label_performance(joined, cfg)
    m = measured
    details = [
        f"labels from {labels_path}: {m['provisional_labels']} provisional, {m['final_labels']} final "
        f"for {m['predictions']} predictions",
        f"accuracy on provisional labels {m['accuracy_provisional']:.3f}, on final labels {m['accuracy_final']:.3f} "
        f"(baseline {m['baseline_accuracy']:.3f})",
        f"the stewards changed {m['labels_changed']} labels: {m['right_to_wrong']} predictions went from right "
        f"to wrong, {m['wrong_to_right']} from wrong to right",
    ]
    if reasons:
        return Check("delayed labels", "alert", "; ".join(reasons), details, measured)
    return Check("delayed labels", "pass",
                 f"accuracy {m['accuracy_final']:.3f} on {m['final_labels']} final labels "
                 f"(baseline {m['baseline_accuracy']:.3f})", details, measured)


# ---------------------------------------------------------------------------------------------
# Check 5: prediction drift
# ---------------------------------------------------------------------------------------------

def check_prediction_drift(window: pd.DataFrame, cfg: dict) -> Check:
    seen = answered(window)
    if seen.empty:
        return Check("prediction drift", "skip", "no predictions in this window")
    t = cfg["prediction_drift"]

    # TODO 4
    share = seen["prediction"].mean()
    change = abs(share - t["expected_share"])

    measured = {"share_predicted_points": round(share, 4), "expected_share": t["expected_share"]}
    summary = (f"{share:.2f} of the cars predicted to score points (expected {t['expected_share']}, "
               f"max change {t['max_share_change']})")
    return Check("prediction drift", "alert" if change > t["max_share_change"] else "pass", summary,
                 [], measured)


# ---------------------------------------------------------------------------------------------
# Report: print, alerts.json, summary.md, exit code
# ---------------------------------------------------------------------------------------------

def describe_window(manifest: dict, window: pd.DataFrame) -> str:
    versions = ", ".join(sorted(str(v) for v in window["model_version"].dropna().unique())) or "none"
    n = len(manifest["requests"])
    return (f"{manifest['window']}: {n} request{'s' if n != 1 else ''}, {len(window)} entries logged, "
            f"model version {versions}, replayed {manifest['replayed_at']}")


def write_outputs(out: Path, manifest: dict, window: pd.DataFrame, checks: list[Check]) -> list[Check]:
    alerts = [c for c in checks if c.status == "alert"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "alerts.json").write_text(json.dumps({
        "window": manifest["window"],
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "alert" if alerts else "ok",
        "alerts": [{"check": c.name, "reason": c.summary} for c in alerts],
        "checks": [asdict(c) for c in checks],
    }, indent=1, default=str))

    verdict = f"ALERT: {', '.join(c.name for c in alerts)}" if alerts else "OK, no alerts"
    md = [f"### Monitoring: {verdict}", "", f"Window {describe_window(manifest, window)}.", "",
          "| Check | Result | Reason |", "|---|---|---|"]
    md += [f"| {c.name} | {STATUS_LABEL[c.status]} | {c.summary.replace('|', '/')} |" for c in checks]
    for c in alerts:
        md += ["", f"**{c.name}**", ""] + [f"- `{d}`" for d in c.details]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return alerts


def run(name: str, check, *args) -> Check:
    """Run one check. A TODO that is not written yet gives the result TODO, not a crash."""
    try:
        return check(*args)
    except NotImplementedError as todo:
        return Check(name, "todo", str(todo))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl")))
    parser.add_argument("--manifest", type=Path, default=Path("logs/replay_manifest.json"))
    parser.add_argument("--labels", type=Path, help="CSV with race_id, driver_code and the labels that arrived")
    parser.add_argument("--config", type=Path, default=Path("exercises/monitoring/monitoring.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results/monitoring"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    reference = pd.DataFrame(json.loads(Path(cfg["reference"]).read_text())["rows"])
    manifest, window = load_window(args.log, args.manifest)
    labels = args.labels or (Path(manifest["labels"]) if manifest.get("labels") else None)
    (args.out / "drift_report.html").unlink(missing_ok=True)  # never show the report of an older window

    checks = [
        run("service health", check_service_health, window, cfg),
        run("input drift", check_input_drift, window, reference, cfg, args.out),
        run("training-serving skew", check_skew, window, cfg),
        run("delayed labels", check_delayed_labels, window, labels, cfg),
        run("prediction drift", check_prediction_drift, window, cfg),
    ]
    print(f"Monitoring window {describe_window(manifest, window)}")
    for c in checks:
        print(f"  {STATUS_LABEL[c.status]:<6} {c.name:<22} {c.summary}")
        for d in c.details:
            print(f"  {'':<6} {'':<22}   {d}")
    alerts = write_outputs(args.out, manifest, window, checks)
    print(f"Reports: {args.out}/alerts.json, {args.out}/summary.md, {args.out}/drift_report.html")
    todo = [c.name for c in checks if c.status == "todo"]
    if todo:
        checks_are = "1 check is" if len(todo) == 1 else f"{len(todo)} checks are"
        print(f"WARNING: {checks_are} not written yet and checked nothing: {', '.join(todo)}")
    if alerts:
        print(f"MONITOR ALERT: {', '.join(c.name for c in alerts)}")
        sys.exit(1)
    print("MONITOR OK: no alerts")


if __name__ == "__main__":
    main()
