"""Set the final decision of the release dossier: the human step after accept.py.

Only a named person can set "ship", and only when every gate passed. Anyone named can set
"do not ship" at any time, with a reason. decide.py refuses when:
- the dossier was edited after accept.py wrote it (its sha256 no longer matches),
- the decision is "ship" but a gate failed or its evidence is missing,
- the approver is not named, is a bot, or is not in release.yaml's approvers
  (in GitHub Actions: --via github-environment, where the environment's reviewers decide).

It writes the decision into dossier.md and dossier.json and tags the model version in MLflow.

Usage:
    python exercises/acceptance/decide.py --decision ship --approver "Your Name"
    python exercises/acceptance/decide.py --decision "do not ship" --approver "Your Name" --reason "p95 too close to the SLO"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from accept import DECISION_END, DECISION_START, DOSSIER, body_sha256, decision_block  # noqa: E402

BOTS = ("github-actions", "github-actions[bot]", "accept.py", "ci", "bot", "")


def refuse(reason: str):
    print(f"REFUSED: {reason}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decision", required=True, choices=["ship", "do not ship"])
    parser.add_argument("--approver", required=True, help="the person who decides")
    parser.add_argument("--reason", default="", help="why (required for do not ship)")
    parser.add_argument(
        "--via", default="decide.py", help='how the approval was given, e.g. "github-environment production"'
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()

    if not DOSSIER.exists():
        refuse(f"{DOSSIER} not found. Run accept.py first.")
    markdown = DOSSIER.read_text()
    dossier = json.loads(DOSSIER.with_suffix(".json").read_text())
    if body_sha256(markdown) != dossier["dossier_sha256"]:
        refuse(f"{DOSSIER} was changed after accept.py wrote it. Run accept.py again.")

    approver = args.approver.strip()
    if approver.lower() in BOTS or approver.lower().endswith("[bot]"):
        refuse(f"{approver!r} is not a person. A release decision needs a named human approver.")
    if (
        args.decision == "ship"
        and not args.via.startswith("github-environment")
        and approver not in dossier["approvers"]
    ):
        # In GitHub Actions the environment's required reviewers are the approver list.
        refuse(f"{approver!r} is not an approver in release.yaml (approvers: {', '.join(dossier['approvers'])}).")

    reason = args.reason
    if args.decision == "ship":
        if dossier["recommendation"] != "ready for a human decision":
            refuse(f"the suite says do not ship: {dossier['decision']['reason']}. Nobody can ship this release.")
        reason = reason or "every gate passed; I reviewed the dossier"
    elif not reason:
        refuse('"do not ship" needs a --reason.')

    warnings = []
    if approver == dossier["owners"].get("model"):
        warnings.append("the approver is also the model owner: no separation of duties")
    decision = {
        "value": args.decision,
        "by": approver,
        "via": args.via,
        "reason": reason + "".join(f" (WARNING: {w})" for w in warnings),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    start, end = markdown.index(DECISION_START), markdown.index(DECISION_END) + len(DECISION_END)
    DOSSIER.write_text(markdown[:start] + decision_block(decision, dossier["recommendation"]) + markdown[end:])
    dossier["decision"] = decision
    DOSSIER.with_suffix(".json").write_text(json.dumps(dossier, indent=2) + "\n")

    mlflow.set_tracking_uri(args.tracking_uri)
    try:
        client = MlflowClient()
        for key, value in {
            "release.decision": args.decision,
            "release.decided_by": approver,
            "release.decided_at": decision["at"],
            "release.decided_via": args.via,
            "release.dossier_sha256": dossier["dossier_sha256"],
        }.items():
            client.set_model_version_tag(dossier["model_name"], dossier["model_version"], key, value)
        tagged = f"Tagged {dossier['model_name']} version {dossier['model_version']} in MLflow."
    except Exception as err:
        tagged = f"WARNING: could not tag the model version in MLflow: {err}"

    for w in warnings:
        print(f"WARNING: {w}")
    print(
        f"DECISION: {args.decision.upper()} for {dossier['model_name']} version {dossier['model_version']}, "
        f"set by {approver} via {args.via}. Written to {DOSSIER}. {tagged}"
    )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(
                f"### Release decision\n\n**{args.decision}** for {dossier['model_name']} version "
                f"{dossier['model_version']}, set by {approver} via {args.via}. {reason}\n"
            )


if __name__ == "__main__":
    main()
