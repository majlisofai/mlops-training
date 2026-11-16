"""Run every gate of the course against the release and write the release dossier.

The release is the model version behind @champion (release.yaml), served by the API at
API_URL. accept.py runs ten gates, one after the other, and does not stop at a failure: the
dossier must show all the evidence. Then it writes dossier.md (for people) and dossier.json
(for decide.py) and tags the model version with the result.

The suite never sets the decision to "ship". When every gate passes, the decision stays
PENDING until a named approver sets it with decide.py. When a gate fails, or its evidence is
missing, the suite sets the decision to "do not ship" itself.

Exit 0: every gate passed. Exit 1: a gate failed or its evidence is missing.

Environment:
    MLFLOW_TRACKING_URI   default http://127.0.0.1:5000
    API_URL               the running API, default http://127.0.0.1:8000
    API_CLIENT_KEY, API_ADMIN_KEY   the keys that the API uses (set -a; source .secrets/api.env; set +a)

Usage:
    python exercises/acceptance/accept.py --image lab09-api:1
    python exercises/acceptance/accept.py --reference --image lab09-api:ref   # the reference access tests
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

import httpx2  # noqa: E402
import mlflow  # noqa: E402
import yaml  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

HERE = Path(os.path.relpath(Path(__file__).parent))  # relative paths read better in the dossier
OUT = Path("results/acceptance")
DOSSIER = Path("dossier.md")
DECISION_START, DECISION_END = "<!-- decision -->", "<!-- /decision -->"

# Where each gate comes from, and the published controls it gives evidence for.
# NIST AI RMF 1.0 (NIST AI 100-1) subcategories and SDAIA AI Ethics Principles. README.md lists the sources.
CONTROLS = {
    "data": ("NIST MAP 2.3", "SDAIA Reliability & Safety"),
    "integrity": ("NIST MEASURE 2.7", "SDAIA Privacy & Security"),
    "quality": ("NIST MEASURE 2.3, MEASURE 2.5", "SDAIA Reliability & Safety"),
    "deployment": ("NIST MEASURE 2.8", "SDAIA Accountability & Responsibility"),
    "contract": ("NIST MEASURE 2.5", "SDAIA Reliability & Safety"),
    "access": ("NIST MEASURE 2.7", "SDAIA Privacy & Security"),
    "slo": ("NIST MEASURE 2.3", "SDAIA Reliability & Safety"),
    "dependencies": ("NIST MEASURE 2.7, MANAGE 3.1", "SDAIA Privacy & Security"),
    "model_card": ("NIST MEASURE 2.9", "SDAIA Transparency & Explainability"),
    "fairness": ("NIST MEASURE 2.11", "SDAIA Fairness"),
}


@dataclass
class Gate:
    key: str
    name: str
    lesson: str
    status: str = "MISSING"  # PASS, FAIL or MISSING (no evidence: not run, not written, not attached)
    evidence: str = ""
    log: str = ""
    seconds: float = 0.0
    details: dict = field(default_factory=dict)


def run(cmd: list[str], log: Path, env: dict | None = None) -> tuple[int, str]:
    """Run a command, keep its output in the log file, and return (exit code, output)."""
    full_env = {**os.environ, **(env or {})}
    full_env.pop("GITHUB_STEP_SUMMARY", None)  # accept.py writes the run summary itself
    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    output = result.stdout + result.stderr
    log.write_text(f"$ {' '.join(cmd)}\n\n{output}")
    return result.returncode, output


def junit_counts(xml_file: Path) -> dict:
    """Count the outcomes in a pytest --junitxml report."""
    if not xml_file.exists():
        return {}
    suite = ET.parse(xml_file).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    counts = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return counts


def pytest_gate(gate: Gate, test_file: str, env: dict, log: Path):
    xml = log.with_suffix(".xml")
    xml.unlink(missing_ok=True)
    code, output = run(
        [
            sys.executable,
            "-m",
            "pytest",
            test_file,
            "-q",
            "-rs",
            "--tb=line",
            "--no-header",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml}",
        ],
        log,
        env,
    )
    c = junit_counts(xml)
    if not c:
        gate.status, gate.evidence = "FAIL", f"pytest did not run: {output.strip().splitlines()[-1:]}"
        return
    gate.details = c
    gate.evidence = (
        f"{c['passed']} passed, {c['failures'] + c['errors']} failed, {c['skipped']} not written ({test_file})"
    )
    if c["failures"] or c["errors"]:
        failed = [line for line in output.splitlines() if "Error" in line and ".py:" in line][:3]
        gate.status = "FAIL"
        gate.evidence += "; " + "; ".join(f.split(": ", 1)[-1][:160] for f in failed)
    elif c["skipped"]:
        gate.status = "MISSING"  # a skipped test checks nothing: no evidence
    else:
        gate.status = "PASS"


class Acceptance:
    def __init__(self, args):
        self.args = args
        self.release = yaml.safe_load(args.release.read_text())
        self.api_url = args.api_url.rstrip("/")
        self.client_key = os.environ.get("API_CLIENT_KEY", "")
        self.mlflow = MlflowClient()
        name, alias = self.release["model_name"], self.release["alias"]
        self.version = self.mlflow.get_model_version_by_alias(name, alias)
        self.run = self.mlflow.get_run(self.version.run_id)
        self.facts = {}
        files = "solution/" if args.reference else ""
        self.access_tests = str(HERE / f"{files}test_access.py")
        self.evidence = dict(self.release["evidence"])

    # ---- the gates --------------------------------------------------------------------

    def gate_data(self, gate: Gate, log: Path):
        csv = self.release["integrity"]["files"][0]
        code, output = run([sys.executable, "exercises/quality-gates/validate_data.py", csv], log)
        found = re.search(r"data version (\w+): validation (\w+) \((\d+) failure cases\)", output)
        if not found:
            gate.status, gate.evidence = "FAIL", "validate_data.py did not run: " + output.strip()[-200:]
            return
        md5, status, failures = found.groups()
        trained_on = self.run.data.tags.get("data.version", "unknown")
        self.facts.update(data_md5=md5, data_md5_trained=trained_on, data_path=csv)
        same = md5 == trained_on
        gate.status = "PASS" if code == 0 and same else "FAIL"
        gate.evidence = f"{failures} failure cases; md5 {md5[:12]}" + (
            ", the data the model was trained on" if same else f" but the model was trained on {trained_on[:12]}"
        )

    def gate_integrity(self, gate: Gate, log: Path):
        result_file = OUT / "integrity.json"
        result_file.unlink(missing_ok=True)
        code, output = run([sys.executable, str(HERE / "manifest.py"), "verify", "--json", str(result_file)], log)
        if not result_file.exists():
            gate.status, gate.evidence = "FAIL", "manifest.py did not run: " + output.strip()[-200:]
            return
        result = json.loads(result_file.read_text())
        gate.details = result
        self.facts["manifest_sha256"] = result.get("manifest_sha256", "")
        self.facts["signed_by"] = result.get("signed_by", "")
        if result["status"] == "not signed":
            gate.status, gate.evidence = "MISSING", "not signed: run manifest.py sign"
        elif result["status"] == "passed":
            gate.status = "PASS"
            gate.evidence = (
                f"{len(result['files'])} files match the manifest signed by {result['signed_by']}; "
                f"skops types: {', '.join(result['skops_untrusted_types'])}"
            )
        else:
            gate.status, gate.evidence = "FAIL", "; ".join(result["problems"])

    def gate_quality(self, gate: Gate, log: Path):
        report = OUT / "gate_report.json"
        report.unlink(missing_ok=True)
        gates_file = "exercises/quality-gates/solution/test_gates.py"
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                gates_file,
                "-q",
                "-rs",
                "--tb=line",
                "--no-header",
                "-p",
                "no:cacheprovider",
            ],
            log,
            {"GATE_ALIAS": self.release["alias"], "GATE_REPORT": str(report)},
        )
        if not report.exists():
            gate.status, gate.evidence = "FAIL", "the gates did not run"
            return
        r = json.loads(report.read_text())
        measured = {k: v for g in r["gates"] for k, v in g["measured"].items()}
        counts = r["counts"]
        gate.details = r
        failed = [g["gate"] for g in r["gates"] if g["outcome"] in ("failed", "error")]
        gate.status = "FAIL" if failed else ("MISSING" if counts["skipped"] else "PASS")
        gate.evidence = (
            f"{counts['passed']} of {len(r['gates'])} gates passed on version {r.get('version')}: "
            + ", ".join(
                f"{k} {measured[k]}" for k in ("accuracy", "roc_auc", "margin", "rookie_accuracy") if k in measured
            )
            + (f"; failed: {', '.join(failed)}" if failed else "")
        )

    def gate_deployment(self, gate: Gate, log: Path):
        try:
            health = httpx2.get(f"{self.api_url}/health", timeout=10).json()
        except Exception as err:
            log.write_text(f"GET {self.api_url}/health\n{err}\n")
            gate.status, gate.evidence = "FAIL", f"the API at {self.api_url} does not answer: {err}"
            return
        log.write_text(f"GET {self.api_url}/health\n{json.dumps(health, indent=2)}\n")
        self.facts["health"] = health
        served = str(health.get("model_version"))
        gate.status = "PASS" if served == str(self.version.version) else "FAIL"
        gate.evidence = f"{self.api_url} serves {health.get('model_name')} version {served}" + (
            "" if gate.status == "PASS" else f", not version {self.version.version} that is under acceptance"
        )

    def gate_contract(self, gate: Gate, log: Path):
        pytest_gate(
            gate,
            "exercises/serve-model/solution/test_api.py",
            {"API_URL": self.api_url, "API_KEY": self.client_key},
            log,
        )

    def gate_access(self, gate: Gate, log: Path):
        pytest_gate(gate, self.access_tests, {"API_URL": self.api_url}, log)

    def gate_slo(self, gate: Gate, log: Path):
        prefix = OUT / "locust"
        code, output = run(
            ["bash", "exercises/serve-model/load_test.sh"],
            log,
            {"API_URL": self.api_url, "API_KEY": self.client_key, "OUT": str(prefix)},
        )
        rows = re.findall(r"^\s+(PASS|FAIL)\s+(requests|p95 latency|failure rate)\s+(\S+(?: ms)?)", output, re.M)
        verdict = re.search(r"^SLO (MET|BROKEN.*)$", output, re.M)
        if not verdict:
            gate.status, gate.evidence = "FAIL", "the load test did not run: " + output.strip()[-200:]
            return
        gate.status = "PASS" if code == 0 else "FAIL"
        gate.evidence = f"SLO {verdict.group(1)}: " + ", ".join(f"{n} {v}" for _, n, v in rows)

    def gate_dependencies(self, gate: Gate, log: Path):
        result_file = OUT / "dependencies.json"
        result_file.unlink(missing_ok=True)
        cmd = [sys.executable, str(HERE / "audit_deps.py"), "--json", str(result_file)]
        cmd += ["--image", self.args.image] if self.args.image else []
        code, _ = run(cmd, log)
        if not result_file.exists():
            gate.status, gate.evidence = "MISSING", "audit_deps.py did not run"
            return
        r = json.loads(result_file.read_text())
        gate.details = r
        if r["status"] == "error":
            gate.status, gate.evidence = "MISSING", f"the audit could not run: {r.get('error', '')[:160]}"
            return
        accepted = [f for f in r["findings"] if f["accepted_by"]]
        open_ = [f for f in r["findings"] if not f["accepted_by"]]
        gate.status = "FAIL" if open_ else "PASS"
        gate.evidence = (
            f"pip-audit, {r['target']}: {r['packages']} packages, {len(open_)} open findings, "
            f"{len(accepted)} accepted" + "".join(f"; {f['package']} {f['id']}" for f in open_[:3])
        )

    def gate_model_card(self, gate: Gate, log: Path):
        path = Path(self.evidence["model_card"])
        if not path.exists():
            log.write_text(f"{path} not found\n")
            gate.status, gate.evidence = "MISSING", f"no model card at {path}"
            return
        text = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)  # ignore the instructions in comments
        version = re.search(r"\*\*Model version:\*\*\s*(\S+)", text)
        todos = len(re.findall(r"\bTODO\b", text))
        problems = []
        if todos:
            problems.append(f"{todos} TODO left")
        if not version:
            problems.append('no "**Model version:**" line')
        elif version.group(1) != str(self.version.version):
            problems.append(f"it describes version {version.group(1)}, not {self.version.version}")
        log.write_text(f"{path}: {len(text)} characters\n" + "\n".join(problems) + "\n")
        gate.status = "MISSING" if problems else "PASS"
        gate.evidence = f"{path}" + (
            ": " + "; ".join(problems) if problems else f", for version {self.version.version}"
        )

    def gate_fairness(self, gate: Gate, log: Path):
        path = Path(self.evidence["fairness_report"])
        if not path.exists():
            log.write_text(f"{path} not found\n")
            gate.status, gate.evidence = "MISSING", f"no fairness report at {path}"
            return
        log.write_text(path.read_text())
        try:
            report = json.loads(path.read_text())
            passed, version = report["passed"], str(report["model_version"])
        except (ValueError, KeyError) as err:
            gate.status, gate.evidence = "MISSING", f"{path} cannot be read: {err}"
            return
        gate.details = report
        if version != str(self.version.version):
            gate.status, gate.evidence = "MISSING", f"{path} is for version {version}, not {self.version.version}"
            return
        checks = report.get("checks", [])
        worst = min(checks, key=lambda c: c.get("accuracy", 1), default=None)
        gate.status = "PASS" if passed else "FAIL"
        gate.evidence = f"{sum(bool(c.get('passed')) for c in checks)} of {len(checks)} subgroup checks passed" + (
            f"; lowest: {worst['name']} accuracy {worst['accuracy']}" if worst else ""
        )

    # ---- running and writing -----------------------------------------------------------

    def gates(self) -> list[Gate]:
        return [
            Gate("data", "Data validation and lineage", "02, 04"),
            Gate("integrity", "Integrity: signed SHA-256 manifest", "09"),
            Gate("quality", "Quality gates", "04"),
            Gate("deployment", "Deployed version", "05"),
            Gate("contract", "API contract tests", "05"),
            Gate("access", "Access tests", "09"),
            Gate("slo", "Load test and SLO", "05"),
            Gate("dependencies", "Dependency audit", "09"),
            Gate("model_card", "Model card", "08, 09"),
            Gate("fairness", "Subgroup and fairness evidence", "08"),
        ]

    def run_all(self) -> list[Gate]:
        OUT.mkdir(parents=True, exist_ok=True)
        gates = self.gates()
        name, v = self.release["model_name"], self.version.version
        print(f"Acceptance of {name} version {v} (@{self.release['alias']}), API {self.api_url}")
        for i, gate in enumerate(gates, 1):
            log = OUT / f"{i:02d}-{gate.key}.log"
            print(f"== {i}/{len(gates)} {gate.name} (lesson {gate.lesson})", flush=True)
            start = time.monotonic()
            try:
                getattr(self, f"gate_{gate.key}")(gate, log)
            except Exception as err:  # a gate that crashes gives no evidence
                gate.status, gate.evidence = "FAIL", f"{type(err).__name__}: {err}"
            gate.seconds = round(time.monotonic() - start, 1)
            gate.log = str(log)
            print(f"   {gate.status:<8} {gate.evidence}", flush=True)
        return gates

    def write(self, gates: list[Gate]) -> dict:
        name, v = self.release["model_name"], str(self.version.version)
        blocked = [g for g in gates if g.status != "PASS"]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if blocked:
            decision = {
                "value": "do not ship",
                "by": "accept.py (automatic)",
                "at": now,
                "via": "acceptance suite",
                "reason": "; ".join(
                    f"{g.name}: {'FAILED' if g.status == 'FAIL' else 'EVIDENCE MISSING'}" for g in blocked
                ),
            }
            recommendation = "do not ship"
        else:
            decision = {
                "value": "PENDING",
                "by": "",
                "at": "",
                "via": "",
                "reason": "every gate passed. Only a named approver can set the decision to ship (decide.py).",
            }
            recommendation = "ready for a human decision"
        facts = self.collect_facts(now)
        dossier = {
            "model_name": name,
            "model_version": v,
            "facts": facts,
            "owners": self.release["owners"],
            "approvers": self.release.get("approvers", []),
            "gates": [asdict(g) for g in gates],
            "recommendation": recommendation,
            "decision": decision,
        }
        markdown = render(dossier)
        dossier["dossier_sha256"] = body_sha256(markdown)
        DOSSIER.write_text(markdown)
        DOSSIER.with_suffix(".json").write_text(json.dumps(dossier, indent=2) + "\n")
        for key, value in {
            "acceptance.result": "passed" if not blocked else "blocked",
            "acceptance.at": now,
            "acceptance.dossier_sha256": dossier["dossier_sha256"],
            "release.decision": decision["value"],
        }.items():
            self.mlflow.set_model_version_tag(name, v, key, value)
        return dossier

    def collect_facts(self, now: str) -> dict:
        tracking = mlflow.get_tracking_uri()
        ui = tracking if tracking.startswith("http") else "(no MLflow UI: the store is the file " + tracking + ")"
        exp = self.run.info.experiment_id
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        image_id = ""
        if self.args.image:
            image_id = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", self.args.image], capture_output=True, text=True
            ).stdout.strip()
        return {
            **self.facts,
            "generated_at": now,
            "generated_by": "exercises/acceptance/accept.py" + (" --reference" if self.args.reference else ""),
            "alias": self.release["alias"],
            "run_id": self.version.run_id,
            "run_url": f"{ui}/#/experiments/{exp}/runs/{self.version.run_id}" if ui.startswith("http") else ui,
            "model_version_url": f"{ui}/#/models/{self.release['model_name']}/versions/{self.version.version}"
            if ui.startswith("http")
            else ui,
            "proposed_by": self.version.tags.get("proposed_by", "unknown"),
            "promoted_by": self.version.tags.get("promoted_by", "not promoted"),
            "promoted_at": self.version.tags.get("promoted_at", ""),
            "git_commit": commit + (" (with uncommitted changes)" if dirty else ""),
            "api_url": self.api_url,
            "image": f"{self.args.image} ({image_id[:19]})" if self.args.image else "not given",
            "github_run": (
                f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                f"{os.environ['GITHUB_RUN_ID']}"
                if os.environ.get("GITHUB_RUN_ID")
                else ""
            ),
        }


def body_sha256(markdown: str) -> str:
    """The fingerprint of the dossier without its decision block, which decide.py fills in."""
    start, end = markdown.index(DECISION_START), markdown.index(DECISION_END)
    return hashlib.sha256((markdown[:start] + markdown[end:]).encode()).hexdigest()


def decision_block(decision: dict, recommendation: str) -> str:
    return "\n".join(
        [
            DECISION_START,
            "| | |",
            "|---|---|",
            f"| **Decision** | **{decision['value']}** |",
            f"| Recommendation of the suite | {recommendation} |",
            f"| Set by | {decision['by'] or '(nobody yet)'} |",
            f"| At | {decision['at'] or '-'} |",
            f"| Via | {decision['via'] or '-'} |",
            f"| Reason | {decision['reason']} |",
            DECISION_END,
        ]
    )


def render(d: dict) -> str:
    f, g = d["facts"], d["gates"]
    passed = sum(x["status"] == "PASS" for x in g)
    lines = [
        f"# Release dossier: {d['model_name']} version {d['model_version']}",
        "",
        f"Generated by `{f['generated_by']}` at {f['generated_at']}. {passed} of {len(g)} gates passed.",
        "",
        "## Decision",
        "",
        decision_block(d["decision"], d["recommendation"]),
        "",
        "Only a person named in `release.yaml` (`approvers`), or a required reviewer of the GitHub environment "
        "`production`, can set the decision to **ship**, with `decide.py`. Any failed gate or missing evidence "
        "means **do not ship**.",
        "",
        "## The release",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{d['model_name']}` version **{d['model_version']}** (`@{f['alias']}`) |",
        f"| MLflow run | [{f['run_id']}]({f['run_url']}) |"
        if f["run_url"].startswith("http")
        else f"| MLflow run | `{f['run_id']}` {f['run_url']} |",
        f"| MLflow model version | [{d['model_name']}/{d['model_version']}]({f['model_version_url']}) |"
        if f["model_version_url"].startswith("http")
        else f"| MLflow model version | {d['model_name']}/{d['model_version']} |",
        f"| Data | `{f.get('data_path', '?')}`, md5 `{f.get('data_md5', '?')}` (trained on `{f.get('data_md5_trained', '?')}`) |",
        f"| Integrity manifest | sha256 `{f.get('manifest_sha256') or 'not signed'}`, signed by {f.get('signed_by') or '-'} |",
        f"| Proposed by / promoted by | {f['proposed_by']} / {f['promoted_by']} {f['promoted_at']} |",
        f"| Code | git commit `{f['git_commit']}` |",
        f"| Serving | {f['api_url']}, image {f['image']} |",
    ]
    if f["github_run"]:
        lines.append(f"| GitHub Actions run | {f['github_run']} |")
    lines += ["", "## Owners", "", "| Role | Owner |", "|---|---|"]
    lines += [f"| {role} | {who} |" for role, who in d["owners"].items()]
    lines += [f"| approvers | {', '.join(d['approvers']) or '(none)'} |"]
    lines += ["", "## Gates", "", "| # | Gate | Lesson | Result | Evidence | Controls |", "|---|---|---|---|---|---|"]
    for i, x in enumerate(g, 1):
        result = {"PASS": "PASS", "FAIL": "**FAIL**", "MISSING": "**EVIDENCE MISSING**"}[x["status"]]
        evidence = x["evidence"].replace("|", "\\|")
        nist, sdaia = CONTROLS[x["key"]]
        lines.append(f"| {i} | {x['name']} | {x['lesson']} | {result} | {evidence} | {nist}; {sdaia} |")
    lines += [
        "",
        "Logs: " + ", ".join(f"`{x['log']}`" for x in g),
        "",
        "Controls: NIST AI RMF 1.0 (NIST AI 100-1) subcategories and SDAIA AI Ethics Principles. "
        "They show which published control a gate gives evidence for; they are not a compliance claim.",
        "",
    ]
    return "\n".join(lines)


def write_summary(dossier: dict):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    d = dossier
    lines = [
        f"### Acceptance: {d['model_name']} version {d['model_version']}: {d['recommendation']}",
        "",
        f"**Decision: {d['decision']['value']}**"
        + (
            f" ({d['decision']['reason']})"
            if d["decision"]["value"] != "PENDING"
            else ": waiting for a named approver (job **Ship decision**)"
        ),
        "",
        "| # | Gate | Result | Evidence |",
        "|---|---|---|---|",
    ]
    for i, g in enumerate(d["gates"], 1):
        lines.append(f"| {i} | {g['name']} | {g['status']} | {g['evidence'].replace('|', '/')} |")
    with open(summary, "a") as f:
        f.write("\n".join(lines) + "\n\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="the Docker image of the API, for the dependency audit and the dossier")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument(
        "--reference", action="store_true", help="use the reference solution's access tests"
    )
    parser.add_argument("--release", type=Path, default=HERE / "release.yaml")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri  # the gates run as separate processes
    for variable in ("API_CLIENT_KEY", "API_ADMIN_KEY"):
        if not os.environ.get(variable):
            sys.exit(f"ERROR: {variable} is not set. Load the API keys first: set -a; source .secrets/api.env; set +a")

    acceptance = Acceptance(args)
    dossier = acceptance.write(acceptance.run_all())
    write_summary(dossier)
    d = dossier["decision"]
    print(f"\nDossier: {DOSSIER} (and {DOSSIER.with_suffix('.json')}). Logs: {OUT}/")
    print(
        f"DECISION: {d['value'].upper()}"
        + (
            f" ({d['reason']})"
            if d["value"] != "PENDING"
            else ": every gate passed. A named approver decides with decide.py."
        )
    )
    sys.exit(0 if dossier["recommendation"] == "ready for a human decision" else 1)


if __name__ == "__main__":
    main()
