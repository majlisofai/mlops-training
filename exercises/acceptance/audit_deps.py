"""Check the packages of the serving image for known vulnerabilities with pip-audit.

With --image, it lists the packages that are really installed in the image (with Python's
importlib.metadata, inside a container) and audits exactly those, including what the base
image brought along. Without --image, it audits
requirements-serve.txt (pip-audit resolves it in a temporary environment first).

pip-audit looks the packages up in PyPI's vulnerability data (its default service), so it
needs the internet. Exit 0: no finding, or every finding is
accepted in release.yaml. Exit 1: a finding that nobody accepted. Exit 2: the audit could not run.

Usage:
    python exercises/acceptance/audit_deps.py --image lab09-api:1
    python exercises/acceptance/audit_deps.py
    python exercises/acceptance/audit_deps.py -r .devcontainer/requirements.txt   # the codespace tools
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import yaml

HERE = Path(os.path.relpath(Path(__file__).parent))


# Lists every installed distribution as name==version. It needs only Python, not pip,
# so it works on an image from which pip was removed.
LIST_PACKAGES = (
    "import importlib.metadata as m; "
    "print('\\n'.join(sorted({f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions()})))"
)


def frozen_image_packages(image: str, out: Path) -> Path:
    """Write the packages that are really installed in the image, pinned, one per line."""
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image, "-c", LIST_PACKAGES], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot list the packages in {image}: {result.stderr.strip()[:300]}")
    out.write_text(result.stdout)
    return out


def run_pip_audit(requirements: Path, frozen: bool) -> dict:
    # A frozen list pins every package, so pip-audit can check it as it is, without resolving.
    extra = ["--disable-pip", "--no-deps"] if frozen else []
    with tempfile.NamedTemporaryFile(suffix=".json") as report:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "-r",
                str(requirements),
                *extra,
                "--progress-spinner",
                "off",
                "-f",
                "json",
                "-o",
                report.name,
            ],
            capture_output=True,
            text=True,
        )
        # pip-audit exits 1 when it finds vulnerabilities; anything else without a report is an error.
        text = Path(report.name).read_text()
        if result.returncode not in (0, 1) or not text:
            raise RuntimeError(f"pip-audit failed: {(result.stderr or result.stdout).strip()[-500:]}")
        return json.loads(text)


def findings(report: dict) -> list[dict]:
    """One entry per (package, vulnerability id), without duplicates."""
    seen, found = set(), []
    for dep in report["dependencies"]:
        for vuln in dep.get("vulns", []):
            key = (dep["name"], vuln["id"])
            if key not in seen:
                seen.add(key)
                found.append(
                    {
                        "package": dep["name"],
                        "version": dep.get("version"),
                        "id": vuln["id"],
                        "aliases": vuln.get("aliases", []),
                        "fix_versions": vuln.get("fix_versions", []),
                    }
                )
    return found


def is_accepted(finding: dict, accepted: list[dict]) -> dict | None:
    """Return the acceptance entry that covers this finding, if it has not expired."""
    ids = {finding["id"], *finding["aliases"]}
    for entry in accepted:
        if entry.get("id") in ids and entry.get("package", finding["package"]) == finding["package"]:
            if date.fromisoformat(str(entry["expires"])) >= date.today():
                return entry
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", help="audit the packages installed in this Docker image")
    parser.add_argument("-r", "--requirements", type=Path, default=HERE / "requirements-serve.txt")
    parser.add_argument("--release", type=Path, default=HERE / "release.yaml")
    parser.add_argument("--json", type=Path, help="write the result to this file")
    args = parser.parse_args()
    accepted = (yaml.safe_load(args.release.read_text()).get("dependencies") or {}).get("accepted") or []

    target = f"image {args.image}" if args.image else str(args.requirements)
    result = {"target": target, "status": "error", "packages": 0, "findings": []}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if args.image:
                report = run_pip_audit(frozen_image_packages(args.image, Path(tmp) / "freeze.txt"), frozen=True)
            else:
                report = run_pip_audit(args.requirements, frozen=False)
    except RuntimeError as err:
        result["error"] = str(err)
        print(f"AUDIT ERROR for {target}: {err}")
        write(args.json, result)
        sys.exit(2)

    result["packages"] = len(report["dependencies"])
    for f in findings(report):
        entry = is_accepted(f, accepted)
        f["accepted_by"] = entry.get("accepted_by") if entry else None
        result["findings"].append(f)
    open_findings = [f for f in result["findings"] if not f["accepted_by"]]
    result["status"] = "failed" if open_findings else "passed"
    write(args.json, result)

    print(
        f"Dependency audit of {target}: {result['packages']} packages, {len(result['findings'])} known vulnerabilities"
    )
    for f in result["findings"]:
        fix = ", ".join(f["fix_versions"]) or "no fix yet"
        state = f"accepted by {f['accepted_by']}" if f["accepted_by"] else "OPEN"
        print(f"  {state:<20} {f['package']} {f['version']}  {f['id']}  (fixed in: {fix})")
    if open_findings:
        print(
            f"AUDIT FAILED: {len(open_findings)} open findings. Upgrade, or accept each one in release.yaml with a reason."
        )
        sys.exit(1)
    print("AUDIT OK")


def write(path: Path | None, result: dict):
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
