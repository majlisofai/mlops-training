"""Fingerprint the release with SHA-256, sign it off in MLflow, and verify it later.

sign    Downloads the files of the model version behind the alias from the MLflow registry
        and computes the SHA-256 of each one, and of the data and release files listed in
        release.yaml (integrity.files). Writes them to manifest.json, logs the manifest to the
        model's MLflow run (integrity/manifest-v<N>.json) and tags the model version with the
        manifest's own SHA-256, who signed it and when.
verify  Downloads the model files again, hashes everything again and compares with the
        signed manifest. Exit 0: every file is unchanged. Exit 1: a file changed, or the model
        file holds a type that release.yaml does not allow. Exit 2: the version was never signed.

A SHA-256 changes completely when one byte of the file changes. Equal hashes mean equal files.

Usage:
    python exercises/acceptance/manifest.py sign --signer "Your Name"
    python exercises/acceptance/manifest.py verify
    python exercises/acceptance/manifest.py verify --json results/acceptance/integrity.json
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

import mlflow  # noqa: E402
import skops.io as sio  # noqa: E402
import yaml  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

HERE = Path(__file__).parent
TAG = "integrity.manifest_sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_manifest(manifest: dict) -> str:
    """The fingerprint of the manifest itself, over a canonical form of its JSON."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_files(name: str, version: str) -> dict[str, dict]:
    """Download the files of a model version from the registry and hash them."""
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version}", dst_path=tmp))
        files = {
            f"model/{p.relative_to(local).as_posix()}": {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(local.rglob("*"))
            if p.is_file()
        }
        untrusted = sorted(set(sio.get_untrusted_types(file=str(local / "model.skops"))))
        declared = (
            yaml.safe_load((local / "MLmodel").read_text())["flavors"].get("sklearn", {}).get("skops_trusted_types")
            or []
        )
    return files, untrusted, sorted(declared)


def repo_files(paths: list[str]) -> dict[str, dict]:
    files = {}
    for path in paths:
        p = Path(path)
        files[path] = {"sha256": sha256_file(p), "bytes": p.stat().st_size} if p.exists() else None
    return files


def resolve(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        sys.exit(f"ERROR: {name}@{alias} not found. Is the MLflow server running, and is there a @{alias}?")


def sign(client, release, version, signer: str, force: bool, out: Path):
    name = release["model_name"]
    if TAG in version.tags and not force:
        sys.exit(
            f"REFUSED: {name} version {version.version} was already signed by "
            f"{version.tags.get('integrity.signed_by')} at {version.tags.get('integrity.signed_at')}. "
            "Signing again would hide a change. Use --force only if you know why the files changed."
        )
    files, untrusted, _ = model_files(name, version.version)
    files.update(repo_files(release["integrity"]["files"]))
    missing = [p for p, f in files.items() if f is None]
    if missing:
        sys.exit(f"ERROR: cannot sign, files not found: {', '.join(missing)}")
    manifest = {
        "model_name": name,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "signed_by": signer,
        "signed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skops_untrusted_types": untrusted,
        "files": files,
    }
    digest = sha256_manifest(manifest)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    with tempfile.TemporaryDirectory() as tmp:
        logged = Path(tmp) / f"manifest-v{version.version}.json"
        logged.write_text(json.dumps(manifest, indent=2) + "\n")
        client.log_artifact(version.run_id, str(logged), artifact_path="integrity")
    tags = {
        TAG: digest,
        "integrity.signed_by": signer,
        "integrity.signed_at": manifest["signed_at"],
        "integrity.files": str(len(files)),
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, version.version, key, value)

    print(f"Manifest of {name} version {version.version} ({len(files)} files):")
    for path, f in files.items():
        print(f"  {f['sha256'][:16]}  {f['bytes']:>10}  {path}")
    print(
        f"SIGNED: manifest sha256 {digest[:16]}... by {signer}. Written to {out}, logged to run "
        f"{version.run_id} as integrity/manifest-v{version.version}.json, tagged {TAG}."
    )


def verify(client, release, version) -> dict:
    """Return the result of the integrity check as a dict (see main for the exit codes)."""
    name = release["model_name"]
    result = {"model_name": name, "model_version": str(version.version), "problems": [], "files": {}}
    signed = version.tags.get(TAG)
    if not signed:
        result.update(status="not signed", problems=[f"{name} version {version.version} has no tag {TAG}"])
        return result
    with tempfile.TemporaryDirectory() as tmp:
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=version.run_id, artifact_path=f"integrity/manifest-v{version.version}.json", dst_path=tmp
            )
        except Exception as err:
            result.update(
                status="not signed", problems=[f"the signed manifest is missing from run {version.run_id}: {err}"]
            )
            return result
        manifest = json.loads(Path(path).read_text())
    result.update(signed_by=manifest["signed_by"], signed_at=manifest["signed_at"], manifest_sha256=signed)
    problems = result["problems"]
    if sha256_manifest(manifest) != signed:
        problems.append("the manifest itself changed after the sign-off (its sha256 differs from the tag)")

    current, untrusted, declared = model_files(name, version.version)
    current.update(repo_files([p for p in manifest["files"] if not p.startswith("model/")]))
    for path in sorted(manifest["files"].keys() | current.keys()):
        before, now = manifest["files"].get(path), current.get(path)
        if before is None:
            state = "NEW"
        elif now is None:
            state = "MISSING"
        else:
            state = "OK" if now["sha256"] == before["sha256"] else "CHANGED"
        result["files"][path] = {
            "state": state,
            "signed": (before or {}).get("sha256"),
            "now": (now or {}).get("sha256"),
        }
        if state != "OK":
            problems.append(f"{path}: {state}")

    allowed = set(release["integrity"]["allowed_skops_types"])
    for label, types in (("model.skops holds", untrusted), ("MLmodel trusts", declared)):
        extra = sorted(set(types) - allowed)
        if extra:
            problems.append(f"{label} types that release.yaml does not allow: {', '.join(extra)}")
    result["skops_untrusted_types"] = untrusted
    result["status"] = "failed" if problems else "passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["sign", "verify"])
    parser.add_argument("--signer", help="who signs the manifest off (sign)")
    parser.add_argument("--force", action="store_true", help="sign again a version that is already signed")
    parser.add_argument("--release", type=Path, default=HERE / "release.yaml")
    parser.add_argument("--out", type=Path, default=Path("manifest.json"), help="local copy of the manifest (sign)")
    parser.add_argument("--json", type=Path, help="write the verify result to this file")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    release = yaml.safe_load(args.release.read_text())
    version = resolve(client, release["model_name"], release["alias"])

    if args.action == "sign":
        if not args.signer:
            sys.exit('ERROR: sign needs --signer "Your Name": a sign-off without a name is worth nothing.')
        sign(client, release, version, args.signer, args.force, args.out)
        return

    result = verify(client, release, version)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Integrity of {release['model_name']} version {version.version}:")
    for path, f in result["files"].items():
        print(f"  {f['state']:<8} {path}")
    if result["status"] == "not signed":
        print(f"NOT SIGNED: {result['problems'][0]}")
        sys.exit(2)
    print(f"  skops types in model.skops: {', '.join(result['skops_untrusted_types']) or 'none'}")
    if result["problems"]:
        print("INTEGRITY FAILED:\n" + "\n".join(f"- {p}" for p in result["problems"]))
        sys.exit(1)
    print(
        f"INTEGRITY OK: {len(result['files'])} files match the manifest signed by "
        f"{result['signed_by']} at {result['signed_at']}."
    )


if __name__ == "__main__":
    main()
