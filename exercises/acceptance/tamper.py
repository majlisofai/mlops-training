"""For the trainer: tamper with the release after it was signed off, to test the integrity gate.

--target model   Replaces the model file of the version behind @champion, in the MLflow
                 artifact store on disk, with a pruned copy: the same forest with only its
                 first 30 trees. It still loads, answers every request and passes the contract
                 tests. This is what someone with write access to the artifact store could do.
--target data    Changes one value in f1/data/races.csv: the driver_form of one 2024 entry,
                 by +0.01. The data is still valid.
--restore        Puts the original files back.

The originals are kept in results/tamper/. Restart (or reload) the API after tampering
with the model: the API loads the model file at start-up.

Usage:
    python exercises/acceptance/tamper.py --target model
    python exercises/acceptance/tamper.py --target data
    python exercises/acceptance/tamper.py --restore
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import skops.io as sio
import yaml
from mlflow import MlflowClient

HERE = Path(__file__).parent
BACKUP = Path("results/tamper")
KEEP_TREES = 30


def model_file(client: MlflowClient, release: dict, artifacts_root: Path) -> Path:
    """Find model.skops of the version behind the alias in the artifact store on disk."""
    version = client.get_model_version_by_alias(release["model_name"], release["alias"])
    location = client.get_logged_model(version.source.rsplit("/", 1)[-1]).artifact_location
    url = urlparse(location)
    if url.scheme == "mlflow-artifacts":  # the MLflow server stores them under --artifacts-destination
        path = artifacts_root / url.path.lstrip("/") / "model.skops"
    elif url.scheme in ("", "file"):
        path = Path(url.path) / "model.skops"
    else:
        sys.exit(f"tamper.py works on a local artifact store only, not on {location}")
    if not path.exists():
        sys.exit(f"{path} not found. Run tamper.py from the repository root, next to mlartifacts/.")
    return path


def tamper_model(path: Path, trusted: list[str]):
    model = sio.load(path, trusted=trusted)
    forest = model[-1]
    forest.estimators_ = forest.estimators_[:KEEP_TREES]
    forest.n_estimators = KEEP_TREES
    sio.dump(model, path)
    print(f"TAMPERED: {path} now holds a forest of {KEEP_TREES} trees instead of 300. It still loads and answers.")


def tamper_data(path: Path):
    """Change one value on one line. Every other byte of the file stays the same."""
    lines = path.read_text().splitlines(keepends=True)
    header = lines[0].rstrip("\n").split(",")
    season, form, race, driver = (header.index(c) for c in ("season", "driver_form", "race_id", "driver_code"))
    rows_2024 = [i for i, line in enumerate(lines[1:], 1) if line.split(",")[season] == "2024"]
    i = rows_2024[100]
    fields = lines[i].rstrip("\n").split(",")
    before = fields[form]
    fields[form] = f"{float(before) + 0.01:.2f}"
    lines[i] = ",".join(fields) + "\n"
    path.write_text("".join(lines))
    print(
        f"TAMPERED: {path}, line {i + 1} ({fields[race]} {fields[driver]}): driver_form {before} -> {fields[form]}. "
        "The data is still valid."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", choices=["model", "data"])
    group.add_argument("--restore", action="store_true")
    parser.add_argument(
        "--artifacts", type=Path, default=Path("mlartifacts"), help="the MLflow --artifacts-destination"
    )
    parser.add_argument("--release", type=Path, default=HERE / "release.yaml")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    release = yaml.safe_load(args.release.read_text())
    index = BACKUP / "originals.json"
    originals = json.loads(index.read_text()) if index.exists() else {}

    if args.restore:
        if not originals:
            sys.exit("Nothing to restore.")
        for target, backup in originals.items():
            shutil.copy2(backup, target)
            print(f"RESTORED: {target}")
        index.unlink()
        return

    if args.target == "model":
        path = model_file(MlflowClient(), release, args.artifacts)
    else:
        path = Path(release["integrity"]["files"][0])  # f1/data/races.csv
    if str(path) in originals:
        sys.exit(f"{path} is already tampered with. Run --restore first.")
    BACKUP.mkdir(parents=True, exist_ok=True)
    backup = BACKUP / f"{len(originals)}-{path.name}"
    shutil.copy2(path, backup)
    originals[str(path)] = str(backup)
    index.write_text(json.dumps(originals, indent=2))

    if args.target == "model":
        tamper_model(path, release["integrity"]["allowed_skops_types"])
    else:
        tamper_data(path)


if __name__ == "__main__":
    main()
