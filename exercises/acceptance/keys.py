"""Make a client key and an admin key for the API and write them to .secrets/api.env.

The file is in .gitignore, so git never commits it, and .dockerignore keeps it out of the
image. The API reads the keys from environment variables:

    set -a; source .secrets/api.env; set +a            # for uvicorn and the tests
    docker run --env-file .secrets/api.env ...           # for the container

Usage:
    python exercises/acceptance/keys.py                  # refuses to overwrite existing keys
    python exercises/acceptance/keys.py --rotate         # new keys; the old ones stop working
    python exercises/acceptance/keys.py --out other.env  # another file
"""

import argparse
import os
import secrets
import subprocess
import sys
from pathlib import Path


def is_ignored(path: Path) -> bool:
    """True if git ignores the file, so that `git add .` cannot commit it by mistake."""
    result = subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(".secrets/api.env"))
    parser.add_argument("--rotate", action="store_true", help="replace existing keys")
    args = parser.parse_args()

    if args.out.exists() and not args.rotate:
        sys.exit(f"{args.out} exists. Use --rotate to make new keys (the old ones stop working).")
    if not is_ignored(args.out):
        sys.exit(f"REFUSED: git does not ignore {args.out}. Add it to .gitignore first, so that it is never committed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # 32 random bytes from the operating system's secure random source, as 43 URL-safe characters.
    content = (
        "# API keys for the f1-points-finish API (lesson 09). Do not commit, share or paste this file.\n"
        f"API_CLIENT_KEY={secrets.token_urlsafe(32)}\n"
        f"API_ADMIN_KEY={secrets.token_urlsafe(32)}\n"
    )
    # Only the owner may read the file.
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    os.chmod(args.out, 0o600)  # also when --rotate overwrote an older file
    print(f"Wrote a client key and an admin key to {args.out} (git ignores it).")
    print(f"Load them in this terminal with:  set -a; source {args.out}; set +a")


if __name__ == "__main__":
    main()
