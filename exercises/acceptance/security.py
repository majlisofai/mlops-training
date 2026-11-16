"""API-key authentication with two roles for the f1-points-finish API (lesson 09).

Every caller sends a key in the X-API-Key header. Each key has one role:

    client   may ask for predictions        POST /v1/predict
    admin    may reload the model           POST /admin/reload

The keys come from two environment variables, API_CLIENT_KEY and API_ADMIN_KEY. They are
never in the code, the image or the repository. keys.py makes a pair for you.

The answers follow HTTP:
    401 Unauthorized   no key, or a key that the API does not know ("who are you?")
    403 Forbidden      a known key with the wrong role ("I know you, but you may not do this")

You do not need to change this file. app.py uses require_role() to protect its endpoints.
"""

import hashlib
import hmac
import os
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

ROLES = {"client": "API_CLIENT_KEY", "admin": "API_ADMIN_KEY"}  # role -> environment variable
MIN_KEY_LENGTH = 32  # keys.py makes 43 characters (32 random bytes)

# FastAPI adds this header to the /docs page: the "Authorize" button asks for the key.
# auto_error=False: we answer a missing key ourselves, with 401 and a clear message.
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Your API key. A client key for /v1/predict, an admin key for /admin/reload.",
)


@dataclass(frozen=True)
class Caller:
    role: str
    fingerprint: str  # the first 12 hex digits of the key's SHA-256: safe to log, unlike the key


def _digest(key: str) -> bytes:
    return hashlib.sha256(key.encode()).digest()


def load_keys() -> dict[str, bytes]:
    """Read one key per role from the environment and keep only their SHA-256 digests.

    Fail closed: if a key is missing or too short, the API does not start. An API that
    starts without keys would be open to everyone, and nobody would notice.
    """
    keys = {}
    for role, variable in ROLES.items():
        key = os.environ.get(variable, "")
        if len(key) < MIN_KEY_LENGTH:
            raise RuntimeError(
                f"{variable} is not set or shorter than {MIN_KEY_LENGTH} characters. "
                "The API does not start without its keys. Make them with exercises/acceptance/keys.py."
            )
        keys[role] = _digest(key)
    if keys["client"] == keys["admin"]:
        raise RuntimeError("API_CLIENT_KEY and API_ADMIN_KEY must be different keys.")
    return keys


def identify(key: str, keys: dict[str, bytes]) -> Caller | None:
    """Return the caller for a key, or None if the API does not know the key.

    hmac.compare_digest takes the same time whether the first or the last byte differs,
    so an attacker cannot guess a key byte by byte from the response time.
    """
    digest = _digest(key)
    role = None
    for name, known in keys.items():
        if hmac.compare_digest(digest, known):
            role = name
    return Caller(role, digest.hex()[:12]) if role else None


def require_role(role: str):
    """A FastAPI dependency: the request must carry a key with this role.

    Usage in app.py:
        def predict(request: PredictRequest, caller: Caller = Depends(require_role("client"))):
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; the roles are {sorted(ROLES)}")

    def check(request: Request, key: str | None = Depends(api_key_header)) -> Caller:
        if not key:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="missing API key: send it in the X-API-Key header",
                headers={"WWW-Authenticate": "APIKey"},
            )
        caller = identify(key, request.app.state.keys)
        if caller is None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED, detail="unknown API key", headers={"WWW-Authenticate": "APIKey"}
            )
        if caller.role != role:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=f"this key has the role {caller.role!r}; {request.url.path} needs the role {role!r}",
            )
        return caller

    return check
