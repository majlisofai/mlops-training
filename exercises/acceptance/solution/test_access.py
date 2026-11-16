"""Access tests for the f1-points-finish API: who may call which endpoint.

Reference solution. Try exercises/acceptance/test_access.py first.
The fixtures (client, client_key, admin_key, race) come from exercises/acceptance/conftest.py.

Usage:
    pytest exercises/acceptance/solution/test_access.py
    API_URL=http://127.0.0.1:8000 pytest exercises/acceptance/solution/test_access.py
"""

import secrets
import subprocess
from pathlib import Path

APP_FILE = Path(__file__).parent / "app.py"  # conftest.py starts this app in the test process


def key(value: str) -> dict:
    return {"X-API-Key": value}


def test_health_is_public(client):
    """GET /health needs no key: a load balancer must be able to check the API."""
    assert client.get("/health").status_code == 200


def test_predict_without_a_key_is_401(client, race):
    """POST /v1/predict without a key gets 401, and the answer says how to authenticate."""
    response = client.post("/v1/predict", json=race)
    assert response.status_code == 401, f"status {response.status_code}: /v1/predict answers without a key"
    assert response.headers.get("WWW-Authenticate") == "APIKey"


def test_reload_without_a_key_is_401(client):
    """POST /admin/reload without a key gets 401. Nobody anonymous may change the model."""
    response = client.post("/admin/reload")
    assert response.status_code == 401, f"status {response.status_code}: /admin/reload answers without a key"


def test_unknown_key_is_401(client, race):
    """A key that the API does not know gets 401 on both endpoints."""
    stranger = key(secrets.token_urlsafe(32))
    response = client.post("/v1/predict", json=race, headers=stranger)
    assert response.status_code == 401, f"status {response.status_code}: /v1/predict accepts an unknown key"
    response = client.post("/admin/reload", headers=stranger)
    assert response.status_code == 401, f"status {response.status_code}: /admin/reload accepts an unknown key"


def test_client_key_can_predict(client, race, client_key):
    """A client key gets predictions: 20 cars in, 20 predictions out."""
    response = client.post("/v1/predict", json=race, headers=key(client_key))
    assert response.status_code == 200, response.text
    assert len(response.json()["predictions"]) == 20


def test_wrong_role_is_403(client, race, client_key, admin_key):
    """A known key with the wrong role gets 403: a client cannot reload, an admin cannot predict."""
    response = client.post("/admin/reload", headers=key(client_key))
    assert response.status_code == 403, f"status {response.status_code}: a client key reloaded the model"
    assert "needs the role 'admin'" in response.json()["detail"]

    response = client.post("/v1/predict", json=race, headers=key(admin_key))
    assert response.status_code == 403, f"status {response.status_code}: an admin key got predictions"
    assert "needs the role 'client'" in response.json()["detail"]


def test_admin_key_can_reload(client, admin_key):
    """An admin key reloads the model, and the API then serves the version that /health names."""
    response = client.post("/admin/reload", headers=key(admin_key))
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == client.get("/health").json()["model_version"]


def test_keys_are_not_committed(client_key, admin_key):
    """No file that git tracks contains a key, and git ignores the key file."""
    for value in (client_key, admin_key):
        found = subprocess.run(["git", "grep", "-l", "-F", "-e", value], capture_output=True, text=True)
        assert found.returncode != 0, f"a key is in a file that git tracks: {found.stdout.split()}"
        assert found.returncode == 1, f"git grep failed: {found.stderr}"
    ignored = subprocess.run(["git", "check-ignore", "-q", ".secrets/api.env"])
    assert ignored.returncode == 0, ".secrets/api.env is not in .gitignore"
