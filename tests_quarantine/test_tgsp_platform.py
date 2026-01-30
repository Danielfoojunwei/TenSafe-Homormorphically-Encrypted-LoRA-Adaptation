
import pytest
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives import serialization
import secrets
import json


@pytest.fixture(scope="module")
def platform_client():
    """
    Lazy import the platform app to avoid database initialization at import time.
    The database engine is created when the module is imported, so we defer this.
    """
    from tensorguard.platform.main import app
    from tensorguard.platform.api.identity_endpoints import verify_fleet_auth
    from tensorguard.platform.models.core import Fleet

    mock_fleet = Fleet(id="fleet_1", tenant_id="tenant_1", name="test_fleet", api_key_hash="hash")
    app.dependency_overrides[verify_fleet_auth] = lambda: mock_fleet
    return TestClient(app)


def test_attestation_verify(platform_client):
    payload = {
        "agent_id": "agent_1",
        "fleet_id": "fleet_1",
        "claims": {"os": "linux", "version": "1.0"},
        "nonce": secrets.token_hex(16),
        "signature": "mock_sig"
    }
    response = platform_client.post("/api/v1/attestation/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "allow"
    assert "attestation_id" in data
    assert "claims_hash" in data


def test_key_release_v03(platform_client):
    from cryptography.hazmat.primitives.asymmetric import x25519
    priv = x25519.X25519PrivateKey.generate()
    pub = priv.public_key()
    pub_hex = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()

    payload = {
        "package_id": "pkg_1",
        "recipient_id": "fleet:h1",
        "tgsp_version": "0.3",
        "manifest_hash": "hash1",
        "claims_hash": "chash1",
        "device_hpke_pubkey": pub_hex
    }
    response = platform_client.post("/api/v1/tgsp/key-release", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "allow"
    assert "rewrapped" in data
    assert data["rewrapped"]["alg"] == "V03_HPKE_MVP"
