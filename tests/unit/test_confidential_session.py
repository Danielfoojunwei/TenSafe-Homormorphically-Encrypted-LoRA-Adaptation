"""
Unit tests for Confidential Session Manager.

Tests HPKE session lifecycle, encryption/decryption, and session management.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorguard.confidential.session import (
    ConfidentialSession,
    ConfidentialSessionManager,
    DEFAULT_SESSION_TTL_SECONDS,
    MAX_SESSION_TTL_SECONDS,
)
from tensorguard.tgsp.hpke_v03 import (
    generate_keypair,
    hpke_seal,
    hpke_open,
    public_key_to_bytes,
)


@pytest.fixture
def session_manager():
    return ConfidentialSessionManager(default_ttl=3600, max_sessions=100)


@pytest.fixture
def session(session_manager):
    return session_manager.create_session()


@pytest.fixture
def client_keypair():
    """Generate a client X25519 key pair."""
    return generate_keypair()


class TestSessionCreation:

    def test_create_session(self, session_manager):
        session = session_manager.create_session()
        assert session is not None
        assert session.session_id.startswith("cs-")
        assert session.is_active is True
        assert session.is_expired is False
        assert session.requests_processed == 0

    def test_create_session_with_ttl(self, session_manager):
        session = session_manager.create_session(ttl=60)
        assert session.is_active is True
        delta = (session.expires_at - session.created_at).total_seconds()
        assert 59 <= delta <= 61

    def test_create_session_max_ttl_enforced(self, session_manager):
        session = session_manager.create_session(ttl=999999)
        delta = (session.expires_at - session.created_at).total_seconds()
        assert delta <= MAX_SESSION_TTL_SECONDS + 1

    def test_create_session_with_attestation_binding(self, session_manager):
        session = session_manager.create_session(
            attestation_quote_hash="abc123def456"
        )
        assert session.attestation_quote_hash == "abc123def456"

    def test_create_session_has_keypair(self, session_manager):
        session = session_manager.create_session()
        assert session.server_private_key is not None
        assert session.server_public_key is not None
        assert len(session.server_public_key_bytes) == 32
        assert len(session.server_public_key_hex) == 64

    def test_max_sessions_enforced(self):
        manager = ConfidentialSessionManager(max_sessions=3)
        manager.create_session()
        manager.create_session()
        manager.create_session()
        with pytest.raises(RuntimeError, match="Maximum sessions"):
            manager.create_session()


class TestSessionLookup:

    def test_get_existing_session(self, session_manager, session):
        found = session_manager.get_session(session.session_id)
        assert found is not None
        assert found.session_id == session.session_id

    def test_get_nonexistent_session(self, session_manager):
        found = session_manager.get_session("cs-nonexistent")
        assert found is None

    def test_get_expired_session(self):
        manager = ConfidentialSessionManager(default_ttl=1)
        session = manager.create_session(ttl=1)
        sid = session.session_id
        time.sleep(1.1)
        found = manager.get_session(sid)
        assert found is None

    def test_destroy_session(self, session_manager, session):
        assert session_manager.destroy_session(session.session_id) is True
        assert session_manager.get_session(session.session_id) is None

    def test_destroy_nonexistent_session(self, session_manager):
        assert session_manager.destroy_session("cs-nonexistent") is False


class TestSessionEncryption:

    def test_decrypt_request(self, session, client_keypair):
        """Test that session can decrypt HPKE-encrypted data."""
        client_priv, client_pub = client_keypair

        # Client encrypts a request to the server's public key
        plaintext = json.dumps({"model": "test", "messages": []}).encode()
        sealed = hpke_seal(
            plaintext=plaintext,
            recipient_public_key=session.server_public_key,
            info=session.session_id.encode(),
        )

        # Server decrypts inside session
        decrypted = session.decrypt_request(sealed)
        assert decrypted == plaintext
        assert session.requests_processed == 1
        assert session.bytes_decrypted == len(plaintext)

    def test_encrypt_response(self, session, client_keypair):
        """Test that session can encrypt response for client."""
        client_priv, client_pub = client_keypair
        session.client_public_key = client_pub

        response = b'{"result": "encrypted-response"}'
        sealed = session.encrypt_response(response, client_pub)

        assert "enc" in sealed
        assert "ciphertext" in sealed
        assert session.bytes_encrypted == len(response)

        # Client decrypts
        decrypted = hpke_open(
            sealed=sealed,
            recipient_private_key=client_priv,
            info=session.session_id.encode(),
        )
        assert decrypted == response

    def test_full_roundtrip(self, session, client_keypair):
        """Full request/response encryption roundtrip."""
        client_priv, client_pub = client_keypair
        session.client_public_key = client_pub

        # Client -> Server (encrypted request)
        request_data = {"model": "llama-3", "messages": [{"role": "user", "content": "Hello"}]}
        request_bytes = json.dumps(request_data).encode()
        sealed_request = hpke_seal(
            plaintext=request_bytes,
            recipient_public_key=session.server_public_key,
            info=session.session_id.encode(),
        )

        # Server decrypts
        decrypted_request = session.decrypt_request(sealed_request)
        assert json.loads(decrypted_request) == request_data

        # Server -> Client (encrypted response)
        response_data = {"choices": [{"message": {"content": "Hi!"}}]}
        response_bytes = json.dumps(response_data).encode()
        sealed_response = session.encrypt_response(response_bytes, client_pub)

        # Client decrypts
        decrypted_response = hpke_open(
            sealed=sealed_response,
            recipient_private_key=client_priv,
            info=session.session_id.encode(),
        )
        assert json.loads(decrypted_response) == response_data

    def test_decrypt_wrong_key_fails(self, session):
        """Encryption to wrong key should fail on decrypt."""
        wrong_priv, wrong_pub = generate_keypair()

        plaintext = b"test"
        sealed = hpke_seal(
            plaintext=plaintext,
            recipient_public_key=wrong_pub,  # Encrypted to wrong key
            info=session.session_id.encode(),
        )

        with pytest.raises(ValueError, match="HPKE decryption failed"):
            session.decrypt_request(sealed)

    def test_multiple_requests(self, session, client_keypair):
        """Session handles multiple sequential requests."""
        client_priv, client_pub = client_keypair

        for i in range(5):
            plaintext = f'{{"request": {i}}}'.encode()
            sealed = hpke_seal(
                plaintext=plaintext,
                recipient_public_key=session.server_public_key,
                info=session.session_id.encode(),
            )
            decrypted = session.decrypt_request(sealed)
            assert decrypted == plaintext

        assert session.requests_processed == 5


class TestSessionMetadata:

    def test_get_metadata(self, session):
        meta = session.get_metadata()
        assert meta["session_id"] == session.session_id
        assert meta["is_active"] is True
        assert meta["requests_processed"] == 0
        assert meta["attestation_bound"] is False

    def test_metadata_with_attestation(self, session_manager):
        session = session_manager.create_session(
            attestation_quote_hash="test-hash"
        )
        meta = session.get_metadata()
        assert meta["attestation_bound"] is True


class TestSessionManagerStats:

    def test_statistics(self, session_manager):
        session_manager.create_session()
        session_manager.create_session()
        stats = session_manager.get_statistics()
        assert stats["active_sessions"] == 2
        assert stats["total_created"] == 2

    def test_cleanup(self):
        manager = ConfidentialSessionManager(default_ttl=1)
        manager.create_session(ttl=1)
        manager.create_session(ttl=1)
        time.sleep(1.1)
        removed = manager.cleanup()
        assert removed == 2
        assert manager.get_statistics()["active_sessions"] == 0
