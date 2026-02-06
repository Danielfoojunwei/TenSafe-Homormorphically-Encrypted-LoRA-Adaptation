"""
End-to-End Integration Test for Confidential Inference.

Tests the full flow:
    1. Client creates a confidential session (gets TEE attestation + server pubkey)
    2. Client verifies attestation
    3. Client encrypts prompt with HPKE
    4. Server decrypts inside TEE, runs inference
    5. Server encrypts response, generates privacy receipt
    6. Client decrypts response and verifies receipt
    7. Session is destroyed

This tests the complete integration of:
- TDX/SEV-SNP attestation providers (simulation)
- HPKE key exchange
- Confidential session manager
- Encrypted request/response middleware
- Privacy receipt generation and verification
- FastAPI API endpoints
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorguard.attestation.tdx import TDXAttestationProvider, TDXVerificationPolicy
from tensorguard.attestation.sev import SEVSNPAttestationProvider, SNPVerificationPolicy
from tensorguard.attestation.provider import VerificationPolicy
from tensorguard.confidential.session import ConfidentialSessionManager
from tensorguard.confidential.middleware import ConfidentialInferenceMiddleware
from tensorguard.confidential.receipt import PrivacyReceiptGenerator
from tensorguard.tgsp.hpke_v03 import (
    generate_keypair,
    hpke_seal,
    hpke_open,
    public_key_to_bytes,
)


# ---- Fixtures ----


@pytest.fixture
def tdx_provider():
    return TDXAttestationProvider(use_simulation=True)


@pytest.fixture
def sev_provider():
    return SEVSNPAttestationProvider(use_simulation=True)


@pytest.fixture
def session_manager():
    return ConfidentialSessionManager(default_ttl=3600)


@pytest.fixture
def middleware(session_manager):
    return ConfidentialInferenceMiddleware(session_manager=session_manager)


@pytest.fixture
def receipt_generator():
    return PrivacyReceiptGenerator(
        tee_platform="intel-tdx",
        attestation_quote_hash="e2e-test-hash",
    )


@pytest.fixture
def client_keypair():
    return generate_keypair()


# ---- E2E Tests ----


class TestConfidentialInferenceE2ETDX:
    """End-to-end test with Intel TDX attestation."""

    def test_full_confidential_flow_tdx(
        self, tdx_provider, session_manager, middleware, receipt_generator, client_keypair
    ):
        """Complete confidential inference flow with TDX."""
        client_priv, client_pub = client_keypair

        # --- Step 1: Client requests attestation ---
        client_nonce = os.urandom(32)
        attestation_quote = tdx_provider.generate_quote(
            nonce=client_nonce,
            extra_data=b"tensafe-confidential-inference",
        )

        # --- Step 2: Client verifies attestation ---
        policy = TDXVerificationPolicy(
            policy_id="client-policy",
            name="Client Verification",
            max_quote_age_seconds=300,
        )
        verification = tdx_provider.verify_quote(
            attestation_quote, policy, expected_nonce=client_nonce
        )
        assert verification.verified is True, (
            f"Attestation verification failed: {verification.failure_reasons}"
        )
        assert verification.platform_info["tee_type"] == "TDX"

        # --- Step 3: Create confidential session ---
        import hashlib

        quote_hash = hashlib.sha256(attestation_quote.quote_data).hexdigest()
        session = session_manager.create_session(
            attestation_quote_hash=quote_hash,
        )
        assert session.is_active
        assert session.attestation_quote_hash == quote_hash

        # --- Step 4: Client encrypts prompt ---
        prompt_data = {
            "model": "meta-llama/Llama-3-8B",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is homomorphic encryption?"},
            ],
            "max_tokens": 256,
            "temperature": 0.7,
        }
        prompt_bytes = json.dumps(prompt_data).encode("utf-8")

        sealed_request = hpke_seal(
            plaintext=prompt_bytes,
            recipient_public_key=session.server_public_key,
            info=session.session_id.encode(),
        )

        # --- Step 5: Server unwraps request inside TEE ---
        envelope = {
            "session_id": session.session_id,
            "encrypted_payload": sealed_request,
            "client_public_key": public_key_to_bytes(client_pub).hex(),
        }
        plaintext_request, returned_session, _ = middleware.unwrap_request(envelope)

        assert plaintext_request == prompt_data
        assert returned_session.session_id == session.session_id
        assert session.requests_processed == 1

        # --- Step 6: Server runs inference (mock) ---
        inference_result = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "meta-llama/Llama-3-8B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Homomorphic encryption allows computation on encrypted data.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

        # --- Step 7: Generate privacy receipt ---
        receipt = receipt_generator.generate(
            session_id=session.session_id,
            he_mode="HE_ONLY",
            he_backend="CKKS-MOAI",
            adapter_encrypted=True,
            he_metrics={"rotations": 0, "operations": 12, "compute_time_ms": 3.2},
            tssp_hash="sha256:test-package-hash",
            dp_certificate={"epsilon": 8.0, "delta": 1e-5},
            latency_ms=53.0,
        )
        assert receipt.verify() is True

        # --- Step 8: Server encrypts response ---
        response_envelope = middleware.wrap_response(
            response_data=inference_result,
            session=session,
            privacy_receipt=receipt.to_dict(),
        )

        assert "encrypted_response" in response_envelope
        assert "privacy_receipt" in response_envelope

        # --- Step 9: Client decrypts response ---
        encrypted_response = response_envelope["encrypted_response"]
        decrypted_bytes = hpke_open(
            sealed=encrypted_response,
            recipient_private_key=client_priv,
            info=session.session_id.encode(),
        )
        decrypted_response = json.loads(decrypted_bytes)
        assert decrypted_response == inference_result
        assert decrypted_response["choices"][0]["message"]["content"].startswith(
            "Homomorphic"
        )

        # --- Step 10: Client verifies privacy receipt ---
        receipt_data = response_envelope["privacy_receipt"]
        assert receipt_data["tee_attestation"]["platform"] == "intel-tdx"
        assert receipt_data["tee_attestation"]["verified"] is True
        assert receipt_data["he_execution"]["mode"] == "HE_ONLY"
        assert receipt_data["he_execution"]["adapter_encrypted"] is True
        assert receipt_data["he_execution"]["rotations"] == 0
        assert receipt_data["adapter_provenance"]["dp_certificate"]["epsilon"] == 8.0
        assert receipt_data["audit_hash"] != ""

        # Reconstruct receipt and verify hash
        reconstructed = PrivacyReceiptGenerator(
            tee_platform="intel-tdx",
            attestation_quote_hash="e2e-test-hash",
        ).generate(
            session_id=session.session_id,
            he_mode="HE_ONLY",
            he_backend="CKKS-MOAI",
            adapter_encrypted=True,
            he_metrics={"rotations": 0, "operations": 12, "compute_time_ms": 3.2},
            tssp_hash="sha256:test-package-hash",
            dp_certificate={"epsilon": 8.0, "delta": 1e-5},
            latency_ms=53.0,
        )
        # The audit hash verifies the claims are consistent
        assert reconstructed.verify() is True

        # --- Step 11: Destroy session ---
        destroyed = session_manager.destroy_session(session.session_id)
        assert destroyed is True
        assert session_manager.get_session(session.session_id) is None


class TestConfidentialInferenceE2ESEV:
    """End-to-end test with AMD SEV-SNP attestation."""

    def test_full_confidential_flow_sev(
        self, sev_provider, session_manager, middleware, client_keypair
    ):
        """Complete confidential inference flow with SEV-SNP."""
        client_priv, client_pub = client_keypair

        # Step 1: Attestation
        nonce = os.urandom(32)
        quote = sev_provider.generate_quote(nonce=nonce)

        # Step 2: Verify
        policy = SNPVerificationPolicy(
            policy_id="sev-e2e",
            name="SEV E2E Policy",
            max_quote_age_seconds=300,
        )
        result = sev_provider.verify_quote(quote, policy, expected_nonce=nonce)
        assert result.verified is True
        assert result.platform_info["tee_type"] == "SEV-SNP"

        # Step 3: Create session
        import hashlib

        session = session_manager.create_session(
            attestation_quote_hash=hashlib.sha256(quote.quote_data).hexdigest(),
        )

        # Step 4: Encrypt prompt
        prompt = {"model": "test", "prompt": "Hello from SEV-SNP"}
        sealed = hpke_seal(
            plaintext=json.dumps(prompt).encode(),
            recipient_public_key=session.server_public_key,
            info=session.session_id.encode(),
        )

        # Step 5: Decrypt
        envelope = {
            "session_id": session.session_id,
            "encrypted_payload": sealed,
            "client_public_key": public_key_to_bytes(client_pub).hex(),
        }
        decrypted, sess, _ = middleware.unwrap_request(envelope)
        assert decrypted == prompt

        # Step 6: Encrypt response
        response = {"result": "SEV-SNP protected"}
        resp_envelope = middleware.wrap_response(
            response_data=response,
            session=sess,
        )

        # Step 7: Client decrypts
        dec_bytes = hpke_open(
            sealed=resp_envelope["encrypted_response"],
            recipient_private_key=client_priv,
            info=session.session_id.encode(),
        )
        assert json.loads(dec_bytes) == response


class TestMultipleSessionsE2E:
    """Test concurrent sessions."""

    def test_multiple_independent_sessions(
        self, session_manager, middleware, client_keypair
    ):
        """Multiple sessions should not interfere with each other."""
        sessions = []
        for _ in range(5):
            s = session_manager.create_session()
            sessions.append(s)

        client_priv, client_pub = client_keypair

        # Send different messages to each session
        for i, session in enumerate(sessions):
            msg = {"id": i, "content": f"message-{i}"}
            sealed = hpke_seal(
                plaintext=json.dumps(msg).encode(),
                recipient_public_key=session.server_public_key,
                info=session.session_id.encode(),
            )
            envelope = {
                "session_id": session.session_id,
                "encrypted_payload": sealed,
                "client_public_key": public_key_to_bytes(client_pub).hex(),
            }
            decrypted, _, _ = middleware.unwrap_request(envelope)
            assert decrypted["id"] == i
            assert decrypted["content"] == f"message-{i}"

        # Verify each session has 1 request
        for session in sessions:
            assert session.requests_processed == 1


class TestCrossSessionIsolation:
    """Test that sessions are cryptographically isolated."""

    def test_cross_session_decryption_fails(self, session_manager):
        """Data encrypted for one session cannot be decrypted by another."""
        s1 = session_manager.create_session()
        s2 = session_manager.create_session()

        plaintext = b"secret-for-session-1"
        sealed = hpke_seal(
            plaintext=plaintext,
            recipient_public_key=s1.server_public_key,
            info=s1.session_id.encode(),
        )

        # Decrypt with correct session works
        decrypted = s1.decrypt_request(sealed)
        assert decrypted == plaintext

        # Decrypt with wrong session fails
        with pytest.raises(ValueError, match="HPKE decryption failed"):
            s2.decrypt_request(sealed)


class TestReceiptChainE2E:
    """Test privacy receipt hash chain across multiple requests."""

    def test_receipt_chain_across_requests(
        self, session_manager, middleware, receipt_generator, client_keypair
    ):
        """Privacy receipts should form a verifiable hash chain."""
        client_priv, client_pub = client_keypair
        session = session_manager.create_session()

        receipts = []
        for i in range(3):
            # Encrypt request
            msg = {"model": "test", "messages": [{"role": "user", "content": f"q{i}"}]}
            sealed = hpke_seal(
                plaintext=json.dumps(msg).encode(),
                recipient_public_key=session.server_public_key,
                info=session.session_id.encode(),
            )
            envelope = {
                "session_id": session.session_id,
                "encrypted_payload": sealed,
                "client_public_key": public_key_to_bytes(client_pub).hex(),
            }
            middleware.unwrap_request(envelope)

            # Generate receipt
            receipt = receipt_generator.generate(
                session_id=session.session_id,
                he_mode="HE_ONLY",
                latency_ms=10.0 * (i + 1),
            )
            receipts.append(receipt)

        # Verify chain
        assert receipts[0].previous_audit_hash is None
        assert receipts[1].previous_audit_hash == receipts[0].audit_hash
        assert receipts[2].previous_audit_hash == receipts[1].audit_hash

        # All receipts self-verify
        for r in receipts:
            assert r.verify() is True


class TestAttestationFactoryIntegration:
    """Test attestation factory with new TDX/SEV providers."""

    def test_factory_creates_tdx(self):
        from tensorguard.attestation.factory import (
            create_attestation_provider,
            AttestationConfig,
        )

        config = AttestationConfig(provider="tdx", is_production=False)
        provider = create_attestation_provider(config)
        assert provider.is_available is True
        quote = provider.generate_quote()
        assert quote.quote_id.startswith("tdx-")

    def test_factory_creates_sev(self):
        from tensorguard.attestation.factory import (
            create_attestation_provider,
            AttestationConfig,
        )

        config = AttestationConfig(provider="sev-snp", is_production=False)
        provider = create_attestation_provider(config)
        assert provider.is_available is True
        quote = provider.generate_quote()
        assert quote.quote_id.startswith("snp-")


class TestConfidentialAPIE2E:
    """E2E test with the FastAPI router (no vLLM engine)."""

    def test_api_session_and_inference(self, tdx_provider, client_keypair):
        """Test the API endpoints end-to-end."""
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("fastapi not installed")

        from tensorguard.confidential.api import create_confidential_router

        app = FastAPI()
        router = create_confidential_router(
            engine=None,  # Mock engine
            attestation_provider=tdx_provider,
            require_api_key=False,
        )
        app.include_router(router)
        client = TestClient(app)
        client_priv, client_pub = client_keypair

        # 1. Create session
        resp = client.post(
            "/v1/confidential/session",
            json={"ttl_seconds": 3600},
        )
        assert resp.status_code == 200
        session_data = resp.json()
        assert session_data["tee_platform"] == "sgx"  # TDX uses SGX DCAP
        session_id = session_data["session_id"]
        server_pub_hex = session_data["server_public_key"]

        # 2. Reconstruct server public key
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

        server_pub = X25519PublicKey.from_public_bytes(
            bytes.fromhex(server_pub_hex)
        )

        # 3. Encrypt request
        prompt = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hello TEE!"}],
        }
        sealed = hpke_seal(
            plaintext=json.dumps(prompt).encode(),
            recipient_public_key=server_pub,
            info=session_id.encode(),
        )

        # 4. Send encrypted chat completion
        resp = client.post(
            "/v1/confidential/chat/completions",
            json={
                "session_id": session_id,
                "encrypted_payload": sealed,
                "client_public_key": public_key_to_bytes(client_pub).hex(),
            },
        )
        assert resp.status_code == 200
        result = resp.json()

        assert "encrypted_response" in result
        assert "privacy_receipt" in result

        # 5. Decrypt response
        decrypted_bytes = hpke_open(
            sealed=result["encrypted_response"],
            recipient_private_key=client_priv,
            info=session_id.encode(),
        )
        response = json.loads(decrypted_bytes)
        assert response["object"] == "chat.completion"
        assert len(response["choices"]) > 0

        # 6. Verify privacy receipt
        receipt = result["privacy_receipt"]
        assert receipt["session_id"] == session_id
        assert receipt["audit_hash"] != ""

        # 7. Check session status
        resp = client.get(f"/v1/confidential/session/{session_id}")
        assert resp.status_code == 200
        status = resp.json()
        assert status["is_active"] is True
        assert status["requests_processed"] == 1

        # 8. Destroy session
        resp = client.delete(f"/v1/confidential/session/{session_id}")
        assert resp.status_code == 200

        # 9. Verify session is gone
        resp = client.get(f"/v1/confidential/session/{session_id}")
        assert resp.status_code == 404

    def test_api_attestation_endpoint(self, tdx_provider):
        """Test the attestation convenience endpoint."""
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("fastapi not installed")

        from tensorguard.confidential.api import create_confidential_router

        app = FastAPI()
        router = create_confidential_router(
            engine=None,
            attestation_provider=tdx_provider,
            require_api_key=False,
        )
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/v1/confidential/attestation")
        assert resp.status_code == 200
        data = resp.json()
        assert "attestation_quote" in data
        assert "server_public_key" in data
        assert "session_id" in data
        assert data["attestation_quote"] != "simulation"  # TDX sim produces real bytes

    def test_api_stats_endpoint(self, tdx_provider):
        """Test statistics endpoint."""
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("fastapi not installed")

        from tensorguard.confidential.api import create_confidential_router

        app = FastAPI()
        router = create_confidential_router(
            engine=None,
            attestation_provider=tdx_provider,
            require_api_key=False,
        )
        app.include_router(router)
        client = TestClient(app)

        resp = client.get("/v1/confidential/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert "active_sessions" in stats

    def test_api_invalid_session(self, tdx_provider):
        """Test error handling for invalid session."""
        try:
            from fastapi.testclient import TestClient
            from fastapi import FastAPI
        except ImportError:
            pytest.skip("fastapi not installed")

        from tensorguard.confidential.api import create_confidential_router

        app = FastAPI()
        router = create_confidential_router(
            engine=None,
            attestation_provider=tdx_provider,
            require_api_key=False,
        )
        app.include_router(router)
        client = TestClient(app)

        resp = client.post(
            "/v1/confidential/chat/completions",
            json={
                "session_id": "cs-nonexistent",
                "encrypted_payload": {"enc": "aa", "ciphertext": "bb"},
                "client_public_key": "cc" * 32,
            },
        )
        assert resp.status_code == 404
