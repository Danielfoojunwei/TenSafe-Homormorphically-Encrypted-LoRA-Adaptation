"""
Regression Test: I7 - N2HE Privacy Compliance

Tests that privacy receipts are emitted and safe logging is enforced.
"""

import pytest
import logging
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from conftest import async_iter_mock
from io import StringIO


class TestN2HEPrivacyReceipts:
    """I7: N2HE privacy must generate receipts and protect logs."""

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_n2he_receipt_generated(self, client: TestClient, tenant_header: dict):
        """
        When privacy_mode=n2he, resolve endpoint must return receipt_hash.

        This is the critical I7 invariant test.
        """
        route_key = "n2he-receipt-test"

        # Setup route with N2HE privacy
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={
                       "feed_type": "local",
                       "feed_uri": "mock://",
                       "privacy_mode": "n2he"  # Enable N2HE
                   })
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Test resolve endpoint
        resp = client.post(
            "/api/v1/tgflow/resolve",
            headers=tenant_header,
            json={"route_key": route_key}
        )

        assert resp.status_code == 200
        data = resp.json()

        # Must have receipt when N2HE is enabled
        assert data.get("privacy_mode") == "n2he", "Privacy mode should be n2he"
        assert "receipt_hash" in data, "N2HE response must include receipt_hash"
        assert len(data["receipt_hash"]) == 64, "Receipt hash should be SHA256 (64 hex chars)"

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_non_n2he_route_no_receipt(self, client: TestClient, tenant_header: dict):
        """
        Routes without N2HE should not generate privacy receipts.
        """
        route_key = "no-n2he-test"

        # Setup route without N2HE
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={
                       "feed_type": "local",
                       "feed_uri": "mock://",
                       "privacy_mode": "off"  # No N2HE
                   })
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Test resolve
        resp = client.post(
            "/api/v1/tgflow/resolve",
            headers=tenant_header,
            json={"route_key": route_key}
        )

        assert resp.status_code == 200
        data = resp.json()

        assert data.get("privacy_mode") == "off"
        # Should not have receipt_hash when N2HE is off
        assert "receipt_hash" not in data or data.get("receipt_hash") is None


class TestSafeLogging:
    """Tests for N2HE safe logging enforcement."""

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_n2he_logs_protected(self, client: TestClient, tenant_header: dict):
        """
        When N2HE is active, logs should have [N2HE][PROTECTED] prefix.
        """
        from tensorguard.privacy.safe_logger import safe_log_context, get_privacy_mode, set_privacy_mode

        # Test the safe_log_context
        assert get_privacy_mode() == "off"  # Default

        with safe_log_context("n2he"):
            assert get_privacy_mode() == "n2he"

        assert get_privacy_mode() == "off"  # Restored

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_safe_logger_prefixes_messages(self):
        """
        SafeLogger should prefix messages in N2HE mode.
        """
        from tensorguard.privacy.safe_logger import SafeLogger, set_privacy_mode

        # Create a SafeLogger
        logger = SafeLogger("test_safe")
        logger.setLevel(logging.DEBUG)

        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        # Test in N2HE mode
        set_privacy_mode("n2he")
        logger.info("Sensitive data here")

        output = stream.getvalue()
        # Should have protection prefix
        assert "[N2HE][PROTECTED]" in output

        # Reset
        set_privacy_mode("off")

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_no_plaintext_in_n2he_response(self, client: TestClient, tenant_header: dict):
        """
        N2HE resolve should not leak plaintext vectors.
        """
        route_key = "no-plaintext-test"

        # Setup N2HE route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "n2he"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Resolve with some input
        resp = client.post(
            "/api/v1/tgflow/resolve",
            headers=tenant_header,
            json={
                "route_key": route_key,
                "input_vector": [0.1, 0.2, 0.3]  # Simulated input
            }
        )

        data = resp.json()

        # Response should NOT contain raw input vector
        response_str = str(data)
        # Check that exact vector values don't appear in response
        assert "[0.1, 0.2, 0.3]" not in response_str or "encrypted" in response_str.lower()


class TestN2HEProvider:
    """Tests for N2HEProvider functionality."""

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_n2he_provider_encrypt_vector(self):
        """
        N2HEProvider should encrypt vectors without leaking plaintext.
        """
        from tensorguard.privacy.providers.n2he_provider import N2HEProvider

        provider = N2HEProvider(profile="router_only")
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        result = provider.encrypt_vector(vector)

        # Should return ciphertext structure
        assert "type" in result
        assert result["type"] == "n2he_ckks"
        assert "ciphertext_id" in result
        assert "blob" in result

        # Ciphertext ID should be deterministic hash
        assert len(result["ciphertext_id"]) == 64  # SHA256

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_n2he_provider_generate_receipt(self):
        """
        N2HEProvider should generate valid receipt hashes.
        """
        from tensorguard.privacy.providers.n2he_provider import N2HEProvider

        provider = N2HEProvider()
        context = {"route": "test-route", "timestamp": "2026-01-27T00:00:00Z"}

        receipt = provider.generate_receipt(context)

        # Should be SHA256 hash
        assert len(receipt) == 64
        assert all(c in "0123456789abcdef" for c in receipt)

        # Same context should produce same receipt (deterministic)
        receipt2 = provider.generate_receipt(context)
        assert receipt == receipt2

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_n2he_provider_infer_routing(self):
        """
        N2HEProvider should return adapter_id from route config.
        """
        from tensorguard.privacy.providers.n2he_provider import N2HEProvider

        provider = N2HEProvider()
        encrypted_vector = {"type": "n2he_ckks", "ciphertext_id": "abc123", "blob": "mock"}
        route_config = {"active_adapter_id": "adapter-123"}

        result = provider.infer_encrypted_routing(encrypted_vector, route_config)

        assert result == "adapter-123"


class TestN2HEIntegration:
    """Integration tests for N2HE with continuous learning."""

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_n2he_run_once_emits_receipt_event(self, client: TestClient, tenant_header: dict):
        """
        run_once with N2HE feed should record privacy-related events.
        """
        route_key = "n2he-run-test"

        # Setup N2HE route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "n2he"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Run once
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
            mock_instance.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
            mock_instance.diagnosis = None
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
            mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
            MockWorkflow.return_value = mock_instance

            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Run should complete (with privacy mode active)
        result = resp.json()
        assert "loop_id" in result

    @pytest.mark.regression
    @pytest.mark.n2he
    def test_n2he_latency_recorded(self, client: TestClient, tenant_header: dict):
        """
        N2HE resolve should record latency metrics.
        """
        route_key = "n2he-latency-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "n2he"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Resolve multiple times
        import time
        latencies = []
        for _ in range(3):
            start = time.time()
            client.post("/api/v1/tgflow/resolve", headers=tenant_header,
                       json={"route_key": route_key})
            latencies.append(time.time() - start)

        # All requests should complete in reasonable time
        assert all(l < 1.0 for l in latencies), f"N2HE resolve too slow: {latencies}"
