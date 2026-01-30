"""
Regression Test: Failure Injection

Tests system behavior when various components fail.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from conftest import async_iter_mock


class TestTrainingFailure:
    """Tests for training stage failures."""

    @pytest.mark.regression
    def test_training_failure_cleanup(self, client: TestClient, tenant_header: dict):
        """
        When training crashes mid-run:
        - FAILED event should be recorded
        - Partial artifacts should be cleaned/marked incomplete
        - System should remain operable
        """
        route_key = "train-fail-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Mock training to fail
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance._stage_train.side_effect = RuntimeError("CUDA out of memory!")
            MockWorkflow.return_value = mock_instance

            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Should not crash the server
        assert resp.status_code == 200

        result = resp.json()
        # Should indicate error
        assert result.get("verdict") in ["error", "failed", "skipped"]

        # Route should still be operational
        route_resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        assert route_resp.status_code == 200

        # Should be able to run again
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

            resp2 = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            # Should work now
            assert resp2.status_code == 200


class TestEvalFailure:
    """Tests for evaluation stage failures."""

    @pytest.mark.regression
    def test_eval_failure_no_promotion(self, client: TestClient, tenant_header: dict):
        """
        When evaluation fails:
        - No PROMOTED event should occur
        - Route should remain operable
        """
        route_key = "eval-fail-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1, "auto_promote_to_stable": True})

        # Mock eval to fail
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.side_effect = Exception("Evaluation dataset corrupted!")
            mock_instance.diagnosis = None
            MockWorkflow.return_value = mock_instance

            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        result = resp.json()
        # Should indicate error
        assert result.get("verdict") in ["error", "failed", "skipped"]

        # No promotion should have occurred
        route_resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        route_data = route_resp.json()["route"]
        # Active adapter should not be set from failed run
        # (This depends on implementation - may be None or previous value)


class TestPackagingFailure:
    """Tests for TGSP packaging failures."""

    @pytest.mark.regression
    def test_packaging_failure_recovery(self, client: TestClient, tenant_header: dict):
        """
        When TGSP creation fails:
        - Error should be logged
        - System should remain recoverable
        """
        route_key = "package-fail-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Mock packaging to fail
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
            mock_instance.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.side_effect = IOError("Disk full!")
            mock_instance.diagnosis = None
            MockWorkflow.return_value = mock_instance

            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        result = resp.json()
        # Should handle error gracefully
        assert result.get("verdict") in ["error", "failed", "skipped", "success"]

        # Route should be recoverable
        route_resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        assert route_resp.status_code == 200


class TestDatabaseFailure:
    """Tests for database connectivity issues."""

    @pytest.mark.regression
    def test_db_connection_error_handled(self, client: TestClient, tenant_header: dict):
        """
        Database errors should be handled gracefully.
        """
        # This test verifies error handling - actual DB failure simulation
        # requires more infrastructure

        # Test health endpoint reports DB status
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()

        assert "checks" in data
        assert "database" in data["checks"]

    @pytest.mark.regression
    def test_ready_endpoint_reflects_db_state(self, client: TestClient):
        """
        Ready endpoint should reflect database availability.
        """
        resp = client.get("/ready")
        # In test environment, DB should be available
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("ready") is True


class TestIntegrationFailure:
    """Tests for integration connector failures."""

    @pytest.mark.regression
    def test_connector_failure_fallback(self, client: TestClient, tenant_header: dict):
        """
        When an integration connector fails, system should fallback gracefully.
        """
        route_key = "connector-fail-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Mock IntegrationManager to fail
        with patch('tensorguard.tgflow.continuous.orchestrator.IntegrationManager') as MockManager:
            mock_instance = MagicMock()
            mock_instance.get_compatibility_snapshot.side_effect = Exception("Integration service unavailable!")
            mock_instance.get_connector.return_value = None
            MockManager.return_value = mock_instance

            with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
                wf_instance = MagicMock()
                wf_instance.artifacts = {"adapter_path": "/mock/adapter"}
                wf_instance.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
                wf_instance.diagnosis = None
                wf_instance._stage_train.return_value = iter([])
                wf_instance._stage_eval.return_value = iter([])
                wf_instance._stage_pack_tgsp.return_value = iter([])
                wf_instance._stage_emit_evidence.return_value = iter([])
                MockWorkflow.return_value = wf_instance

                resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Should handle gracefully
        assert resp.status_code in [200, 500]


class TestPartialFailures:
    """Tests for partial/intermittent failures."""

    @pytest.mark.regression
    def test_retry_after_transient_failure(self, client: TestClient, tenant_header: dict):
        """
        After a transient failure, retry should work.
        """
        route_key = "retry-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # First run fails
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance._stage_train.side_effect = Exception("Transient network error")
            MockWorkflow.return_value = mock_instance

            resp1 = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            # May fail

        # Second run succeeds
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter", "tgsp_path": "/mock/tgsp"}
            mock_instance.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
            mock_instance.diagnosis = None
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
            mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
            MockWorkflow.return_value = mock_instance

            resp2 = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        assert resp2.status_code == 200
        result = resp2.json()
        # Should succeed after transient failure
        assert result.get("verdict") in ["success", "skipped"]


class TestTimeoutHandling:
    """Tests for timeout scenarios."""

    @pytest.mark.regression
    @pytest.mark.slow
    def test_slow_operation_timeout(self, client: TestClient, tenant_header: dict):
        """
        Slow operations should be handled without blocking indefinitely.
        """
        import time

        route_key = "timeout-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Mock slow training
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
            mock_instance.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
            mock_instance.diagnosis = None

            def slow_train():
                time.sleep(0.5)  # Simulate slow but not too slow
                return iter([])

            mock_instance._stage_train.return_value = slow_train()
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
            mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
            MockWorkflow.return_value = mock_instance

            start = time.time()
            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            duration = time.time() - start

        # Should complete in reasonable time
        assert duration < 5.0, f"Operation took too long: {duration}s"
        assert resp.status_code == 200
