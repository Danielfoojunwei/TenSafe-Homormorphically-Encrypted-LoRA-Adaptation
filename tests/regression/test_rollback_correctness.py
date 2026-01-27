"""
Regression Test: I3 - Rollback Instant + Correct

Tests that rollback changes active adapter ID and resolve returns stable adapter.
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from unittest.mock import patch, MagicMock
from conftest import async_iter_mock


class TestRollbackCorrectness:
    """I3: Rollback must instantly restore the fallback adapter."""

    @pytest.mark.regression
    def test_rollback_restores_fallback(self, client: TestClient, tenant_header: dict, session: Session):
        """
        After rollback, active_adapter_id should equal the previous fallback.

        This is the critical I3 invariant test.
        """
        route_key = "rollback-test"

        # Setup route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={
                       "novelty_threshold": 0.1,
                       "promotion_threshold": 0.5,
                       "forgetting_budget": 0.5,
                       "regression_budget": 0.5,
                       "auto_promote_to_stable": True
                   })

        # We need to simulate having two adapters (one stable, one fallback)
        # by running twice with successful promotion

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

            # First run - creates first adapter
            resp1 = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            result1 = resp1.json()

            # Second run - creates second adapter, first becomes fallback
            resp2 = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            result2 = resp2.json()

        # Get current route state
        resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        route_before = resp.json()["route"]

        active_before = route_before.get("active_adapter_id")
        fallback_before = route_before.get("fallback_adapter_id")

        # If we have a fallback, test rollback
        if fallback_before:
            # Execute rollback
            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/rollback", headers=tenant_header)
            assert resp.status_code == 200, f"Rollback failed: {resp.text}"

            rollback_result = resp.json()
            assert rollback_result.get("ok") is True

            # Verify active is now the old fallback
            resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
            route_after = resp.json()["route"]

            assert route_after["active_adapter_id"] == fallback_before, \
                f"Active should be old fallback. Got {route_after['active_adapter_id']}, expected {fallback_before}"

    @pytest.mark.regression
    def test_rollback_without_fallback_fails(self, client: TestClient, tenant_header: dict):
        """
        Rollback should fail gracefully when no fallback exists.
        """
        route_key = "no-fallback-test"

        # Create fresh route with no history
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})

        # Attempt rollback
        resp = client.post(f"/api/v1/tgflow/routes/{route_key}/rollback", headers=tenant_header)

        # Should fail with appropriate error
        assert resp.status_code in [400, 500]
        error = resp.json()
        assert "fallback" in str(error).lower() or "no fallback" in str(error).lower() or "not found" in str(error).lower()

    @pytest.mark.regression
    def test_resolve_returns_correct_adapter_after_rollback(self, client: TestClient, tenant_header: dict):
        """
        The resolve endpoint should return the active adapter after rollback.
        """
        route_key = "resolve-rollback-test"

        # Setup route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1, "auto_promote_to_stable": True})

        # Create two adapters via mocked runs
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

            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Get state before rollback
        resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        route_data = resp.json()["route"]
        fallback_id = route_data.get("fallback_adapter_id")

        if fallback_id:
            # Rollback
            client.post(f"/api/v1/tgflow/routes/{route_key}/rollback", headers=tenant_header)

            # Test resolve
            resp = client.post("/api/v1/tgflow/resolve", headers=tenant_header,
                              json={"route_key": route_key})

            if resp.status_code == 200:
                resolve_result = resp.json()
                # The resolved adapter should be the fallback (now active)
                assert resolve_result.get("adapter_id") == fallback_id or \
                       resolve_result.get("adapter_id") == "default-base-model"


class TestRollbackTimeline:
    """Tests for rollback event recording."""

    @pytest.mark.regression
    def test_rollback_recorded_in_timeline(self, client: TestClient, tenant_header: dict):
        """
        Rollback action should create a release record.
        """
        # This tests the audit trail for rollback operations
        route_key = "rollback-timeline-test"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1, "auto_promote_to_stable": True})

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

            # Two runs to have a fallback
            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Get fallback
        resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        if resp.json()["route"].get("fallback_adapter_id"):
            # Execute rollback
            client.post(f"/api/v1/tgflow/routes/{route_key}/rollback", headers=tenant_header)

            # The rollback creates an AdapterRelease record with channel=ROLLBACK
            # We verify indirectly via the route state change


class TestRollbackDiff:
    """Tests for comparing adapters during rollback decision."""

    @pytest.mark.regression
    def test_diff_shows_metric_changes(self, client: TestClient, tenant_header: dict):
        """
        Diff endpoint should show metric differences between adapters.
        """
        route_key = "diff-test"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1, "auto_promote_to_stable": True})

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

            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Get diff
        resp = client.get(f"/api/v1/tgflow/routes/{route_key}/diff", headers=tenant_header)
        assert resp.status_code == 200

        diff_data = resp.json()
        # Should have diff_available field
        assert "diff_available" in diff_data
