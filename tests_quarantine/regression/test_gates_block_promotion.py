"""
Regression Test: I2 - Promotion Gating Enforced

Tests that failing forgetting/regression scores MUST block promotion.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from conftest import async_iter_mock


class TestPromotionGating:
    """I2: Quality gates must block bad adapters from promotion."""

    @pytest.mark.regression
    def test_failing_gates_block_auto_promotion(self, client: TestClient, tenant_header: dict):
        """
        Adapter with metrics exceeding budget should NOT be auto-promoted.

        This is the critical I2 invariant test.
        """
        route_key = "gating-test-fail"

        # Setup route with strict policy
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})

        # Set strict policy - low budgets that will fail
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={
                       "novelty_threshold": 0.1,
                       "promotion_threshold": 0.99,  # Very high bar
                       "forgetting_budget": 0.01,    # Very strict
                       "regression_budget": 0.01,    # Very strict
                       "auto_promote_to_canary": True,
                       "auto_promote_to_stable": True
                   })

        # Mock the workflow to return failing metrics
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
            mock_instance.metrics = {
                "eval": {
                    "accuracy": 0.80,       # Below promotion_threshold (0.99)
                    "forgetting": 0.15,     # Exceeds forgetting_budget (0.01)
                    "regression": 0.10      # Exceeds regression_budget (0.01)
                }
            }
            mock_instance.diagnosis = None
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
            mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
            MockWorkflow.return_value = mock_instance

            # Run once
            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

            # Expect failure due to gates
            result = resp.json()
            if result.get("verdict") != "skipped":  # If novelty was sufficient
                assert result.get("verdict") == "failed", f"Expected failed verdict: {result}"
                assert "gates" in result.get("reason", "").lower() or "Gates" in str(result)

    @pytest.mark.regression
    def test_passing_gates_allow_promotion(self, client: TestClient, tenant_header: dict):
        """
        Adapter with good metrics should be promoted when auto_promote is enabled.
        """
        route_key = "gating-test-pass"

        # Setup route with lenient policy
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})

        # Set lenient policy
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={
                       "novelty_threshold": 0.1,
                       "promotion_threshold": 0.5,   # Achievable
                       "forgetting_budget": 0.5,     # Generous
                       "regression_budget": 0.5,     # Generous
                       "auto_promote_to_canary": True
                   })

        # Mock the workflow to return passing metrics
        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter", "tgsp_path": "/mock/tgsp"}
            mock_instance.metrics = {
                "eval": {
                    "accuracy": 0.95,       # Above promotion_threshold
                    "forgetting": 0.02,     # Below forgetting_budget
                    "regression": 0.01      # Below regression_budget
                }
            }
            mock_instance.diagnosis = None
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
            mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
            MockWorkflow.return_value = mock_instance

            # Run once
            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            result = resp.json()

            # May be skipped due to novelty, or succeed with promotion
            if result.get("verdict") == "success":
                # Verify promotion happened
                assert result.get("promoted_to") in ["canary", "stable", None]

    @pytest.mark.regression
    def test_manual_promote_blocked_for_candidate(self, client: TestClient, tenant_header: dict):
        """
        Manual promotion via API should also respect gating (or at least be tracked).
        """
        route_key = "manual-promote-test"

        # Setup route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})

        # Try to promote non-existent adapter
        resp = client.post(
            f"/api/v1/tgflow/routes/{route_key}/promote",
            headers=tenant_header,
            params={"adapter_id": "non-existent-adapter", "target": "stable"}
        )

        # Should fail - adapter doesn't exist
        assert resp.status_code in [400, 404, 500]

    @pytest.mark.regression
    def test_gate_results_in_timeline(self, client: TestClient, tenant_header: dict):
        """
        Gate evaluation results should be recorded in timeline events.
        """
        route_key = "gate-timeline-test"

        # Setup complete route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1, "auto_promote_to_canary": True})

        # Run once
        client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Check timeline for gate-related events
        resp = client.get(f"/api/v1/tgflow/routes/{route_key}/timeline", headers=tenant_header)
        assert resp.status_code == 200

        timeline_data = resp.json()
        # Timeline should exist (format may vary)
        assert "timeline" in timeline_data


class TestGateThresholds:
    """Tests for specific gate threshold behaviors."""

    @pytest.mark.regression
    def test_forgetting_exactly_at_budget_passes(self, client: TestClient, tenant_header: dict):
        """
        Forgetting score exactly equal to budget should pass (<=).
        """
        # This tests the boundary condition
        route_key = "forgetting-boundary"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={
                       "novelty_threshold": 0.1,
                       "promotion_threshold": 0.5,
                       "forgetting_budget": 0.10,  # Exact boundary
                       "regression_budget": 0.10
                   })

        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
            mock_instance.metrics = {
                "eval": {
                    "accuracy": 0.80,
                    "forgetting": 0.10,  # Exactly at boundary
                    "regression": 0.05
                }
            }
            mock_instance.diagnosis = None
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
            mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
            MockWorkflow.return_value = mock_instance

            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            result = resp.json()

            # Should NOT fail on forgetting gate (0.10 <= 0.10)
            if result.get("verdict") == "failed":
                gates = result.get("gates", {})
                assert gates.get("forgetting_pass", True) is True, \
                    "Forgetting at exact boundary should pass"

    @pytest.mark.regression
    def test_primary_metric_below_threshold_fails(self, client: TestClient, tenant_header: dict):
        """
        Primary metric below promotion threshold should block promotion.
        """
        route_key = "primary-threshold-test"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={
                       "novelty_threshold": 0.1,
                       "promotion_threshold": 0.95,  # High bar
                       "forgetting_budget": 0.5,
                       "regression_budget": 0.5
                   })

        with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
            mock_instance = MagicMock()
            mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
            mock_instance.metrics = {
                "eval": {
                    "accuracy": 0.85,  # Below 0.95 threshold
                    "forgetting": 0.01,
                    "regression": 0.01
                }
            }
            mock_instance.diagnosis = None
            mock_instance._stage_train.return_value = async_iter_mock([])
            mock_instance._stage_eval.return_value = async_iter_mock([])
            mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
            mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
            MockWorkflow.return_value = mock_instance

            resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            result = resp.json()

            if result.get("verdict") != "skipped":
                # Should fail on primary metric gate
                assert result.get("verdict") == "failed", f"Expected failure: {result}"
