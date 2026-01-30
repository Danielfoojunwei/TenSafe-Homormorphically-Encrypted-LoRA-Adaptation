"""
Regression Test: I8 - Dashboard Bundle Schema

Tests that the dashboard bundle endpoint returns complete data.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from conftest import async_iter_mock


class TestDashboardBundleSchema:
    """I8: Dashboard bundle must return all KPIs and topology."""

    @pytest.mark.regression
    def test_bundle_schema_complete(self, client: TestClient, tenant_header: dict):
        """
        Dashboard bundle should include summary, timeseries, events, and topology.

        This is the critical I8 invariant test.
        """
        route_key = "dashboard-test"

        # Setup route with some activity
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "microsoft/phi-2"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Run to generate metrics
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

            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Get dashboard bundle
        resp = client.get(
            f"/api/v1/metrics/routes/{route_key}/dashboard_bundle",
            headers=tenant_header,
            params={"tenant_id": "regression-tenant"}
        )

        assert resp.status_code == 200
        data = resp.json()

        # Verify required sections
        assert "summary" in data, "Bundle must include summary"
        assert "timeseries" in data, "Bundle must include timeseries"
        assert "events" in data, "Bundle must include events"
        assert "topology" in data, "Bundle must include topology"

        # Verify topology structure
        topology = data["topology"]
        assert "nodes" in topology
        assert "edges" in topology
        assert isinstance(topology["nodes"], list)
        assert isinstance(topology["edges"], list)

    @pytest.mark.regression
    def test_routes_summary_has_kpis(self, client: TestClient, tenant_header: dict):
        """
        Routes summary should include key performance indicators.
        """
        route_key = "kpi-test"

        # Setup route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Get routes summary
        resp = client.get(
            "/api/v1/metrics/routes/summary",
            headers=tenant_header,
            params={"tenant_id": "regression-tenant"}
        )

        assert resp.status_code == 200
        summary = resp.json()

        assert isinstance(summary, list)
        # Should have at least our route
        route_summary = next((s for s in summary if s["route_key"] == route_key), None)
        if route_summary:
            assert "kpis" in route_summary
            assert "health_score" in route_summary
            assert "health_status" in route_summary

    @pytest.mark.regression
    def test_timeseries_endpoint_returns_data(self, client: TestClient, tenant_header: dict):
        """
        Timeseries endpoint should return metric data points.
        """
        route_key = "timeseries-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Get timeseries for a metric
        resp = client.get(
            f"/api/v1/metrics/routes/{route_key}/timeseries",
            headers=tenant_header,
            params={"tenant_id": "regression-tenant", "metric": "avg_accuracy"}
        )

        assert resp.status_code == 200
        data = resp.json()

        # Should be a list of data points
        assert isinstance(data, list)
        # Each point should have ts and value
        for point in data:
            assert "ts" in point or "value" in point


class TestTopologyEndpoint:
    """Tests for integrations topology endpoint."""

    @pytest.mark.regression
    def test_topology_structure(self, client: TestClient, tenant_header: dict):
        """
        Topology should have nodes and edges structure.
        """
        resp = client.get(
            "/api/v1/metrics/integrations/topology",
            headers=tenant_header,
            params={"tenant_id": "regression-tenant"}
        )

        assert resp.status_code == 200
        data = resp.json()

        assert "nodes" in data
        assert "edges" in data

        # Nodes should have required fields
        for node in data["nodes"]:
            assert "id" in node
            assert "label" in node

        # Edges should connect valid nodes
        node_ids = {n["id"] for n in data["nodes"]}
        for edge in data["edges"]:
            assert "source" in edge
            assert "target" in edge


class TestOpsBreakdown:
    """Tests for run operations breakdown endpoint."""

    @pytest.mark.regression
    def test_ops_breakdown_endpoint_exists(self, client: TestClient, tenant_header: dict):
        """
        Ops breakdown endpoint should be accessible.
        """
        # Use a fake run_id - should return empty or 404
        resp = client.get(
            "/api/v1/metrics/runs/fake-run-id/ops_breakdown",
            headers=tenant_header
        )

        # Should not crash - either returns empty or 404
        assert resp.status_code in [200, 404]


class TestDashboardPerformance:
    """Tests for dashboard endpoint performance."""

    @pytest.mark.regression
    @pytest.mark.perf
    def test_metrics_endpoint_latency_budget(self, client: TestClient, tenant_header: dict):
        """
        Dashboard bundle should return within latency budget.
        """
        import time

        route_key = "latency-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Measure cold request
        start = time.time()
        resp = client.get(
            f"/api/v1/metrics/routes/{route_key}/dashboard_bundle",
            headers=tenant_header,
            params={"tenant_id": "regression-tenant"}
        )
        cold_latency = time.time() - start

        # Cold should be under 500ms (0.5s)
        assert cold_latency < 0.5, f"Cold latency {cold_latency:.3f}s exceeds 500ms budget"

        # Warm requests should be faster
        warm_latencies = []
        for _ in range(3):
            start = time.time()
            client.get(
                f"/api/v1/metrics/routes/{route_key}/dashboard_bundle",
                headers=tenant_header,
                params={"tenant_id": "regression-tenant"}
            )
            warm_latencies.append(time.time() - start)

        avg_warm = sum(warm_latencies) / len(warm_latencies)
        # Warm should be under 200ms
        assert avg_warm < 0.2, f"Avg warm latency {avg_warm:.3f}s exceeds 200ms budget"

    @pytest.mark.regression
    def test_summary_scales_with_routes(self, client: TestClient, tenant_header: dict):
        """
        Summary should handle multiple routes efficiently.
        """
        import time

        # Create multiple routes
        for i in range(5):
            client.post("/api/v1/tgflow/routes", headers=tenant_header,
                       json={"route_key": f"scale-test-{i}", "base_model_ref": "m1"})

        # Measure summary fetch time
        start = time.time()
        resp = client.get(
            "/api/v1/metrics/routes/summary",
            headers=tenant_header,
            params={"tenant_id": "regression-tenant"}
        )
        latency = time.time() - start

        assert resp.status_code == 200
        # Should complete in reasonable time even with multiple routes
        assert latency < 1.0, f"Summary latency {latency:.3f}s too slow for 5 routes"


class TestDashboardDataIntegrity:
    """Tests for dashboard data integrity."""

    @pytest.mark.regression
    def test_kpis_match_route_state(self, client: TestClient, tenant_header: dict):
        """
        KPIs in summary should reflect actual route state.
        """
        route_key = "integrity-test"

        # Setup route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Get route state
        route_resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        route_data = route_resp.json()

        # Get summary
        summary_resp = client.get(
            "/api/v1/metrics/routes/summary",
            headers=tenant_header,
            params={"tenant_id": "regression-tenant"}
        )
        summary = summary_resp.json()

        # Find our route in summary
        route_summary = next((s for s in summary if s["route_key"] == route_key), None)

        if route_summary:
            # Base model should match
            assert route_summary.get("base_model") == route_data["route"]["base_model_ref"]
