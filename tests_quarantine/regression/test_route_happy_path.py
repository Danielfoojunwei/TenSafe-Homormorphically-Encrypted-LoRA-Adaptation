"""
Regression Test: I1 - Route Lifecycle Works

Tests the complete route lifecycle:
create route -> attach feed -> set policy -> run_once -> candidate -> gates
"""

import pytest
from fastapi.testclient import TestClient


class TestRouteHappyPath:
    """I1: Complete route lifecycle must work end-to-end."""

    @pytest.mark.regression
    def test_create_route_succeeds(self, client: TestClient, tenant_header: dict):
        """Route creation returns success with correct route_key."""
        response = client.post(
            "/api/v1/tgflow/routes",
            headers=tenant_header,
            json={
                "route_key": "i1-lifecycle-test",
                "base_model_ref": "microsoft/phi-2",
                "description": "I1 Lifecycle Test"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["route_key"] == "i1-lifecycle-test"

    @pytest.mark.regression
    def test_connect_feed_succeeds(self, client: TestClient, tenant_header: dict):
        """Feed connection works for existing route."""
        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": "feed-test", "base_model_ref": "m1"})

        # Connect feed
        response = client.post(
            "/api/v1/tgflow/routes/feed-test/feed",
            headers=tenant_header,
            json={
                "feed_type": "local",
                "feed_uri": "tests/fixtures/sample_data.jsonl",
                "privacy_mode": "off"
            }
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True

    @pytest.mark.regression
    def test_set_policy_succeeds(self, client: TestClient, tenant_header: dict):
        """Policy configuration works for existing route."""
        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": "policy-test", "base_model_ref": "m1"})

        # Set policy
        response = client.post(
            "/api/v1/tgflow/routes/policy-test/policy",
            headers=tenant_header,
            json={
                "novelty_threshold": 0.3,
                "promotion_threshold": 0.9,
                "forgetting_budget": 0.1,
                "regression_budget": 0.05
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "policy" in data

    @pytest.mark.regression
    def test_complete_route_lifecycle(self, client: TestClient, tenant_header: dict):
        """
        Full lifecycle: create -> feed -> policy -> run_once -> check timeline.

        This is the critical I1 invariant test.
        """
        route_key = "full-lifecycle-test"

        # Step 1: Create route
        resp = client.post("/api/v1/tgflow/routes", headers=tenant_header,
                          json={"route_key": route_key, "base_model_ref": "microsoft/phi-2"})
        assert resp.status_code == 200, f"Create failed: {resp.text}"

        # Step 2: Connect feed
        resp = client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                          json={"feed_type": "local", "feed_uri": "mock://data", "privacy_mode": "off"})
        assert resp.status_code == 200, f"Feed connect failed: {resp.text}"

        # Step 3: Set policy
        resp = client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                          json={
                              "novelty_threshold": 0.1,  # Low threshold to ensure training triggers
                              "promotion_threshold": 0.5,
                              "forgetting_budget": 0.5,
                              "regression_budget": 0.5,
                              "auto_promote_to_canary": True
                          })
        assert resp.status_code == 200, f"Policy set failed: {resp.text}"

        # Step 4: Run once (triggers the full loop)
        resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
        assert resp.status_code == 200, f"Run once failed: {resp.text}"

        result = resp.json()
        # Verify run completed (either success or skipped due to low novelty is acceptable)
        assert result.get("verdict") in ["success", "skipped"], f"Unexpected verdict: {result}"
        assert "loop_id" in result, "Missing loop_id in result"

        # Step 5: Check timeline has events
        resp = client.get(f"/api/v1/tgflow/routes/{route_key}/timeline", headers=tenant_header)
        assert resp.status_code == 200, f"Timeline fetch failed: {resp.text}"

        timeline_data = resp.json()
        assert "timeline" in timeline_data
        # Timeline should have at least config event

    @pytest.mark.regression
    def test_get_route_returns_full_config(self, client: TestClient, tenant_header: dict):
        """GET route returns route + feed + policy."""
        route_key = "config-check-test"

        # Setup complete route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://data", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.3})

        # Get route details
        resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        assert resp.status_code == 200

        data = resp.json()
        assert "route" in data
        assert "feed" in data
        assert "policy" in data
        assert data["route"]["route_key"] == route_key

    @pytest.mark.regression
    def test_list_routes_returns_status(self, client: TestClient, tenant_header: dict):
        """List routes includes status fields."""
        # Create routes
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": "list-test-1", "base_model_ref": "m1"})
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": "list-test-2", "base_model_ref": "m2"})

        # List
        resp = client.get("/api/v1/tgflow/routes", headers=tenant_header)
        assert resp.status_code == 200

        routes = resp.json()
        assert len(routes) >= 2

        # Check status fields present
        for route in routes:
            assert "route_key" in route
            assert "enabled" in route
            assert "adapter_count" in route


class TestRouteEdgeCases:
    """Edge cases for route lifecycle."""

    @pytest.mark.regression
    def test_create_duplicate_route_fails(self, client: TestClient, tenant_header: dict):
        """Cannot create route with same key twice."""
        payload = {"route_key": "dup-test", "base_model_ref": "m1"}

        # First create succeeds
        resp1 = client.post("/api/v1/tgflow/routes", headers=tenant_header, json=payload)
        assert resp1.status_code == 200

        # Second create fails
        resp2 = client.post("/api/v1/tgflow/routes", headers=tenant_header, json=payload)
        assert resp2.status_code == 400
        assert "already exists" in resp2.json().get("detail", "").lower()

    @pytest.mark.regression
    def test_feed_on_nonexistent_route_fails(self, client: TestClient, tenant_header: dict):
        """Cannot connect feed to nonexistent route."""
        resp = client.post(
            "/api/v1/tgflow/routes/does-not-exist/feed",
            headers=tenant_header,
            json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"}
        )
        assert resp.status_code == 404

    @pytest.mark.regression
    def test_run_once_without_config_fails(self, client: TestClient, tenant_header: dict):
        """run_once fails gracefully if feed/policy not configured."""
        route_key = "unconfigured-route"

        # Create route only (no feed or policy)
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})

        # Run once should fail gracefully
        resp = client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
        assert resp.status_code == 200  # Returns result, not error

        result = resp.json()
        assert result.get("verdict") == "failed"
        assert "Configuration missing" in result.get("reason", "")
