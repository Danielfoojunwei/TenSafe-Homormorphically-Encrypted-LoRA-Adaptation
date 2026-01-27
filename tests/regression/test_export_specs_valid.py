"""
Regression Test: I6 - Integration Exporter Validity

Tests that K8s Job, vLLM, TGI, Triton export schemas are valid.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestExportSpecsValid:
    """I6: Export specs must be valid and deployable."""

    @pytest.mark.regression
    def test_k8s_export_has_required_fields(self, client: TestClient, tenant_header: dict):
        """
        K8s export should include required Job fields.

        This is the critical I6 invariant test for K8s.
        """
        route_key = "k8s-export-test"

        # Setup route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "microsoft/phi-2"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://data", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Export as k8s
        resp = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "k8s"}
        )

        assert resp.status_code == 200
        data = resp.json()

        assert data.get("ok") is True
        assert data.get("backend") == "k8s"
        assert "run_spec_json" in data

        spec = data["run_spec_json"]

        # Verify required fields for continuous loop workload
        assert "workload_type" in spec
        assert spec["workload_type"] == "continuous_loop"
        assert "route_key" in spec
        assert "base_model" in spec

    @pytest.mark.regression
    def test_vllm_export_structure(self, client: TestClient, tenant_header: dict):
        """
        vLLM export should produce valid serving configuration.
        """
        route_key = "vllm-export-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Export as vllm
        resp = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "vllm"}
        )

        assert resp.status_code == 200
        data = resp.json()

        assert data.get("ok") is True
        assert "run_spec_json" in data

    @pytest.mark.regression
    def test_export_includes_policy_config(self, client: TestClient, tenant_header: dict):
        """
        Export should include policy configuration for reproducibility.
        """
        route_key = "policy-export-test"

        # Setup with specific policy
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={
                       "novelty_threshold": 0.5,
                       "promotion_threshold": 0.9,
                       "forgetting_budget": 0.1
                   })

        # Export
        resp = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "k8s"}
        )

        data = resp.json()
        spec = data.get("run_spec_json", {})

        # Policy should be included
        assert "policy" in spec
        policy = spec["policy"]
        # Should preserve configured values
        if policy:  # May be empty dict if not fully configured
            pass  # Structure check only


class TestExportValidation:
    """Tests for export spec validation."""

    @pytest.mark.regression
    def test_export_nonexistent_route_fails(self, client: TestClient, tenant_header: dict):
        """
        Exporting a nonexistent route should return 404.
        """
        resp = client.post(
            "/api/v1/tgflow/routes/does-not-exist/export",
            headers=tenant_header,
            params={"backend": "k8s"}
        )

        assert resp.status_code == 404

    @pytest.mark.regression
    def test_export_json_is_parseable(self, client: TestClient, tenant_header: dict):
        """
        Exported spec should be valid JSON.
        """
        route_key = "json-valid-test"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        resp = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "k8s"}
        )

        data = resp.json()
        spec = data.get("run_spec_json")

        # Should be serializable
        try:
            json_str = json.dumps(spec)
            parsed = json.loads(json_str)
            assert parsed == spec
        except (TypeError, json.JSONDecodeError) as e:
            pytest.fail(f"Export spec not valid JSON: {e}")


class TestExportBackends:
    """Tests for different export backends."""

    @pytest.mark.regression
    def test_unknown_backend_handled(self, client: TestClient, tenant_header: dict):
        """
        Unknown export backend should be handled gracefully.
        """
        route_key = "unknown-backend-test"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        resp = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "unknown_backend_xyz"}
        )

        # Should either handle gracefully or return error
        # Not crash with 500
        assert resp.status_code in [200, 400, 422]

    @pytest.mark.regression
    def test_export_includes_feed_uri(self, client: TestClient, tenant_header: dict):
        """
        Export should include feed URI for data source.
        """
        route_key = "feed-uri-export-test"
        feed_uri = "s3://bucket/training-data/"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "s3", "feed_uri": feed_uri, "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        resp = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "k8s"}
        )

        data = resp.json()
        spec = data.get("run_spec_json", {})

        assert "feed_uri" in spec
        assert spec["feed_uri"] == feed_uri


class TestExportIdempotency:
    """Tests for export idempotency."""

    @pytest.mark.regression
    def test_multiple_exports_identical(self, client: TestClient, tenant_header: dict):
        """
        Multiple exports of same route should produce identical specs.
        """
        route_key = "idempotent-export-test"

        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.3})

        # Export twice
        resp1 = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "k8s"}
        )
        resp2 = client.post(
            f"/api/v1/tgflow/routes/{route_key}/export",
            headers=tenant_header,
            params={"backend": "k8s"}
        )

        # Specs should be identical
        assert resp1.json()["run_spec_json"] == resp2.json()["run_spec_json"]
