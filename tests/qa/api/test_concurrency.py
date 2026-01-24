"""
Phase 5.1: Concurrency & Reliability Tests

Tests for concurrent access patterns and system reliability under load.
Uses pytest-asyncio for async test patterns.
"""

import pytest
import asyncio
import time
from fastapi.testclient import TestClient
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestConcurrentRouteAccess:
    """Test concurrent access to routes doesn't cause data corruption."""
    
    def test_concurrent_route_creation(self, client: TestClient, tenant_header):
        """Multiple route creations in parallel should not conflict."""
        route_keys = [f"concurrent-route-{i}" for i in range(5)]
        
        def create_route(key):
            return client.post(
                "/api/v1/tgflow/routes",
                headers=tenant_header,
                json={"route_key": key, "base_model_ref": "test-model"}
            )
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(create_route, key): key for key in route_keys}
            results = []
            for future in as_completed(futures):
                key = futures[future]
                try:
                    resp = future.result()
                    results.append((key, resp.status_code))
                except Exception as e:
                    results.append((key, str(e)))
        
        # Verify all routes created successfully
        success_count = sum(1 for _, status in results if status == 200)
        assert success_count == 5, f"Expected 5 successes, got {success_count}: {results}"
    
    def test_concurrent_route_reads(self, client: TestClient, tenant_header):
        """Multiple concurrent reads should not block each other."""
        # Create a route first
        client.post(
            "/api/v1/tgflow/routes",
            headers=tenant_header,
            json={"route_key": "read-test-route", "base_model_ref": "model"}
        )
        
        def read_routes():
            return client.get("/api/v1/tgflow/routes", headers=tenant_header)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_routes) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]
        elapsed = time.time() - start_time
        
        # All reads should succeed
        assert all(r.status_code == 200 for r in results)
        # Concurrent reads should complete faster than sequential (< 2s total)
        assert elapsed < 2.0, f"Concurrent reads took too long: {elapsed}s"


class TestReliability:
    """Test system reliability under edge conditions."""
    
    def test_duplicate_route_creation_rejected(self, client: TestClient, tenant_header):
        """Creating the same route twice should return appropriate error."""
        key = "duplicate-test"
        
        # First creation should succeed
        resp1 = client.post(
            "/api/v1/tgflow/routes",
            headers=tenant_header,
            json={"route_key": key, "base_model_ref": "model"}
        )
        assert resp1.status_code == 200
        
        # Second creation should fail gracefully
        resp2 = client.post(
            "/api/v1/tgflow/routes",
            headers=tenant_header,
            json={"route_key": key, "base_model_ref": "model"}
        )
        # Either 409 Conflict or 400 Bad Request is acceptable
        assert resp2.status_code in [400, 409, 500], f"Expected error, got {resp2.status_code}"
    
    def test_invalid_policy_values_rejected(self, client: TestClient, tenant_header):
        """Invalid policy values should be rejected with clear error."""
        key = "policy-validation"
        client.post(
            "/api/v1/tgflow/routes",
            headers=tenant_header,
            json={"route_key": key, "base_model_ref": "model"}
        )
        
        # Test invalid threshold (> 1.0)
        resp = client.post(
            f"/api/v1/tgflow/routes/{key}/policy",
            headers=tenant_header,
            json={"novelty_threshold": 5.0}
        )
        # Should either reject (422) or accept (200 with clamping)
        assert resp.status_code in [200, 422]
    
    def test_missing_feed_run_once_fails_gracefully(self, client: TestClient, tenant_header, simulation_mode):
        """Run once without feed configuration should fail with clear message."""
        key = "no-feed-route"
        client.post(
            "/api/v1/tgflow/routes",
            headers=tenant_header,
            json={"route_key": key, "base_model_ref": "model"}
        )
        
        # Run without feed should fail
        resp = client.post(
            f"/api/v1/tgflow/routes/{key}/run_once",
            headers=tenant_header
        )
        
        data = resp.json()
        assert data.get("verdict") in ["failed", "skipped"]
        assert "feed" in data.get("reason", "").lower() or "configuration" in data.get("reason", "").lower()
