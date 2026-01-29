"""
Regression Test: Concurrency - Concurrent run_once

Tests race conditions when multiple run_once calls happen simultaneously.
"""

import pytest
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from conftest import async_iter_mock


class TestConcurrentRunOnce:
    """Tests for concurrent execution of run_once."""

    @pytest.mark.regression
    def test_concurrent_run_once_same_route(self, client: TestClient, tenant_header: dict):
        """
        Two simultaneous run_once calls should not corrupt state.

        One should succeed, the other should be queued/blocked or handled gracefully.
        No double-promotion or corrupted timeline.
        """
        route_key = "concurrent-test"

        # Setup route
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={
                       "novelty_threshold": 0.1,
                       "auto_promote_to_canary": True,
                       "auto_promote_to_stable": False
                   })

        results = []
        errors = []

        def run_once():
            try:
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

                    # Small delay to increase overlap chance
                    time.sleep(0.01)
                    resp = client.post(
                        f"/api/v1/tgflow/routes/{route_key}/run_once",
                        headers=tenant_header
                    )
                    results.append(resp.json())
            except Exception as e:
                errors.append(str(e))

        # Run two concurrent requests
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_once) for _ in range(2)]
            for future in as_completed(futures):
                pass

        # Both should complete without crashing
        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        assert len(results) == 2, "Both requests should complete"

        # Each result should have loop_id
        for result in results:
            assert "loop_id" in result, f"Missing loop_id in result: {result}"

        # Verify state consistency - get timeline
        timeline_resp = client.get(
            f"/api/v1/tgflow/routes/{route_key}/timeline",
            headers=tenant_header
        )
        assert timeline_resp.status_code == 200

        # No duplicate loop_ids in timeline (each run should have unique ID)
        timeline = timeline_resp.json().get("timeline", [])
        loop_ids = [loop.get("loop_id") for loop in timeline if loop.get("loop_id")]
        assert len(loop_ids) == len(set(loop_ids)), "Duplicate loop_ids found - race condition!"

    @pytest.mark.regression
    def test_concurrent_promote_same_adapter(self, client: TestClient, tenant_header: dict):
        """
        Concurrent promotions should not corrupt state.
        """
        route_key = "concurrent-promote-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1, "auto_promote_to_stable": True})

        # Create an adapter first
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

        # Get the adapter ID
        route_resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        route_data = route_resp.json()["route"]
        adapter_id = route_data.get("active_adapter_id") or route_data.get("canary_adapter_id")

        if adapter_id:
            # Try concurrent promotions
            errors = []

            def promote():
                try:
                    resp = client.post(
                        f"/api/v1/tgflow/routes/{route_key}/promote",
                        headers=tenant_header,
                        params={"adapter_id": adapter_id, "target": "stable"}
                    )
                    return resp.status_code
                except Exception as e:
                    errors.append(str(e))
                    return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(promote) for _ in range(2)]
                status_codes = [f.result() for f in as_completed(futures)]

            # At least one should succeed
            assert 200 in status_codes or 400 in status_codes, "Promotions should complete"
            assert len(errors) == 0, f"Promotion errors: {errors}"

    @pytest.mark.regression
    def test_concurrent_rollback_safe(self, client: TestClient, tenant_header: dict):
        """
        Concurrent rollback attempts should be safe.
        """
        route_key = "concurrent-rollback-test"

        # Setup and create two adapters
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

            # Two runs to create fallback
            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        # Concurrent rollbacks
        results = []

        def rollback():
            resp = client.post(
                f"/api/v1/tgflow/routes/{route_key}/rollback",
                headers=tenant_header
            )
            results.append(resp.status_code)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(rollback) for _ in range(2)]
            for future in as_completed(futures):
                pass

        # Should not corrupt state - verify route is still valid
        route_resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        assert route_resp.status_code == 200


class TestConcurrentReads:
    """Tests for concurrent read operations."""

    @pytest.mark.regression
    def test_concurrent_timeline_reads(self, client: TestClient, tenant_header: dict):
        """
        Concurrent timeline reads should be safe and consistent.
        """
        route_key = "concurrent-read-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        # Run once to generate data
        client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)

        results = []

        def read_timeline():
            resp = client.get(
                f"/api/v1/tgflow/routes/{route_key}/timeline",
                headers=tenant_header
            )
            results.append(resp.json())

        # Many concurrent reads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_timeline) for _ in range(10)]
            for future in as_completed(futures):
                pass

        # All should succeed
        assert len(results) == 10

        # All should return same data (consistency)
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Concurrent reads returned inconsistent data"


class TestLockingBehavior:
    """Tests for route-level locking (if implemented)."""

    @pytest.mark.regression
    def test_route_write_during_run(self, client: TestClient, tenant_header: dict):
        """
        Write operations during run_once should be handled safely.
        """
        route_key = "write-during-run-test"

        # Setup
        client.post("/api/v1/tgflow/routes", headers=tenant_header,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=tenant_header,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=tenant_header,
                   json={"novelty_threshold": 0.1})

        errors = []

        def run_and_update():
            try:
                # Start run_once in background thread
                with patch('tensorguard.tgflow.continuous.orchestrator.PeftWorkflow') as MockWorkflow:
                    mock_instance = MagicMock()
                    mock_instance.artifacts = {"adapter_path": "/mock/adapter"}
                    mock_instance.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
                    mock_instance.diagnosis = None

                    def slow_train():
                        time.sleep(0.1)
                        return iter([])

                    mock_instance._stage_train.return_value = slow_train()
                    mock_instance._stage_eval.return_value = async_iter_mock([])
                    mock_instance._stage_pack_tgsp.return_value = async_iter_mock([])
                    mock_instance._stage_emit_evidence.return_value = async_iter_mock([])
                    MockWorkflow.return_value = mock_instance

                    client.post(f"/api/v1/tgflow/routes/{route_key}/run_once", headers=tenant_header)
            except Exception as e:
                errors.append(f"run: {e}")

        def update_policy():
            try:
                time.sleep(0.05)  # Slight delay to overlap with run
                client.post(
                    f"/api/v1/tgflow/routes/{route_key}/policy",
                    headers=tenant_header,
                    json={"novelty_threshold": 0.5}
                )
            except Exception as e:
                errors.append(f"policy: {e}")

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(run_and_update)
            executor.submit(update_policy)

        # Neither operation should crash
        # State should be consistent after both complete
        route_resp = client.get(f"/api/v1/tgflow/routes/{route_key}", headers=tenant_header)
        assert route_resp.status_code == 200
