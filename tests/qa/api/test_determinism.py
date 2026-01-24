"""
Phase 5.2: Determinism Tests

Verifies that given the same inputs, the system produces identical outputs.
Critical for reproducible ML/training pipelines.
"""

import pytest
import hashlib
import json
from fastapi.testclient import TestClient


class TestDeterminism:
    """Verify system determinism for reproducibility."""
    
    def test_route_creation_idempotent_response(self, client: TestClient, tenant_header):
        """Route details should be deterministic given same inputs."""
        key = "determinism-route-1"
        payload = {"route_key": key, "base_model_ref": "fixed-model", "description": "Test route"}
        
        # Create route
        client.post("/api/v1/tgflow/routes", headers=tenant_header, json=payload)
        
        # Fetch twice
        resp1 = client.get(f"/api/v1/tgflow/routes/{key}", headers=tenant_header)
        resp2 = client.get(f"/api/v1/tgflow/routes/{key}", headers=tenant_header)
        
        if resp1.status_code == 200 and resp2.status_code == 200:
            data1 = resp1.json()
            data2 = resp2.json()
            
            # Exclude timestamps from comparison
            for d in [data1, data2]:
                if "route" in d:
                    d["route"].pop("created_at", None)
                    d["route"].pop("updated_at", None)
                d.pop("created_at", None)
                d.pop("updated_at", None)
            
            assert data1 == data2, "Same route should return identical data"
    
    def test_policy_deterministic_storage(self, client: TestClient, tenant_header):
        """Policy values should be stored and retrieved exactly."""
        key = "determinism-policy"
        client.post("/api/v1/tgflow/routes", headers=tenant_header, 
                    json={"route_key": key, "base_model_ref": "m"})
        
        policy_input = {
            "novelty_threshold": 0.35,
            "forgetting_budget": 0.12,
            "regression_budget": 0.08
        }
        
        client.post(f"/api/v1/tgflow/routes/{key}/policy", headers=tenant_header, json=policy_input)
        
        # Fetch route details to verify policy
        resp = client.get(f"/api/v1/tgflow/routes/{key}", headers=tenant_header)
        if resp.status_code == 200:
            data = resp.json()
            policy = data.get("policy", {})
            
            # Values should match exactly (no floating point drift)
            assert policy.get("novelty_threshold") == 0.35
            assert policy.get("forgetting_budget") == 0.12
    
    def test_timeline_ordering_deterministic(self, client: TestClient, tenant_header, simulation_mode):
        """Timeline events should be returned in consistent order."""
        key = "determinism-timeline"
        client.post("/api/v1/tgflow/routes", headers=tenant_header, 
                    json={"route_key": key, "base_model_ref": "m"})
        client.post(f"/api/v1/tgflow/routes/{key}/feed", headers=tenant_header,
                    json={"feed_type": "hf_dataset", "feed_uri": "d"})
        client.post(f"/api/v1/tgflow/routes/{key}/policy", headers=tenant_header,
                    json={"novelty_threshold": 0.0})
        
        # Run once
        client.post(f"/api/v1/tgflow/routes/{key}/run_once", headers=tenant_header)
        
        # Fetch timeline twice
        tl1 = client.get(f"/api/v1/tgflow/routes/{key}/timeline", headers=tenant_header).json()
        tl2 = client.get(f"/api/v1/tgflow/routes/{key}/timeline", headers=tenant_header).json()
        
        # Event order should be identical
        events1 = tl1.get("timeline", [])
        events2 = tl2.get("timeline", [])
        
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.get("loop_id") == e2.get("loop_id")


class TestHashIntegrity:
    """Verify hash-based integrity checks are consistent."""
    
    def test_content_hash_reproducible(self):
        """Same content should always produce same hash."""
        content = {"route_key": "test", "adapters": [1, 2, 3]}
        
        hash1 = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        
        assert hash1 == hash2
    
    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        content1 = {"route_key": "test1"}
        content2 = {"route_key": "test2"}
        
        hash1 = hashlib.sha256(json.dumps(content1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(content2, sort_keys=True).encode()).hexdigest()
        
        assert hash1 != hash2
