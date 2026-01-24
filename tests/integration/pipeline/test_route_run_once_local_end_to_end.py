import unittest
import asyncio
import os
import shutil
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

# Set simulation mode for test
os.environ["TG_SIMULATION"] = "true"
os.environ["TG_DEMO_MODE"] = "true"

from tensorguard.platform.models.continuous_models import Route, Feed, Policy, EventType, AdapterStage
from tensorguard.platform.services.continuous_registry import ContinuousRegistryService
from tensorguard.tgflow.continuous.orchestrator import ContinuousOrchestrator
from tensorguard.platform.database import SessionLocal, create_production_engine
from sqlmodel import SQLModel, create_engine

class TestLocalPipelineIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Setup in-memory DB
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        
        self.session = SessionLocal()
        self.registry = ContinuousRegistryService(self.session)
        self.orchestrator = ContinuousOrchestrator(self.registry)
        
        self.tenant_id = "test_tenant"
        self.route_key = f"route_{uuid.uuid4().hex[:8]}"
        
        # Setup test data
        os.makedirs("data/raw", exist_ok=True)
        self.test_data_path = "data/raw/test_pipeline.jsonl"
        with open(self.test_data_path, "w") as f:
            f.write('{"text": "integration test data"}\n')
            
        # Register Route, Feed, Policy
        route = Route(
            tenant_id=self.tenant_id,
            route_key=self.route_key,
            name="Test Integration Route",
            base_model_ref="phi-2",
            is_active=True
        )
        feed = Feed(
            tenant_id=self.tenant_id,
            route_key=self.route_key,
            feed_type="local",
            feed_uri=self.test_data_path
        )
        policy = Policy(
            tenant_id=self.tenant_id,
            route_key=self.route_key,
            novelty_threshold=0.0,
            promotion_threshold=0.0,
            auto_promote_to_stable=True
        )
        
        self.session.add(route)
        self.session.add(feed)
        self.session.add(policy)
        self.session.commit()

    async def asyncTearDown(self):
        self.session.close()
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
        if os.path.exists("outputs"):
            shutil.rmtree("outputs", ignore_errors=True)

    async def test_full_pipeline_with_integration_framework(self):
        """
        Verify that ContinuousOrchestrator correctly uses IntegrationManager 
        and connectors during run_once.
        """
        result = await self.orchestrator.run_once(self.tenant_id, self.route_key)
        
        # 1. Check overall success
        self.assertEqual(result["verdict"], "success")
        self.assertEqual(result["promoted_to"], "stable")
        adapter_id = result["adapter_produced"]
        self.assertIsNotNone(adapter_id)
        
        # 2. Verify Events (Evidence Chain)
        events = self.registry.list_timeline(self.tenant_id, self.route_key)
        
        # Check for CONFIG_UPDATED event with integration snapshot
        config_events = [e for e in events if e.event_type == EventType.CONFIG_UPDATED]
        self.assertTrue(len(config_events) > 0)
        snapshot = config_events[0].event_payload_json.get("env_fingerprint")
        self.assertIsNotNone(snapshot)
        self.assertIn("local_feed", snapshot["integrations"])
        self.assertIn("local_trainer", snapshot["integrations"])
        
        # Check for PROMOTED event with serving pack URI
        promoted_events = [e for e in events if e.event_type == EventType.PROMOTED]
        self.assertTrue(len(promoted_events) > 0)
        serving_pack_uri = promoted_events[0].event_payload_json.get("serving_pack_uri")
        self.assertIsNotNone(serving_pack_uri)
        self.assertTrue(serving_pack_uri.endswith("_serving_pack.json"))
        
        # 3. Verify Serving Pack content
        self.assertTrue(os.path.exists(serving_pack_uri))
        with open(serving_pack_uri, "r") as f:
            pack_content = f.read()
            self.assertIn("vllm_config", pack_content)
            self.assertIn(adapter_id, pack_content)

if __name__ == "__main__":
    unittest.main()
