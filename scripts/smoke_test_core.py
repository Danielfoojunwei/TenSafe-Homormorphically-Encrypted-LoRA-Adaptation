
import asyncio
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

# Force simulation mode BEFORE imports
os.environ["TG_SIMULATION"] = "true"

from sqlmodel import SQLModel, create_engine, Session
from tensorguard.platform.models.continuous_models import FeedType, PrivacyMode
from tensorguard.platform.services.continuous_registry import ContinuousRegistryService
from tensorguard.tgflow.continuous.orchestrator import ContinuousOrchestrator

async def main():
    print("=== TensorGuardFlow Core Smoke Test ===")
    
    # Setup DB
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    
    registry = ContinuousRegistryService(session)
    orchestrator = ContinuousOrchestrator(registry)
    
    tenant_id = "smoke-test-tenant"
    route_key = "smoke-route"
    
    print(f"1. Creating Route '{route_key}'...")
    registry.create_route(tenant_id, route_key, "microsoft/phi-2")
    
    print("2. Connecting Feed...")
    registry.connect_feed(tenant_id, route_key, {
        "feed_type": "local",
        "feed_uri": "dummy_feed",
        "privacy_mode": "off"
    })
    
    print("3. Setting Policy...")
    registry.set_policy(tenant_id, route_key, {
        "novelty_threshold": 0.0, # Force update
        "auto_promote_to_canary": True
    })
    
    print("4. Running Continuous Loop...")
    result = await orchestrator.run_once(tenant_id, route_key)
    
    print(f"   Verdict: {result['verdict']}")
    print(f"   Loop ID: {result['loop_id']}")
    
    if result['verdict'] != 'success':
        print(f"FAILURE: {result.get('reason') or result.get('error')}")
        sys.exit(1)
        
    print("5. Verifying Timeline...")
    timeline = registry.list_timeline(tenant_id, route_key)
    for event in timeline:
        print(f"   [{event.event_type}] {event.created_at.strftime('%H:%M:%S')}")
        
    print("\nSUCCESS: Core loop functional in simulation mode.")

if __name__ == "__main__":
    asyncio.run(main())
