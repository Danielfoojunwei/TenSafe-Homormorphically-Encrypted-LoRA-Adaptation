import pytest
import datetime
from sqlmodel import Session, select, create_engine, SQLModel
from tensorguard.platform.models.continuous_models import (
    Route, Feed, Policy, FeedType, PrivacyMode, CandidateEvent, EventType, AdapterStage
)
from tensorguard.platform.services.continuous_registry import ContinuousRegistryService
from tensorguard.tgflow.continuous.orchestrator import ContinuousOrchestrator

# Use in-memory SQLite for testing
@pytest.fixture(name="session")
def session_fixture():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.mark.asyncio
async def test_continuous_loop_end_to_end(session: Session):
    """
    Verify the complete continuous learning loop:
    Create Route -> Connect Feed -> Set Policy -> Run Loop -> Verify Events -> Verify Artifacts
    """
    tenant_id = "test-tenant-001"
    route_key = "test-route-qa"
    
    registry = ContinuousRegistryService(session)
    orchestrator = ContinuousOrchestrator(registry)
    
    # 1. Create Route
    route = registry.create_route(
        tenant_id=tenant_id,
        route_key=route_key,
        base_model_ref="microsoft/phi-2",
        description="QA integration test route"
    )
    assert route.id is not None
    
    # 2. Connect Feed
    registry.connect_feed(tenant_id, route_key, {
        "feed_type": FeedType.LOCAL,
        "feed_uri": "./data/qa_pairs.json",
        "privacy_mode": PrivacyMode.OFF
    })
    
    # 3. Set Policy
    registry.set_policy(tenant_id, route_key, {
        "novelty_threshold": 0.1,  # Low threshold to ensure update triggers
        "forgetting_budget": 0.2
    })
    
    # 4. Run Loop
    # Mocking ingestion in orchestrator is handled by internal _ingest_feed stub 
    # which returns random hash suitable for 'simulated' change.
    
    result = await orchestrator.run_once(tenant_id, route_key)
    
    # 5. Verify Result
    assert result["verdict"] == "success"
    assert "adapter_id" in result["adapter_produced"]
    
    # 6. Verify Timeline Events
    timeline = registry.list_timeline(tenant_id, route_key)
    event_types = [e.event_type for e in timeline]
    
    assert EventType.FEED_INGESTED in event_types
    assert EventType.UPDATE_PROPOSED in event_types
    assert EventType.TRAIN_DONE in event_types
    assert EventType.EVAL_DONE in event_types
    assert EventType.PACKAGED in event_types
    assert EventType.REGISTERED in event_types
    
    # 7. Check Registered Artifact
    adapter_id = result["adapter_produced"]
    details = registry.get_adapter_details(adapter_id)
    assert details is not None
    assert details["lifecycle"].stage == AdapterStage.CANDIDATE
    assert details["artifact"].base_model_id == "microsoft/phi-2"

@pytest.mark.asyncio
async def test_privacy_mode_flow(session: Session):
    tenant_id = "test-tenant-002"
    route_key = "test-route-secure"
    registry = ContinuousRegistryService(session)
    orchestrator = ContinuousOrchestrator(registry)
    
    registry.create_route(tenant_id, route_key, "llama2")
    registry.connect_feed(tenant_id, route_key, {
        "feed_type": FeedType.S3,
        "feed_uri": "s3://secure-bucket",
        "privacy_mode": PrivacyMode.N2HE
    })
    
    result = await orchestrator.run_once(tenant_id, route_key)
    # Even if it fails due to mocked training, ensure privacy mode was passed
    # In orchestrator stub, it passes privacy mode to config.
    # The result verdict might be 'success' because my Orchestrator stubs purely rely on logic I wrote.
    
    assert result["verdict"] == "success"
    
    timeline = registry.list_timeline(tenant_id, route_key)
    # Check if any event has privacy context?
    # My current implementation doesn't strictly log distinct privacy event types other than generic flow,
    # but the config passed to training had privacy mode.
    # We can check the AdapterArtifact has privacy_mode='n2he'
    
    adapter_id = result["adapter_produced"]
    details = registry.get_adapter_details(adapter_id)
    assert details["artifact"].privacy_mode == "n2he"
