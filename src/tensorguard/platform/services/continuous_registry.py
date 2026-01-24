from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid

from sqlmodel import Session, select, desc
from tensorguard.platform.database import get_session
from tensorguard.platform.models.continuous_models import (
    Route, Feed, Policy, CandidateEvent, AdapterLifecycleState,
    EventType, AdapterLane, AdapterStage
)
from tensorguard.platform.models.tgflow_core_models import (
    AdapterArtifact, AdapterStatus, AdapterRelease, ReleaseChannel
)

class ContinuousRegistryService:
    def __init__(self, session: Session):
        self.session = session
        
    def create_route(self, tenant_id: str, route_key: str, base_model_ref: str, description: str = None) -> Route:
        route = Route(
            tenant_id=tenant_id,
            route_key=route_key,
            base_model_ref=base_model_ref,
            description=description
        )
        self.session.add(route)
        self.session.commit()
        self.session.refresh(route)
        return route

    def get_route(self, tenant_id: str, route_key: str) -> Optional[Route]:
        return self.session.exec(
            select(Route).where(Route.tenant_id == tenant_id, Route.route_key == route_key)
        ).first()

    def list_routes(self, tenant_id: str) -> List[Route]:
        return self.session.exec(select(Route).where(Route.tenant_id == tenant_id)).all()

    def connect_feed(self, tenant_id: str, route_key: str, feed_data: Dict[str, Any]) -> Feed:
        # Check if feed exists, update if so
        existing = self.session.exec(
            select(Feed).where(Feed.tenant_id == tenant_id, Feed.route_key == route_key)
        ).first()
        
        if existing:
            for k, v in feed_data.items():
                setattr(existing, k, v)
            existing.updated_at = datetime.utcnow()
            feed = existing
        else:
            feed = Feed(tenant_id=tenant_id, route_key=route_key, **feed_data)
            self.session.add(feed)
            
        self.session.commit()
        self.session.refresh(feed)
        return feed
        
    def get_feed(self, tenant_id: str, route_key: str) -> Optional[Feed]:
        return self.session.exec(
             select(Feed).where(Feed.tenant_id == tenant_id, Feed.route_key == route_key)
        ).first()

    def set_policy(self, tenant_id: str, route_key: str, policy_data: Dict[str, Any]) -> Policy:
        existing = self.session.exec(
            select(Policy).where(Policy.tenant_id == tenant_id, Policy.route_key == route_key)
        ).first()
        
        if existing:
            for k, v in policy_data.items():
                setattr(existing, k, v)
            existing.updated_at = datetime.utcnow()
            policy = existing
        else:
            policy = Policy(tenant_id=tenant_id, route_key=route_key, **policy_data)
            self.session.add(policy)
            
        self.session.commit()
        self.session.refresh(policy)
        return policy

    def get_policy(self, tenant_id: str, route_key: str) -> Optional[Policy]:
        return self.session.exec(
             select(Policy).where(Policy.tenant_id == tenant_id, Policy.route_key == route_key)
        ).first()

    def record_event(self, tenant_id: str, route_key: str, event_type: EventType, 
                     payload: Dict[str, Any], loop_id: str = None, adapter_id: str = None) -> CandidateEvent:
        event = CandidateEvent(
            tenant_id=tenant_id,
            route_key=route_key,
            event_type=event_type,
            event_payload_json=payload,
            loop_id=loop_id,
            adapter_id=adapter_id
        )
        self.session.add(event)
        self.session.commit()
        return event

    def list_timeline(self, tenant_id: str, route_key: str, limit: int = 50) -> List[CandidateEvent]:
        return self.session.exec(
            select(CandidateEvent)
            .where(CandidateEvent.tenant_id == tenant_id, CandidateEvent.route_key == route_key)
            .order_by(desc(CandidateEvent.created_at))
            .limit(limit)
        ).all()

    def register_candidate_adapter(self, tenant_id: str, route_key: str, 
                                   adapter_metadata: Dict[str, Any],
                                   training_metrics: Dict[str, Any]) -> str:
        """
        Registers a new adapter artifact as CANDIDATE.
        """
        # 1. Create AdapterArtifact (Core)
        artifact = AdapterArtifact(
            tenant_id=tenant_id,
            name=f"{route_key}-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
            base_model_id=adapter_metadata.get("base_model_ref", "unknown"),
            status=AdapterStatus.DRAFT, # Will be promoted to registered via lifecycle
            adapter_path=adapter_metadata.get("adapter_path", ""),
            tgsp_path=adapter_metadata.get("tgsp_path"),
            adapter_hash=adapter_metadata.get("adapter_hash", "sha256:unknown"),
            config_json=adapter_metadata.get("config", {}),
            privacy_mode=adapter_metadata.get("privacy_mode", "off")
        )
        self.session.add(artifact)
        self.session.flush() # Get ID
        
        # 2. Create AdapterLifecycleState (Continuous)
        lifecycle = AdapterLifecycleState(
            tenant_id=tenant_id,
            adapter_id=artifact.id,
            route_key=route_key,
            lane=AdapterLane.FAST,
            stage=AdapterStage.CANDIDATE,
            primary_metric=training_metrics.get("primary_metric"),
            forgetting_score=training_metrics.get("forgetting_score"),
            regression_score=training_metrics.get("regression_score"),
            novelty_score=training_metrics.get("novelty_score")
        )
        self.session.add(lifecycle)
        self.session.commit()
        
        return artifact.id

    def promote_adapter(self, tenant_id: str, route_key: str, adapter_id: str, target_stage: AdapterStage):
        """
        Promotes an adapter to a new stage (CANARY or STABLE).
        Reflects changes in Route and AdapterRelease.
        """
        route = self.get_route(tenant_id, route_key)
        if not route:
            raise ValueError("Route not found")
            
        # Update lifecycle
        lifecycle = self.session.exec(
            select(AdapterLifecycleState)
            .where(AdapterLifecycleState.adapter_id == adapter_id)
        ).first()
        
        if not lifecycle:
            raise ValueError("Adapter lifecycle state not found")
            
        lifecycle.stage = target_stage
        
        # Update Route pointers & Release history
        if target_stage == AdapterStage.STABLE:
            # Shift current stable to fallback if exists
            if route.active_adapter_id:
                route.fallback_adapter_id = route.active_adapter_id
                
            route.active_adapter_id = adapter_id
            lifecycle.promoted_to_stable_at = datetime.utcnow()
            
            # Record release
            release = AdapterRelease(
                tenant_id=tenant_id,
                adapter_id=adapter_id,
                route_key=route_key,
                channel=ReleaseChannel.STABLE,
                previous_adapter_id=route.fallback_adapter_id,
                decision_reason="continuous_promotion"
            )
            self.session.add(release)
            
        elif target_stage == AdapterStage.CANARY:
            route.canary_adapter_id = adapter_id
            lifecycle.promoted_to_canary_at = datetime.utcnow()
            
            release = AdapterRelease(
                tenant_id=tenant_id,
                adapter_id=adapter_id,
                route_key=route_key,
                channel=ReleaseChannel.CANARY,
                decision_reason="continuous_promotion"
            )
            self.session.add(release)
            
        self.session.add(route)
        self.session.add(lifecycle)
        self.session.commit()

    def rollback_route(self, tenant_id: str, route_key: str):
        """
        Rollback to fallback adapter.
        """
        route = self.get_route(tenant_id, route_key)
        if not route or not route.fallback_adapter_id:
            raise ValueError("No fallback adapter available")
            
        current = route.active_adapter_id
        fallback = route.fallback_adapter_id
        
        # Swap
        route.active_adapter_id = fallback
        # route.fallback_adapter_id = current # Optional: keep failed as new fallback? No, usually keep safe one.
        
        # Record release
        release = AdapterRelease(
            tenant_id=tenant_id,
            adapter_id=fallback,
            route_key=route_key,
            channel=ReleaseChannel.ROLLBACK,
            previous_adapter_id=current,
            decision_reason="rollback"
        )
        self.session.add(release)
        self.session.add(route)
        self.session.commit()
        
        return fallback

    def get_adapter_details(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        # Helper to join Artifact + Lifecycle
        artifact = self.session.get(AdapterArtifact, adapter_id)
        if not artifact:
            return None
        lifecycle = self.session.exec(select(AdapterLifecycleState).where(AdapterLifecycleState.adapter_id == adapter_id)).first()
        
        return {
            "artifact": artifact,
            "lifecycle": lifecycle
        }

    def get_route_adapters(self, tenant_id: str, route_key: str) -> List[AdapterLifecycleState]:
        """Lists all adapters associated with a specific route."""
        return self.session.exec(
            select(AdapterLifecycleState)
            .where(AdapterLifecycleState.tenant_id == tenant_id, AdapterLifecycleState.route_key == route_key)
        ).all()
