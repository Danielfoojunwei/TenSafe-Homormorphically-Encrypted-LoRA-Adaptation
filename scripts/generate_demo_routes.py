#!/usr/bin/env python3
"""
Phase 6.2: Demo Route Generator

Creates realistic demo routes for UI testing and demonstrations.
Run with: python scripts/generate_demo_routes.py
"""

import sys
import os
from datetime import datetime, timedelta
import random
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlmodel import Session, SQLModel, create_engine
from tensorguard.platform.models.continuous_models import Route, Feed, Policy, AdapterLane
from tensorguard.platform.models.core import Tenant, User

# Demo configuration
DEMO_TENANT_ID = "fceac734-e672-4a0c-863b-c7bb8e28b88e"
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test_ui.db")

DEMO_ROUTES = [
    {
        "route_key": "robotics-pick-place",
        "base_model_ref": "openvla/openvla-7b",
        "description": "Continuous learning for warehouse picking robots",
        "feed": {"feed_type": "hf_dataset", "feed_uri": "tg-internal/warehouse-v12", "privacy_mode": "n2he"},
        "policy": {"novelty_threshold": 0.15, "forgetting_budget": 0.05, "auto_promote_to_stable": True},
    },
    {
        "route_key": "autonomous-navigation",
        "base_model_ref": "physical-intelligence/pi-0",
        "description": "Edge navigation and obstacle avoidance",
        "feed": {"feed_type": "hf_dataset", "feed_uri": "tg-internal/city-scout-v2", "privacy_mode": "OFF"},
        "policy": {"novelty_threshold": 0.3, "promotion_threshold": 0.9, "auto_promote_to_canary": True},
    },
    {
        "route_key": "customer-support-qa",
        "base_model_ref": "microsoft/phi-2",
        "description": "Support ticket response generation",
        "feed": {"feed_type": "s3", "feed_uri": "s3://tg-data/support-tickets/", "privacy_mode": "n2he"},
        "policy": {"novelty_threshold": 0.25, "forgetting_budget": 0.1, "update_cadence": "daily"},
    },
    {
        "route_key": "fraud-detection",
        "base_model_ref": "meta-llama/Llama-2-7b",
        "description": "Real-time fraud pattern detection with encrypted inference",
        "feed": {"feed_type": "azure_blob", "feed_uri": "https://tgdata.blob.core.windows.net/fraud/", "privacy_mode": "n2he"},
        "policy": {"novelty_threshold": 0.1, "regression_budget": 0.02, "auto_promote_to_stable": True},
    },
    {
        "route_key": "code-assistant",
        "base_model_ref": "bigcode/starcoder2-7b",
        "description": "IDE code completion and refactoring suggestions",
        "feed": {"feed_type": "gcs", "feed_uri": "gs://tg-code-data/completions/", "privacy_mode": "OFF"},
        "policy": {"novelty_threshold": 0.4, "forgetting_budget": 0.15, "update_cadence": "weekly"},
    },
]


def generate_demo_routes():
    """Generate demo routes with realistic data."""
    engine = create_engine(DATABASE_URL)
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        # Cleanup existing demo data for idempotency
        from sqlalchemy import delete
        session.execute(delete(Feed))
        session.execute(delete(Policy))
        session.execute(delete(Route))
        session.commit()
        
        # Ensure demo tenant exists
        tenant = Tenant(id=DEMO_TENANT_ID, name="Demo Enterprise", plan="Enterprise")
        session.merge(tenant)
        
        # Ensure demo user exists
        user = User(
            id="demo-user-001",
            email="demo@tensorguard.local",
            hashed_password="N/A",
            role="org_admin",
            tenant_id=DEMO_TENANT_ID
        )
        session.merge(user)
        session.commit()
        
        for route_config in DEMO_ROUTES:
            print(f"Creating route: {route_config['route_key']}")
            
            # Create route
            route = Route(
                tenant_id=DEMO_TENANT_ID,
                route_key=route_config["route_key"],
                base_model_ref=route_config["base_model_ref"],
                description=route_config["description"],
                enabled=True,
                adapter_count=random.randint(2, 8),
                fast_lane_count=random.randint(1, 3),
                slow_lane_count=random.randint(1, 3),
                active_adapter_id=f"adapter-{uuid.uuid4().hex[:8]}",
                last_loop_at=datetime.now() - timedelta(hours=random.randint(1, 48)),
            )
            session.merge(route)
            
            # Create feed
            feed_data = route_config["feed"]
            feed = Feed(
                tenant_id=DEMO_TENANT_ID,
                route_key=route_config["route_key"],
                feed_type=feed_data["feed_type"],
                feed_uri=feed_data["feed_uri"],
                privacy_mode=feed_data["privacy_mode"],
            )
            session.merge(feed)
            
            # Create policy
            policy_data = route_config["policy"]
            policy = Policy(
                tenant_id=DEMO_TENANT_ID,
                route_key=route_config["route_key"],
                novelty_threshold=policy_data.get("novelty_threshold", 0.3),
                forgetting_budget=policy_data.get("forgetting_budget", 0.1),
                regression_budget=policy_data.get("regression_budget", 0.05),
                auto_promote_to_canary=policy_data.get("auto_promote_to_canary", False),
                auto_promote_to_stable=policy_data.get("auto_promote_to_stable", False),
            )
            session.merge(policy)
        
        session.commit()
    
    print(f"\n[OK] Generated {len(DEMO_ROUTES)} demo routes in {DATABASE_URL}")
    print("Routes created:")
    for r in DEMO_ROUTES:
        privacy = r["feed"]["privacy_mode"]
        print(f"  - {r['route_key']} ({r['base_model_ref']}) [{privacy}]")


if __name__ == "__main__":
    generate_demo_routes()
