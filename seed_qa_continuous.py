import sys
import os
from sqlmodel import Session, SQLModel, create_engine
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.platform.models.continuous_models import Route, Feed, Policy, AdapterStage
from tensorguard.platform.models.core import Tenant, User

# Use the same DB as the running server
DATABASE_URL = "sqlite:///./test_ui.db"
engine = create_engine(DATABASE_URL)

def seed():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        # Cleanup old records for idempotency
        from sqlalchemy import delete
        session.execute(delete(Feed))
        session.execute(delete(Policy))
        session.execute(delete(Route))
        session.commit()
        # 1. Create Tenant (use same ID as demo user in auth.py)
        tenant = Tenant(id="fceac734-e672-4a0c-863b-c7bb8e28b88e", name="Demo Enterprise", plan="Enterprise")
        session.merge(tenant) # use merge to be idempotent
        
        # 2. Create User (admin@tensorguard.ai for simplicity)
        user = User(
            id="qa-user",
            email="admin@tensorguard.ai",
            hashed_password="N/A",
            role="org_admin",
            tenant_id=tenant.id
        )
        session.merge(user)
        
        # 3. Create a few Routes
        routes = [
            Route(
                tenant_id=tenant.id, 
                route_key="robotics-pick-place", 
                name="Robotics Pick & Place", 
                base_model_ref="openvla-7b",
                description="Continuous learning for warehouse picking"
            ),
            Route(
                tenant_id=tenant.id, 
                route_key="autonomous-navigation", 
                name="Autonomous Navigation", 
                base_model_ref="pi-0",
                description="Edge navigation drift control"
            )
        ]
        for r in routes:
            session.merge(r)
        
        # 4. Connect Feeds
        feeds = [
            Feed(
                tenant_id=tenant.id,
                route_key="robotics-pick-place",
                feed_type="hf_dataset",
                feed_uri="tg-internal/warehouse-v12",
                privacy_mode="n2he"
            ),
            Feed(
                tenant_id=tenant.id,
                route_key="autonomous-navigation",
                feed_type="hf_dataset",
                feed_uri="tg-internal/city-scout-v2",
                privacy_mode="OFF"
            )
        ]
        for f in feeds:
            session.merge(f)
            
        # 5. Set Policies
        policies = [
            Policy(
                tenant_id=tenant.id,
                route_key="robotics-pick-place",
                novelty_threshold=0.15,
                forgetting_budget=0.05,
                auto_promote_to_stable=True
            ),
            Policy(
                tenant_id=tenant.id,
                route_key="autonomous-navigation",
                novelty_threshold=0.3,
                promotion_threshold=0.9,
                auto_promote_to_canary=True
            )
        ]
        for p in policies:
            session.merge(p)
            
        session.commit()
    print("QA Continuous Learning data seeded to test_ui.db")

if __name__ == "__main__":
    seed()
