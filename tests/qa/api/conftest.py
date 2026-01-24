import pytest
import os
import sys
from fastapi.testclient import TestClient
from fastapi import Header # Added Header
from sqlmodel import SQLModel, Session, create_engine
from typing import Generator
from sqlalchemy.pool import StaticPool
import os

os.environ["TG_SIMULATION"] = "true" # Force simulation mode for tests # Import StaticPool

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))


# CRITICAL: Patch DB Engine BEFORE importing app
# This prevents import-time queries (which happen in some legacy modules) 
# from hitting the default file-based DB.
from tensorguard.platform import database

# TEST_DB_FILE = "test_qa.db"

test_engine = create_engine(
    "sqlite:///:memory:", 
    connect_args={"check_same_thread": False}, 
    poolclass=StaticPool # Keep data in memory across connections
)
database.engine = test_engine
# Import all models to ensure they are registered with SQLModel.metadata
from tensorguard.platform.models.continuous_models import *
from tensorguard.platform.models.evidence_models import * 
from tensorguard.platform.models.peft_models import *

# Debug logging
with open("debug_output.txt", "w") as f:
    f.write(f"DEBUG: Registered Tables: {list(SQLModel.metadata.tables.keys())}\n")
    f.write(f"DEBUG: Test Engine: {test_engine}\n")
    f.write(f"DEBUG: Database Engine (before patch): {database.engine}\n")

database.engine = test_engine
SQLModel.metadata.create_all(test_engine)

from tensorguard.platform.main import app
from tensorguard.platform import policy_engine
policy_engine.engine = test_engine # explicitly patch

with open("debug_output.txt", "a") as f:
    f.write(f"DEBUG: Database Engine (after patch): {database.engine}\n")
    f.write(f"DEBUG: PolicyEngine Engine: {policy_engine.engine}\n")

from tensorguard.platform.database import get_session
# Import all models to ensure they are registered with SQLModel.metadata
from tensorguard.platform.models.continuous_models import *
from tensorguard.platform.models.evidence_models import * 
from tensorguard.platform.models.peft_models import *

# Use in-memory DB for tests with transaction-based isolation
@pytest.fixture(name="session")
def session_fixture():
    # Create a connection-scoped transaction that we'll roll back after each test
    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    
    yield session
    
    # Rollback the outer transaction to undo all changes from the test
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session
    
    # Mock User
    from tensorguard.platform.auth import get_current_user
    from tensorguard.platform.dependencies import require_tenant_context
    from tensorguard.platform.models.core import User
    
    def get_current_user_override():
        return User(id="qa-user", email="qa@test.com", is_active=True, tenant_id="qa-tenant-1")

    async def require_tenant_context_override(
        x_tenant_id: str = Header("qa-tenant-1", alias="X-Tenant-ID")
    ):
        # Trust the header for QA tests to verify data isolation
        # In real app, this verifies against User.
        return x_tenant_id

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_user] = get_current_user_override
    app.dependency_overrides[require_tenant_context] = require_tenant_context_override
    
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

@pytest.fixture
def tenant_header():
    return {"X-Tenant-ID": "qa-tenant-1"}

@pytest.fixture(name="simulation_mode")
def simulation_mode_fixture():
    """Sets TG_SIMULATION to true for duration of test."""
    original = os.environ.get("TG_SIMULATION")
    os.environ["TG_SIMULATION"] = "true"
    yield
    if original is None:
        del os.environ["TG_SIMULATION"]
    else:
        os.environ["TG_SIMULATION"] = original
