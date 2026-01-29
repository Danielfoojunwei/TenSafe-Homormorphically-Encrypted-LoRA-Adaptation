"""
Regression Test Configuration

Provides fixtures for regression testing TensorGuardFlow Core.
Uses ephemeral in-memory database with transaction isolation.
"""

import pytest
import os
import sys
import logging
from typing import Generator, AsyncIterator
from fastapi.testclient import TestClient
from fastapi import Header
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock


# --- Async Mock Helpers ---

class AsyncIteratorMock:
    """Helper class to create async iterators for mocking async generators."""

    def __init__(self, items=None):
        self.items = items or []
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


def async_iter_mock(items=None):
    """Create a mock that returns an async iterator."""
    return AsyncIteratorMock(items or [])


def create_mock_workflow():
    """Create a properly configured mock PeftWorkflow for tests."""
    mock = MagicMock()
    mock.artifacts = {"adapter_path": "/mock/adapter", "tgsp_path": "/mock/tgsp"}
    mock.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
    mock.diagnosis = None

    # Use async iterators for stage methods
    mock._stage_train.return_value = async_iter_mock([])
    mock._stage_eval.return_value = async_iter_mock([])
    mock._stage_pack_tgsp.return_value = async_iter_mock([])
    mock._stage_emit_evidence.return_value = async_iter_mock([])

    return mock

# Configure logging for regression tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regression")

# Set deterministic environment BEFORE imports
os.environ["TG_DETERMINISTIC"] = "true"
os.environ["TG_DEMO_MODE"] = "true"
os.environ["TG_SIMULATION"] = "true"
os.environ["TG_ENABLE_LABS"] = "false"
os.environ["TG_ENVIRONMENT"] = "development"

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Create test engine BEFORE importing app
from tensorguard.platform import database

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
database.engine = test_engine

# Import models to register with SQLModel
from tensorguard.platform.models.continuous_models import *
from tensorguard.platform.models.evidence_models import *
from tensorguard.platform.models.tgflow_core_models import *
from tensorguard.platform.models.metrics_models import *
from tensorguard.platform.models.core import *

# Create tables
SQLModel.metadata.create_all(test_engine)

# Now import app
from tensorguard.platform.main import app
from tensorguard.platform.database import get_session
from tensorguard.platform.auth import get_current_user
from tensorguard.platform.dependencies import require_tenant_context
from tensorguard.platform.models.core import User


# --- Fixtures ---

@pytest.fixture(name="session", scope="function")
def session_fixture() -> Generator[Session, None, None]:
    """Provides a clean database session with transaction rollback."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(name="client", scope="function")
def client_fixture(session: Session) -> Generator[TestClient, None, None]:
    """Provides a test client with mocked auth and session."""

    def get_session_override():
        return session

    def get_current_user_override():
        return User(
            id="regression-user",
            email="regression@tensorguard.test",
            is_active=True,
            tenant_id="regression-tenant"
        )

    async def require_tenant_context_override(
        x_tenant_id: str = Header("regression-tenant", alias="X-Tenant-ID")
    ):
        return x_tenant_id

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_user] = get_current_user_override
    app.dependency_overrides[require_tenant_context] = require_tenant_context_override

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture(name="tenant_header")
def tenant_header_fixture() -> dict:
    """Standard tenant header for regression tests."""
    return {"X-Tenant-ID": "regression-tenant"}


@pytest.fixture(name="alt_tenant_header")
def alt_tenant_header_fixture() -> dict:
    """Alternative tenant for isolation tests."""
    return {"X-Tenant-ID": "alt-tenant"}


@pytest.fixture(name="sample_route_payload")
def sample_route_payload_fixture() -> dict:
    """Standard route creation payload."""
    return {
        "route_key": "regression-route-alpha",
        "base_model_ref": "microsoft/phi-2",
        "description": "Regression test route"
    }


@pytest.fixture(name="sample_feed_payload")
def sample_feed_payload_fixture() -> dict:
    """Standard feed connection payload."""
    return {
        "feed_type": "local",
        "feed_uri": "tests/fixtures/sample_data.jsonl",
        "privacy_mode": "off"
    }


@pytest.fixture(name="sample_policy_payload")
def sample_policy_payload_fixture() -> dict:
    """Standard policy configuration."""
    return {
        "novelty_threshold": 0.3,
        "promotion_threshold": 0.9,
        "forgetting_budget": 0.1,
        "regression_budget": 0.05,
        "auto_promote_to_canary": True,
        "auto_promote_to_stable": False,
        "max_total_adapters": 10,
        "max_fast_adapters": 5
    }


@pytest.fixture(name="n2he_feed_payload")
def n2he_feed_payload_fixture() -> dict:
    """Feed with N2HE privacy enabled."""
    return {
        "feed_type": "local",
        "feed_uri": "tests/fixtures/sample_data.jsonl",
        "privacy_mode": "n2he"
    }


@pytest.fixture(name="setup_complete_route")
def setup_complete_route_fixture(client: TestClient, tenant_header: dict,
                                  sample_route_payload: dict,
                                  sample_feed_payload: dict,
                                  sample_policy_payload: dict):
    """Creates a fully configured route ready for run_once."""
    route_key = sample_route_payload["route_key"]

    # Create route
    resp = client.post("/api/v1/tgflow/routes", headers=tenant_header, json=sample_route_payload)
    assert resp.status_code == 200, f"Failed to create route: {resp.text}"

    # Connect feed
    resp = client.post(f"/api/v1/tgflow/routes/{route_key}/feed",
                       headers=tenant_header, json=sample_feed_payload)
    assert resp.status_code == 200, f"Failed to connect feed: {resp.text}"

    # Set policy
    resp = client.post(f"/api/v1/tgflow/routes/{route_key}/policy",
                       headers=tenant_header, json=sample_policy_payload)
    assert resp.status_code == 200, f"Failed to set policy: {resp.text}"

    return route_key


# --- Test Data Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def create_test_fixtures():
    """Creates test fixture files if they don't exist."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    sample_data_path = os.path.join(fixtures_dir, "sample_data.jsonl")
    if not os.path.exists(sample_data_path):
        with open(sample_data_path, "w") as f:
            f.write('{"input": "test input 1", "output": "test output 1"}\n')
            f.write('{"input": "test input 2", "output": "test output 2"}\n')
            f.write('{"input": "test input 3", "output": "test output 3"}\n')

    yield
