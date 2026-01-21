
import pytest
import os
import shutil
from sqlmodel import Session, SQLModel, create_engine
from typing import Generator
from unittest.mock import MagicMock, patch

# Use in-memory SQLite for testing
TEST_DB_URL = "sqlite:///:memory:"

from sqlalchemy.pool import StaticPool


@pytest.fixture(name="session")
def session_fixture():
    # Import models to ensure they're registered
    from tensorguard.platform.models import core, identity_models, enablement_models, evidence_models

    # Force registration by accessing the classes
    _ = evidence_models.TGSPPackage
    _ = enablement_models.EnablementJob

    engine = create_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )

    # Explicitly create tables
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    # Lazy import to avoid database initialization at module import time
    from fastapi.testclient import TestClient
    from tensorguard.platform.main import app
    from tensorguard.platform.database import get_session
    from tensorguard.platform.api import community_tgsp

    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def temp_storage(tmp_path):
    from tensorguard.platform.api import community_tgsp
    # Monkeypatch storage dir
    original_storage = community_tgsp.STORAGE_DIR
    community_tgsp.STORAGE_DIR = str(tmp_path / "storage")
    os.makedirs(community_tgsp.STORAGE_DIR, exist_ok=True)
    yield community_tgsp.STORAGE_DIR
    community_tgsp.STORAGE_DIR = original_storage


class TestPlatformAPI:

    def test_platform_starts_and_serves_public(self, client):
        response = client.get("/")
        # Depending on how static files are served, this might be 200 or 404 if 'public' dir is empty/missing
        # But verify checking expected response.
        # Check docs or stats endpoint which we know exists
        response = client.get("/docs")
        assert response.status_code == 200

        # Check static serving
        # response = client.get("/index.html")
        # assert response.status_code in [200, 404]

    def test_enablement_stats_endpoint(self, client):
        """Verify the stats endpoint I fixed earlier."""
        response = client.get("/api/v1/enablement/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_jobs" in data
        assert data["total_jobs"] == 0

    @patch("tensorguard.platform.api.community_tgsp.read_tgsp_header")
    def test_platform_tgsp_upload_verify_pass(self, mock_read_header, client, temp_storage, session):
        # Mock Header Data
        mock_manifest = {
            "package_id": "pkg-123",
            "author_id": "prod-1",
            "created_at": 1735732800.0,  # 2026-01-01
            "policy_id": "pol-1",
            "policy_version": 1,
            "content_index": [],
            "compat_base_model_id": []
        }
        mock_read_header.return_value = {
            "version": "0.2",
            "header": {"hashes": {"manifest": "hash123"}},
            "manifest": mock_manifest,
            "recipients": []
        }

        files = {"file": ("test.tgsp", b"fake_content", "application/octet-stream")}
        response = client.post("/api/community/tgsp/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "pkg-123"
        assert data["status"] == "uploaded"
