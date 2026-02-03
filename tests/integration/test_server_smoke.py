"""
Platform Server Smoke Tests.

Tests that the server boots and responds to basic endpoints.

NOTE: These tests require fresh module imports since the database
configuration is read at module load time. We use importlib.reload
to ensure the test environment is used.
"""

import os
import sys
import tempfile

import pytest


@pytest.fixture(scope="module")
def test_env():
    """Set up test environment before importing platform modules."""
    # Create temp directory for test database
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")

    # Store original env vars
    original_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "TG_ENVIRONMENT": os.environ.get("TG_ENVIRONMENT"),
    }

    # Set test environment BEFORE importing platform modules
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["TG_ENVIRONMENT"] = "test"

    yield tmpdir

    # Restore original env vars
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Clean up temp directory
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="module")
def client(test_env):
    """Create a test client for the platform server."""
    # Remove cached platform modules to force fresh imports with test env
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith("tensorguard.platform")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Now import with test environment set
    from fastapi.testclient import TestClient

    from tensorguard.platform.main import app

    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "3.0.0"

    def test_ready_returns_200(self, client):
        """Ready endpoint should return 200 when healthy."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True

    def test_live_returns_200(self, client):
        """Live endpoint should return 200."""
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True

    def test_version_returns_info(self, client):
        """Version endpoint should return service info."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "TG-Tinker"
        assert data["version"] == "3.0.0"
        assert data["api_version"] == "v1"


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_service_info(self, client):
        """Root should return service information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "TG-Tinker"
        assert "version" in data
        assert "docs" in data


class TestSecurityHeaders:
    """Test security headers are present."""

    def test_x_content_type_options(self, client):
        """X-Content-Type-Options header should be set."""
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client):
        """X-Frame-Options header should be set."""
        response = client.get("/health")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self, client):
        """X-XSS-Protection header should be set."""
        response = client.get("/health")
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"


class TestDocsEndpoints:
    """Test documentation endpoints."""

    def test_openapi_json(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    def test_docs_available(self, client):
        """Swagger UI should be available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """ReDoc should be available."""
        response = client.get("/redoc")
        assert response.status_code == 200
