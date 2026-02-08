"""
Authentication Integration Tests.

Tests the full user signup, login, and token flow.
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
    db_path = os.path.join(tmpdir, "test_auth.db")

    # Store original env vars
    original_env = {
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "TG_ENVIRONMENT": os.environ.get("TG_ENVIRONMENT"),
        "TG_REQUIRE_EMAIL_VERIFICATION": os.environ.get("TG_REQUIRE_EMAIL_VERIFICATION"),
    }

    # Set test environment BEFORE importing platform modules
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["TG_ENVIRONMENT"] = "test"
    os.environ["TG_REQUIRE_EMAIL_VERIFICATION"] = "false"  # Disable for testing

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


class TestUserSignup:
    """Test user registration endpoints."""

    def test_signup_success(self, client):
        """User can register with valid credentials."""
        response = client.post(
            "/auth/signup",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "name": "Test User"
            }
        )
        assert response.status_code == 201, f"Signup failed: {response.json()}"
        data = response.json()
        assert "user_id" in data
        assert data["email"] == "test@example.com"
        assert data["name"] == "Test User"

    def test_signup_duplicate_email(self, client):
        """Cannot register with existing email."""
        # First registration
        client.post(
            "/auth/signup",
            json={
                "email": "duplicate@example.com",
                "password": "SecurePass123!",
                "name": "First User"
            }
        )

        # Second registration with same email
        response = client.post(
            "/auth/signup",
            json={
                "email": "duplicate@example.com",
                "password": "DifferentPass456!",
                "name": "Second User"
            }
        )
        assert response.status_code == 409
        assert "email_exists" in response.json().get("detail", {}).get("error", "")

    def test_signup_weak_password(self, client):
        """Cannot register with weak password."""
        response = client.post(
            "/auth/signup",
            json={
                "email": "weak@example.com",
                "password": "weak",
                "name": "Weak Password User"
            }
        )
        assert response.status_code == 422  # Validation error


class TestUserLogin:
    """Test user login endpoints."""

    def test_login_success(self, client):
        """User can login with valid credentials."""
        # First create a user
        client.post(
            "/auth/signup",
            json={
                "email": "login@example.com",
                "password": "LoginPass123!",
                "name": "Login User"
            }
        )

        # Login
        response = client.post(
            "/auth/token",
            data={
                "username": "login@example.com",
                "password": "LoginPass123!"
            }
        )
        assert response.status_code == 200, f"Login failed: {response.json()}"
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "Bearer"

    def test_login_invalid_password(self, client):
        """Cannot login with wrong password."""
        # First create a user
        client.post(
            "/auth/signup",
            json={
                "email": "wrongpass@example.com",
                "password": "CorrectPass123!",
                "name": "Wrong Pass User"
            }
        )

        # Login with wrong password
        response = client.post(
            "/auth/token",
            data={
                "username": "wrongpass@example.com",
                "password": "WrongPassword!"
            }
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """Cannot login with non-existent email."""
        response = client.post(
            "/auth/token",
            data={
                "username": "nonexistent@example.com",
                "password": "SomePassword123!"
            }
        )
        assert response.status_code == 401


class TestTokenRefresh:
    """Test token refresh endpoints."""

    def test_refresh_token_success(self, client):
        """Can refresh token with valid refresh token."""
        # Create user and login
        client.post(
            "/auth/signup",
            json={
                "email": "refresh@example.com",
                "password": "RefreshPass123!",
                "name": "Refresh User"
            }
        )

        login_response = client.post(
            "/auth/token",
            data={
                "username": "refresh@example.com",
                "password": "RefreshPass123!"
            }
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh
        response = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200, f"Refresh failed: {response.json()}"
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data


class TestGetCurrentUser:
    """Test getting current user info."""

    def test_get_me_success(self, client):
        """Can get current user info with valid token."""
        import uuid
        unique_email = f"me_{uuid.uuid4().hex[:8]}@example.com"

        # Create user and login
        signup_response = client.post(
            "/auth/signup",
            json={
                "email": unique_email,
                "password": "MeSecurePass123!",
                "name": "Me User"
            }
        )
        assert signup_response.status_code == 201, f"Signup failed: {signup_response.json()}"

        login_response = client.post(
            "/auth/token",
            data={
                "username": unique_email,
                "password": "MeSecurePass123!"
            }
        )
        assert login_response.status_code == 200, f"Login failed: {login_response.json()}"
        access_token = login_response.json()["access_token"]

        # Get current user
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        assert response.status_code == 200, f"Get me failed: {response.json()}"
        data = response.json()
        assert data["email"] == unique_email
        assert data["name"] == "Me User"

    def test_get_me_unauthorized(self, client):
        """Cannot get user info without token."""
        response = client.get("/auth/me")
        assert response.status_code == 401


class TestHealthEndpoints:
    """Test that health endpoints still work after auth integration."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_root_includes_auth_info(self, client):
        """Root endpoint should include auth info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "auth" in data
        assert "/auth/signup" in data["auth"]["signup"]
        assert "/auth/token" in data["auth"]["login"]


class TestOpenAPISpec:
    """Test OpenAPI spec includes auth endpoints."""

    def test_openapi_has_auth_tag(self, client):
        """OpenAPI spec should have auth tag."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()

        tags = [t["name"] for t in data.get("tags", [])]
        assert "auth" in tags

    def test_openapi_has_auth_endpoints(self, client):
        """OpenAPI spec should have auth endpoints."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()

        paths = data.get("paths", {})
        assert "/auth/signup" in paths
        assert "/auth/token" in paths
        assert "/auth/me" in paths
