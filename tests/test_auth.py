"""
Authentication Tests

Verifies that:
1. API key authentication works correctly
2. Development mode allows demo tokens
3. Production mode requires valid API keys
4. Tenant isolation is enforced
5. API key lifecycle (create, revoke, expire) works
"""

import sys
from datetime import datetime, timedelta

import pytest

# Skip all tests if sqlmodel not available
sqlmodel = pytest.importorskip("sqlmodel")
from sqlmodel import Session, SQLModel, create_engine

# Import directly to avoid tensorguard's crypto imports
sys.path.insert(0, "src")


def import_auth_module():
    """Import auth module avoiding crypto conflicts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "auth",
        "src/tensorguard/platform/tg_tinker_api/auth.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["auth_module"] = module
    spec.loader.exec_module(module)
    return module


auth_module = import_auth_module()


class TestTenantManagement:
    """Test tenant lifecycle management."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a test database engine."""
        db_path = tmp_path / "test_auth.db"
        engine = create_engine(f"sqlite:///{db_path}")

        # Import models to register them
        Tenant = auth_module.Tenant
        APIKey = auth_module.APIKey

        SQLModel.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def session(self, engine):
        """Create a test database session."""
        with Session(engine) as session:
            yield session

    @pytest.fixture
    def tenant_manager(self, session):
        """Create a tenant manager."""
        TenantManager = auth_module.TenantManager
        return TenantManager(session)

    def test_create_tenant(self, tenant_manager):
        """Test creating a tenant."""
        tenant = tenant_manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        assert tenant.id.startswith("tnt-")
        assert tenant.name == "Test Corp"
        assert tenant.email == "test@example.com"
        assert tenant.active is True
        assert tenant.max_training_clients == 10
        assert tenant.max_pending_jobs == 100

    def test_create_tenant_duplicate_email(self, tenant_manager):
        """Test that duplicate emails are rejected."""
        tenant_manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        with pytest.raises(ValueError) as exc_info:
            tenant_manager.create_tenant(
                name="Test Corp 2",
                email="test@example.com",
            )

        assert "already exists" in str(exc_info.value)

    def test_suspend_and_reactivate_tenant(self, tenant_manager):
        """Test suspending and reactivating a tenant."""
        tenant = tenant_manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        # Suspend
        suspended = tenant_manager.suspend_tenant(tenant.id, "Terms violation")
        assert suspended.active is False
        assert suspended.suspended_at is not None
        assert suspended.suspension_reason == "Terms violation"

        # Reactivate
        reactivated = tenant_manager.reactivate_tenant(tenant.id)
        assert reactivated.active is True
        assert reactivated.suspended_at is None
        assert reactivated.suspension_reason is None

    def test_create_api_key(self, tenant_manager):
        """Test creating an API key."""
        tenant = tenant_manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        api_key, raw_key = tenant_manager.create_api_key(
            tenant_id=tenant.id,
            name="Production Key",
        )

        assert api_key.id.startswith("key-")
        assert api_key.tenant_id == tenant.id
        assert api_key.name == "Production Key"
        assert api_key.active is True
        assert raw_key.startswith("tg_")
        assert len(raw_key) > 20  # Reasonable key length

    def test_create_api_key_with_expiration(self, tenant_manager):
        """Test creating an API key with expiration."""
        tenant = tenant_manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        api_key, raw_key = tenant_manager.create_api_key(
            tenant_id=tenant.id,
            name="Temp Key",
            expires_in_days=30,
        )

        assert api_key.expires_at is not None
        assert api_key.expires_at > datetime.utcnow()
        assert api_key.expires_at < datetime.utcnow() + timedelta(days=31)

    def test_revoke_api_key(self, tenant_manager):
        """Test revoking an API key."""
        tenant = tenant_manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        api_key, raw_key = tenant_manager.create_api_key(
            tenant_id=tenant.id,
            name="Test Key",
        )

        result = tenant_manager.revoke_api_key(api_key.id, "Key compromised")
        assert result is True

        # Check the key is revoked
        keys = tenant_manager.list_api_keys(tenant.id, include_revoked=False)
        assert len(keys) == 0

        keys_all = tenant_manager.list_api_keys(tenant.id, include_revoked=True)
        assert len(keys_all) == 1
        assert keys_all[0].active is False
        assert keys_all[0].revoked_reason == "Key compromised"


class TestAPIKeyAuth:
    """Test API key authentication."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a test database engine."""
        db_path = tmp_path / "test_auth.db"
        engine = create_engine(f"sqlite:///{db_path}")

        Tenant = auth_module.Tenant
        APIKey = auth_module.APIKey

        SQLModel.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def session(self, engine):
        """Create a test database session."""
        with Session(engine) as session:
            yield session

    @pytest.fixture
    def setup_tenant_with_key(self, session):
        """Set up a tenant with an API key."""
        TenantManager = auth_module.TenantManager
        manager = TenantManager(session)

        tenant = manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        api_key, raw_key = manager.create_api_key(
            tenant_id=tenant.id,
            name="Test Key",
        )

        return tenant, api_key, raw_key

    def test_api_key_auth_valid(self, session, setup_tenant_with_key):
        """Test valid API key authentication."""
        tenant, api_key, raw_key = setup_tenant_with_key

        APIKeyAuthProvider = auth_module.APIKeyAuthProvider
        provider = APIKeyAuthProvider()

        auth_ctx = provider.authenticate(raw_key, session)

        assert auth_ctx.tenant_id == tenant.id
        assert auth_ctx.tenant_name == tenant.name
        assert auth_ctx.key_id == api_key.id
        assert auth_ctx.key_name == api_key.name

    def test_api_key_auth_invalid_format(self, session):
        """Test that invalid key format is rejected."""
        from fastapi import HTTPException

        APIKeyAuthProvider = auth_module.APIKeyAuthProvider
        provider = APIKeyAuthProvider()

        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate("invalid_key", session)

        assert exc_info.value.status_code == 401
        assert "INVALID_API_KEY_FORMAT" in str(exc_info.value.detail)

    def test_api_key_auth_invalid_key(self, session, setup_tenant_with_key):
        """Test that invalid key is rejected."""
        from fastapi import HTTPException

        tenant, api_key, raw_key = setup_tenant_with_key

        APIKeyAuthProvider = auth_module.APIKeyAuthProvider
        provider = APIKeyAuthProvider()

        # Use a different key
        fake_key = "tg_" + "x" * 32

        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate(fake_key, session)

        assert exc_info.value.status_code == 401
        assert "INVALID_API_KEY" in str(exc_info.value.detail)

    def test_api_key_auth_expired(self, session):
        """Test that expired keys are rejected."""
        from fastapi import HTTPException

        TenantManager = auth_module.TenantManager
        APIKey = auth_module.APIKey

        manager = TenantManager(session)

        tenant = manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )

        api_key, raw_key = manager.create_api_key(
            tenant_id=tenant.id,
            name="Temp Key",
            expires_in_days=1,
        )

        # Manually expire the key
        api_key.expires_at = datetime.utcnow() - timedelta(days=1)
        session.add(api_key)
        session.commit()

        APIKeyAuthProvider = auth_module.APIKeyAuthProvider
        provider = APIKeyAuthProvider()

        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate(raw_key, session)

        assert exc_info.value.status_code == 401
        assert "API_KEY_EXPIRED" in str(exc_info.value.detail)

    def test_api_key_auth_revoked(self, session, setup_tenant_with_key):
        """Test that revoked keys are rejected."""
        from fastapi import HTTPException

        tenant, api_key, raw_key = setup_tenant_with_key

        TenantManager = auth_module.TenantManager
        manager = TenantManager(session)
        manager.revoke_api_key(api_key.id, "Test revocation")

        APIKeyAuthProvider = auth_module.APIKeyAuthProvider
        provider = APIKeyAuthProvider()

        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate(raw_key, session)

        assert exc_info.value.status_code == 401
        assert "INVALID_API_KEY" in str(exc_info.value.detail)

    def test_api_key_auth_suspended_tenant(self, session, setup_tenant_with_key):
        """Test that suspended tenants are rejected."""
        from fastapi import HTTPException

        tenant, api_key, raw_key = setup_tenant_with_key

        TenantManager = auth_module.TenantManager
        manager = TenantManager(session)
        manager.suspend_tenant(tenant.id, "Test suspension")

        APIKeyAuthProvider = auth_module.APIKeyAuthProvider
        provider = APIKeyAuthProvider()

        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate(raw_key, session)

        assert exc_info.value.status_code == 403
        assert "TENANT_INACTIVE" in str(exc_info.value.detail)


class TestDevelopmentAuth:
    """Test development mode authentication."""

    def test_dev_auth_accepts_any_token(self):
        """Test that development auth accepts any token."""
        DevelopmentAuthProvider = auth_module.DevelopmentAuthProvider
        provider = DevelopmentAuthProvider(allow_demo_tokens=True)

        auth_ctx = provider.authenticate("my-demo-token", None)

        assert auth_ctx.tenant_id.startswith("dev-")
        assert auth_ctx.auth_mode == auth_module.AuthMode.DEVELOPMENT

    def test_dev_auth_consistent_tenant_id(self):
        """Test that same token produces same tenant ID."""
        DevelopmentAuthProvider = auth_module.DevelopmentAuthProvider
        provider = DevelopmentAuthProvider(allow_demo_tokens=True)

        auth_ctx1 = provider.authenticate("my-token", None)
        auth_ctx2 = provider.authenticate("my-token", None)

        assert auth_ctx1.tenant_id == auth_ctx2.tenant_id

    def test_dev_auth_different_tokens_different_tenants(self):
        """Test that different tokens produce different tenant IDs."""
        DevelopmentAuthProvider = auth_module.DevelopmentAuthProvider
        provider = DevelopmentAuthProvider(allow_demo_tokens=True)

        auth_ctx1 = provider.authenticate("token-a", None)
        auth_ctx2 = provider.authenticate("token-b", None)

        assert auth_ctx1.tenant_id != auth_ctx2.tenant_id

    def test_dev_auth_disabled(self):
        """Test that dev auth can be disabled."""
        from fastapi import HTTPException

        DevelopmentAuthProvider = auth_module.DevelopmentAuthProvider
        provider = DevelopmentAuthProvider(allow_demo_tokens=False)

        with pytest.raises(HTTPException) as exc_info:
            provider.authenticate("any-token", None)

        assert exc_info.value.status_code == 401
        assert "DEV_AUTH_DISABLED" in str(exc_info.value.detail)


class TestAuthenticator:
    """Test the main authenticator class."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a test database engine."""
        db_path = tmp_path / "test_auth.db"
        engine = create_engine(f"sqlite:///{db_path}")

        Tenant = auth_module.Tenant
        APIKey = auth_module.APIKey

        SQLModel.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def session(self, engine):
        """Create a test database session."""
        with Session(engine) as session:
            yield session

    def test_authenticator_dev_mode_demo_token(self, session):
        """Test authenticator in development mode with demo token."""
        Authenticator = auth_module.Authenticator
        AuthMode = auth_module.AuthMode

        authenticator = Authenticator(mode=AuthMode.DEVELOPMENT)

        auth_ctx = authenticator.authenticate("demo-token", session)

        assert auth_ctx.tenant_id.startswith("dev-")
        assert auth_ctx.auth_mode == AuthMode.DEVELOPMENT

    def test_authenticator_dev_mode_api_key(self, session):
        """Test authenticator in development mode with API key."""
        TenantManager = auth_module.TenantManager
        Authenticator = auth_module.Authenticator
        AuthMode = auth_module.AuthMode

        # Set up tenant and key
        manager = TenantManager(session)
        tenant = manager.create_tenant(
            name="Test Corp",
            email="test@example.com",
        )
        api_key, raw_key = manager.create_api_key(
            tenant_id=tenant.id,
            name="Test Key",
        )

        authenticator = Authenticator(mode=AuthMode.DEVELOPMENT)

        # API key should work in dev mode too
        auth_ctx = authenticator.authenticate(raw_key, session)

        assert auth_ctx.tenant_id == tenant.id
        assert auth_ctx.auth_mode == AuthMode.PRODUCTION  # Key auth is "production-like"

    def test_authenticator_prod_mode_requires_api_key(self, session):
        """Test authenticator in production mode requires API key."""
        from fastapi import HTTPException

        Authenticator = auth_module.Authenticator
        AuthMode = auth_module.AuthMode

        authenticator = Authenticator(mode=AuthMode.PRODUCTION)

        with pytest.raises(HTTPException) as exc_info:
            authenticator.authenticate("demo-token", session)

        assert exc_info.value.status_code == 401


class TestTenantIsolation:
    """Test tenant isolation."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a test database engine."""
        db_path = tmp_path / "test_isolation.db"
        engine = create_engine(f"sqlite:///{db_path}")

        Tenant = auth_module.Tenant
        APIKey = auth_module.APIKey

        SQLModel.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def session(self, engine):
        """Create a test database session."""
        with Session(engine) as session:
            yield session

    def test_api_key_only_works_for_own_tenant(self, session):
        """Test that API key only authenticates its own tenant."""
        TenantManager = auth_module.TenantManager
        APIKeyAuthProvider = auth_module.APIKeyAuthProvider

        manager = TenantManager(session)

        # Create two tenants
        tenant_a = manager.create_tenant(
            name="Tenant A",
            email="a@example.com",
        )
        tenant_b = manager.create_tenant(
            name="Tenant B",
            email="b@example.com",
        )

        # Create keys for each
        key_a, raw_key_a = manager.create_api_key(tenant_a.id, "Key A")
        key_b, raw_key_b = manager.create_api_key(tenant_b.id, "Key B")

        provider = APIKeyAuthProvider()

        # Key A should authenticate tenant A
        auth_a = provider.authenticate(raw_key_a, session)
        assert auth_a.tenant_id == tenant_a.id

        # Key B should authenticate tenant B
        auth_b = provider.authenticate(raw_key_b, session)
        assert auth_b.tenant_id == tenant_b.id

    def test_different_tokens_isolated_in_dev_mode(self):
        """Test that different tokens get different tenant IDs."""
        DevelopmentAuthProvider = auth_module.DevelopmentAuthProvider

        provider = DevelopmentAuthProvider(allow_demo_tokens=True)

        # Simulate multiple clients
        client_a = provider.authenticate("client-a-token", None)
        client_b = provider.authenticate("client-b-token", None)

        # Should have different tenant IDs
        assert client_a.tenant_id != client_b.tenant_id


class TestAuthContext:
    """Test AuthContext functionality."""

    def test_has_scope_wildcard(self):
        """Test that wildcard scope matches everything."""
        AuthContext = auth_module.AuthContext
        AuthMode = auth_module.AuthMode

        ctx = AuthContext(
            tenant_id="tenant-1",
            tenant_name="Test",
            key_id=None,
            key_name=None,
            scopes=["*"],
            auth_mode=AuthMode.DEVELOPMENT,
        )

        assert ctx.has_scope("any_scope") is True
        assert ctx.has_scope("another_scope") is True

    def test_has_scope_specific(self):
        """Test that specific scopes are checked."""
        AuthContext = auth_module.AuthContext
        AuthMode = auth_module.AuthMode

        ctx = AuthContext(
            tenant_id="tenant-1",
            tenant_name="Test",
            key_id=None,
            key_name=None,
            scopes=["read", "write"],
            auth_mode=AuthMode.DEVELOPMENT,
        )

        assert ctx.has_scope("read") is True
        assert ctx.has_scope("write") is True
        assert ctx.has_scope("admin") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
