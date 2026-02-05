"""
TG-Tinker Authentication Module.

Provides proper tenant authentication with multiple modes:
- Development: Relaxed authentication for local testing
- Production: Strict authentication with API keys or JWTs

In production:
- Requires valid API key or JWT token
- Tenant ID is extracted from the token/key
- Invalid tokens result in 401 Unauthorized
"""

import hashlib
import hmac
import logging
import os
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Header, HTTPException, Request, status
from sqlmodel import JSON, Column, Field, Session, SQLModel, select

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


class AuthMode(str, Enum):
    """Authentication mode."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_auth_mode() -> AuthMode:
    """Get the current authentication mode from environment."""
    mode = os.getenv("TG_AUTH_MODE", os.getenv("TG_ENVIRONMENT", "development")).lower()
    if mode in ("production", "prod"):
        return AuthMode.PRODUCTION
    return AuthMode.DEVELOPMENT


# ==============================================================================
# Database Models
# ==============================================================================


def generate_tenant_id() -> str:
    """Generate a tenant ID."""
    return f"tnt-{uuid.uuid4()}"


def generate_api_key() -> str:
    """Generate a secure API key."""
    # Format: tg_<random_bytes>
    # This makes it easy to identify TG API keys and provides sufficient entropy
    return f"tg_{secrets.token_urlsafe(32)}"


class Tenant(SQLModel, table=True):
    """Tenant model for multi-tenancy."""

    __tablename__ = "tinker_tenants"

    id: str = Field(default_factory=generate_tenant_id, primary_key=True)
    name: str = Field(index=True)
    email: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Status
    active: bool = Field(default=True)
    suspended_at: Optional[datetime] = None
    suspension_reason: Optional[str] = None

    # Quotas
    max_training_clients: int = Field(default=10)
    max_pending_jobs: int = Field(default=100)
    max_storage_bytes: int = Field(default=10 * 1024 * 1024 * 1024)  # 10 GB default


class APIKey(SQLModel, table=True):
    """API key for tenant authentication."""

    __tablename__ = "tinker_api_keys"

    id: str = Field(default_factory=lambda: f"key-{uuid.uuid4()}", primary_key=True)
    tenant_id: str = Field(foreign_key="tinker_tenants.id", index=True)

    # Key is stored as hash for security
    key_hash: str = Field(index=True)
    key_prefix: str = Field(index=True)  # First 8 chars for lookup

    # Metadata
    name: str  # Human-readable name
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Status
    active: bool = Field(default=True)
    revoked_at: Optional[datetime] = None
    revoked_reason: Optional[str] = None

    # Permissions (for future use)
    scopes: List[str] = Field(default_factory=lambda: ["*"], sa_column=Column(JSON))

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()


# ==============================================================================
# Authentication Result
# ==============================================================================


@dataclass
class AuthContext:
    """Authentication context containing tenant and key info."""

    tenant_id: str
    tenant_name: str
    key_id: Optional[str]
    key_name: Optional[str]
    scopes: List[str]
    auth_mode: AuthMode

    def has_scope(self, scope: str) -> bool:
        """Check if context has a specific scope."""
        return "*" in self.scopes or scope in self.scopes


# ==============================================================================
# Authentication Providers
# ==============================================================================


class AuthProvider(ABC):
    """Base class for authentication providers."""

    @abstractmethod
    def authenticate(self, token: str, session: Session) -> AuthContext:
        """
        Authenticate a token and return the auth context.

        Args:
            token: The authentication token (API key or JWT)
            session: Database session

        Returns:
            AuthContext with tenant information

        Raises:
            HTTPException: If authentication fails
        """
        pass


class APIKeyAuthProvider(AuthProvider):
    """API key-based authentication."""

    def authenticate(self, token: str, session: Session) -> AuthContext:
        """Authenticate using API key."""
        if not token.startswith("tg_"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "INVALID_API_KEY_FORMAT",
                        "message": "API key must start with 'tg_'",
                    }
                },
            )

        # Get prefix for lookup
        key_prefix = token[:11]  # "tg_" + 8 chars
        key_hash = APIKey.hash_key(token)

        # Find the key
        statement = select(APIKey).where(
            APIKey.key_prefix == key_prefix,
            APIKey.key_hash == key_hash,
            APIKey.active == True,  # noqa: E712
        )
        api_key = session.exec(statement).first()

        if api_key is None:
            logger.warning(f"Invalid API key attempt with prefix {key_prefix}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "INVALID_API_KEY",
                        "message": "Invalid or revoked API key",
                    }
                },
            )

        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "API_KEY_EXPIRED",
                        "message": "API key has expired",
                    }
                },
            )

        # Get tenant
        tenant = session.get(Tenant, api_key.tenant_id)
        if tenant is None or not tenant.active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": {
                        "code": "TENANT_INACTIVE",
                        "message": "Tenant account is inactive or suspended",
                    }
                },
            )

        # Update last used
        api_key.last_used_at = datetime.utcnow()
        session.add(api_key)
        session.commit()

        return AuthContext(
            tenant_id=tenant.id,
            tenant_name=tenant.name,
            key_id=api_key.id,
            key_name=api_key.name,
            scopes=api_key.scopes or ["*"],
            auth_mode=AuthMode.PRODUCTION,
        )


class DevelopmentAuthProvider(AuthProvider):
    """
    Development authentication provider.

    In development mode, allows any token and derives a consistent
    tenant ID from the token hash. This enables local testing without
    setting up real tenants.

    WARNING: This should NEVER be used in production!
    """

    def __init__(self, allow_demo_tokens: bool = True):
        self.allow_demo_tokens = allow_demo_tokens

    def authenticate(self, token: str, session: Session) -> AuthContext:
        """Authenticate in development mode."""
        if not self.allow_demo_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "DEV_AUTH_DISABLED",
                        "message": "Development authentication is disabled",
                    }
                },
            )

        # Derive consistent tenant ID from token
        tenant_id = f"dev-{hashlib.sha256(token.encode()).hexdigest()[:8]}"

        logger.warning(
            f"Development auth: token hash derived tenant_id={tenant_id}. "
            "This is NOT secure and should only be used for local development."
        )

        return AuthContext(
            tenant_id=tenant_id,
            tenant_name=f"Development ({tenant_id})",
            key_id=None,
            key_name=None,
            scopes=["*"],
            auth_mode=AuthMode.DEVELOPMENT,
        )


# ==============================================================================
# Authenticator
# ==============================================================================


class Authenticator:
    """
    Main authenticator that handles token validation and tenant extraction.

    Uses different providers based on the authentication mode.
    """

    def __init__(
        self,
        mode: Optional[AuthMode] = None,
        allow_demo_tokens: bool = True,
    ):
        """
        Initialize authenticator.

        Args:
            mode: Authentication mode (defaults to environment)
            allow_demo_tokens: Whether to allow demo tokens in development
        """
        self.mode = mode or get_auth_mode()
        self.api_key_provider = APIKeyAuthProvider()
        self.dev_provider = DevelopmentAuthProvider(allow_demo_tokens)

    def authenticate(self, token: str, session: Session) -> AuthContext:
        """
        Authenticate a token.

        Args:
            token: The authentication token
            session: Database session

        Returns:
            AuthContext with tenant information

        Raises:
            HTTPException: If authentication fails
        """
        if self.mode == AuthMode.PRODUCTION:
            # Production: must use valid API key
            return self.api_key_provider.authenticate(token, session)

        # Development: try API key first, fall back to demo
        if token.startswith("tg_"):
            try:
                return self.api_key_provider.authenticate(token, session)
            except HTTPException:
                # Fall through to dev auth
                pass

        return self.dev_provider.authenticate(token, session)


# ==============================================================================
# Tenant Management
# ==============================================================================


class TenantManager:
    """Manages tenant lifecycle operations."""

    def __init__(self, session: Session):
        self.session = session

    def create_tenant(
        self,
        name: str,
        email: str,
        max_training_clients: int = 10,
        max_pending_jobs: int = 100,
        max_storage_bytes: int = 10 * 1024 * 1024 * 1024,
    ) -> Tenant:
        """
        Create a new tenant.

        Args:
            name: Tenant name
            email: Tenant email (must be unique)
            max_training_clients: Maximum training clients allowed
            max_pending_jobs: Maximum pending jobs in queue
            max_storage_bytes: Maximum storage allowed

        Returns:
            Created tenant

        Raises:
            ValueError: If email already exists
        """
        # Check for existing email
        existing = self.session.exec(
            select(Tenant).where(Tenant.email == email)
        ).first()
        if existing:
            raise ValueError(f"Tenant with email {email} already exists")

        tenant = Tenant(
            name=name,
            email=email,
            max_training_clients=max_training_clients,
            max_pending_jobs=max_pending_jobs,
            max_storage_bytes=max_storage_bytes,
        )

        self.session.add(tenant)
        self.session.commit()
        self.session.refresh(tenant)

        logger.info(f"Created tenant {tenant.id}: {name}")
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self.session.get(Tenant, tenant_id)

    def suspend_tenant(self, tenant_id: str, reason: str) -> Optional[Tenant]:
        """Suspend a tenant."""
        tenant = self.session.get(Tenant, tenant_id)
        if tenant is None:
            return None

        tenant.active = False
        tenant.suspended_at = datetime.utcnow()
        tenant.suspension_reason = reason
        tenant.updated_at = datetime.utcnow()

        self.session.add(tenant)
        self.session.commit()
        self.session.refresh(tenant)

        logger.warning(f"Suspended tenant {tenant_id}: {reason}")
        return tenant

    def reactivate_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Reactivate a suspended tenant."""
        tenant = self.session.get(Tenant, tenant_id)
        if tenant is None:
            return None

        tenant.active = True
        tenant.suspended_at = None
        tenant.suspension_reason = None
        tenant.updated_at = datetime.utcnow()

        self.session.add(tenant)
        self.session.commit()
        self.session.refresh(tenant)

        logger.info(f"Reactivated tenant {tenant_id}")
        return tenant

    def create_api_key(
        self,
        tenant_id: str,
        name: str,
        expires_in_days: Optional[int] = None,
        scopes: Optional[List[str]] = None,
    ) -> Tuple[APIKey, str]:
        """
        Create a new API key for a tenant.

        Args:
            tenant_id: Tenant ID
            name: Human-readable name for the key
            expires_in_days: Days until expiration (None for no expiration)
            scopes: Permission scopes (defaults to all)

        Returns:
            Tuple of (APIKey record, raw API key string)

        Raises:
            ValueError: If tenant doesn't exist
        """
        tenant = self.session.get(Tenant, tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant {tenant_id} not found")

        # Generate raw key
        raw_key = generate_api_key()

        # Create record
        api_key = APIKey(
            tenant_id=tenant_id,
            key_hash=APIKey.hash_key(raw_key),
            key_prefix=raw_key[:11],
            name=name,
            scopes=scopes or ["*"],
            expires_at=(
                datetime.utcnow() + timedelta(days=expires_in_days)
                if expires_in_days
                else None
            ),
        )

        self.session.add(api_key)
        self.session.commit()
        self.session.refresh(api_key)

        logger.info(f"Created API key {api_key.id} for tenant {tenant_id}")

        # Return both the record and the raw key
        # (raw key can only be shown once, then only hash is stored)
        return api_key, raw_key

    def revoke_api_key(self, key_id: str, reason: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: API key ID
            reason: Reason for revocation

        Returns:
            True if revoked, False if not found
        """
        api_key = self.session.get(APIKey, key_id)
        if api_key is None:
            return False

        api_key.active = False
        api_key.revoked_at = datetime.utcnow()
        api_key.revoked_reason = reason

        self.session.add(api_key)
        self.session.commit()

        logger.info(f"Revoked API key {key_id}: {reason}")
        return True

    def list_api_keys(self, tenant_id: str, include_revoked: bool = False) -> List[APIKey]:
        """List API keys for a tenant."""
        statement = select(APIKey).where(APIKey.tenant_id == tenant_id)
        if not include_revoked:
            statement = statement.where(APIKey.active == True)  # noqa: E712
        return list(self.session.exec(statement))


# ==============================================================================
# FastAPI Dependencies
# ==============================================================================


# Global authenticator instance
_authenticator: Optional[Authenticator] = None


def get_authenticator() -> Authenticator:
    """Get the global authenticator instance."""
    global _authenticator
    if _authenticator is None:
        _authenticator = Authenticator()
    return _authenticator


def reset_authenticator() -> None:
    """Reset the global authenticator (for testing)."""
    global _authenticator
    _authenticator = None


async def get_auth_context(
    authorization: str = Header(..., description="Bearer token or API key"),
    session: Session = None,
) -> AuthContext:
    """
    FastAPI dependency to get authentication context.

    Usage:
        @router.get("/endpoint")
        async def endpoint(auth: AuthContext = Depends(get_auth_context)):
            tenant_id = auth.tenant_id
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "INVALID_AUTH_HEADER",
                    "message": "Authorization header must start with 'Bearer '",
                }
            },
        )

    token = authorization[7:]

    # Note: session is injected by the caller, not via Depends
    # This allows for proper session handling in routes
    authenticator = get_authenticator()
    return authenticator.authenticate(token, session)


async def get_tenant_id_from_auth(
    authorization: str = Header(..., description="Bearer token or API key"),
) -> str:
    """
    FastAPI dependency to get just the tenant ID.

    This is a backward-compatible replacement for the old get_tenant_id.
    It works in both development and production modes.

    Usage:
        @router.get("/endpoint")
        async def endpoint(tenant_id: str = Depends(get_tenant_id_from_auth)):
            ...
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "INVALID_AUTH_HEADER",
                    "message": "Authorization header must start with 'Bearer '",
                }
            },
        )

    token = authorization[7:]
    mode = get_auth_mode()

    if mode == AuthMode.PRODUCTION:
        # In production, we need a database session to validate the key
        # This dependency doesn't have access to session, so we raise an error
        # and expect the route to use get_auth_context instead
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "AUTH_CONFIG_ERROR",
                    "message": "Production mode requires full auth context. Update route to use get_auth_context.",
                }
            },
        )

    # Development mode: derive tenant ID from token hash
    tenant_id = f"dev-{hashlib.sha256(token.encode()).hexdigest()[:8]}"
    return tenant_id
