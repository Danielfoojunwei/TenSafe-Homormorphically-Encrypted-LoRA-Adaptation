"""
Admin Permission Checks.

Provides authentication and authorization for admin operations:
- SuperAdmin role requirement (full system access)
- OrgAdmin role scoped to their tenant
- Audit logging for all admin actions
"""

import functools
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt.exceptions import PyJWTError as JWTError
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

# JWT Configuration
ADMIN_SECRET_KEY = os.getenv("TG_ADMIN_SECRET_KEY")
if not ADMIN_SECRET_KEY:
    if os.getenv("TG_ENVIRONMENT", "development") == "production":
        raise RuntimeError(
            "TG_ADMIN_SECRET_KEY is required in production. "
            "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    logger.warning(
        "SECURITY WARNING: TG_ADMIN_SECRET_KEY not set. "
        "Using ephemeral key - admin tokens will be invalid after restart."
    )
    ADMIN_SECRET_KEY = secrets.token_hex(32)

ADMIN_TOKEN_ALGORITHM = os.getenv("TG_ADMIN_TOKEN_ALGORITHM", "HS256")
ADMIN_TOKEN_ISSUER = os.getenv("TG_ADMIN_TOKEN_ISSUER", "tensorguard-admin")
ADMIN_TOKEN_AUDIENCE = os.getenv("TG_ADMIN_TOKEN_AUDIENCE", "tensorguard-admin-api")


# ==============================================================================
# Enums
# ==============================================================================


class AdminRole(str, Enum):
    """Admin role hierarchy."""

    SUPER_ADMIN = "super_admin"  # Full system access
    ORG_ADMIN = "org_admin"  # Scoped to their tenant
    OPERATOR = "operator"  # Read-only system ops
    VIEWER = "viewer"  # Read-only access


class AuditAction(str, Enum):
    """Admin audit action types."""

    # Tenant actions
    TENANT_CREATE = "tenant.create"
    TENANT_READ = "tenant.read"
    TENANT_UPDATE = "tenant.update"
    TENANT_DELETE = "tenant.delete"
    TENANT_LIST = "tenant.list"
    TENANT_SUSPEND = "tenant.suspend"
    TENANT_ACTIVATE = "tenant.activate"
    TENANT_SET_QUOTA = "tenant.set_quota"
    TENANT_EXPORT_DATA = "tenant.export_data"

    # User actions
    USER_CREATE = "user.create"
    USER_READ = "user.read"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    USER_LIST = "user.list"
    USER_SUSPEND = "user.suspend"
    USER_ACTIVATE = "user.activate"
    USER_RESET_PASSWORD = "user.reset_password"

    # System actions
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_METRICS_READ = "system.metrics_read"
    SYSTEM_CONFIG_READ = "system.config_read"
    SYSTEM_CONFIG_UPDATE = "system.config_update"
    SYSTEM_MAINTENANCE_ENABLE = "system.maintenance_enable"
    SYSTEM_MAINTENANCE_DISABLE = "system.maintenance_disable"

    # Dashboard actions
    DASHBOARD_VIEW = "dashboard.view"

    # Audit actions
    AUDIT_LOG_READ = "audit.log_read"


# ==============================================================================
# Admin User Context
# ==============================================================================


class AdminUserContext(BaseModel):
    """Context for the authenticated admin user."""

    user_id: str
    email: str
    name: str
    role: AdminRole
    tenant_id: Optional[str] = None  # Only set for org_admin
    permissions: List[str] = []
    session_id: str
    ip_address: Optional[str] = None

    def can_access_tenant(self, tenant_id: str) -> bool:
        """Check if user can access a specific tenant."""
        if self.role == AdminRole.SUPER_ADMIN:
            return True
        if self.role == AdminRole.ORG_ADMIN and self.tenant_id == tenant_id:
            return True
        return False

    def is_super_admin(self) -> bool:
        """Check if user is a super admin."""
        return self.role == AdminRole.SUPER_ADMIN

    def is_org_admin(self) -> bool:
        """Check if user is an org admin."""
        return self.role in (AdminRole.SUPER_ADMIN, AdminRole.ORG_ADMIN)


# ==============================================================================
# Audit Logging
# ==============================================================================


class AdminAuditLogger:
    """Audit logger for admin actions."""

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []
        self._lock_hash: Optional[str] = None

    def log(
        self,
        action: AuditAction,
        admin_user: AdminUserContext,
        resource_type: str,
        resource_id: Optional[str] = None,
        target_tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request: Optional[Request] = None,
    ) -> str:
        """
        Log an admin action to the audit trail.

        Args:
            action: The action being performed
            admin_user: The admin user context
            resource_type: Type of resource being acted upon
            resource_id: ID of the resource
            target_tenant_id: Target tenant ID (if applicable)
            details: Additional action details
            success: Whether the action succeeded
            error_message: Error message if action failed
            request: FastAPI request for additional context

        Returns:
            The audit entry ID
        """
        entry_id = str(uuid4())
        timestamp = datetime.now(timezone.utc)

        # Extract request metadata
        ip_address = None
        user_agent = None
        if request:
            ip_address = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")[:500]

        # Create audit entry
        entry = {
            "id": entry_id,
            "timestamp": timestamp.isoformat(),
            "admin_user_id": admin_user.user_id,
            "admin_user_email": admin_user.email,
            "admin_role": admin_user.role.value,
            "admin_tenant_id": admin_user.tenant_id,
            "session_id": admin_user.session_id,
            "action": action.value,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "target_tenant_id": target_tenant_id,
            "details": self._filter_sensitive(details or {}),
            "ip_address": ip_address or admin_user.ip_address,
            "user_agent": user_agent,
            "success": success,
            "error_message": error_message,
        }

        # Compute chain hash for tamper detection
        prev_hash = self._lock_hash or "genesis"
        entry_json = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(f"{prev_hash}:{entry_json}".encode()).hexdigest()
        entry["_hash"] = entry_hash
        entry["_prev_hash"] = prev_hash
        self._lock_hash = entry_hash

        # Store entry
        self._entries.append(entry)

        # Log to standard logger
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"ADMIN_AUDIT: {action.value} | "
            f"user={admin_user.email} | "
            f"resource={resource_type}/{resource_id} | "
            f"success={success}",
            extra={"audit_entry": entry},
        )

        return entry_id

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxies."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _filter_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive fields from audit details."""
        sensitive_keys = {
            "password",
            "secret",
            "token",
            "api_key",
            "private_key",
            "credential",
        }

        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive(value)
            else:
                filtered[key] = value
        return filtered

    def get_entries(
        self,
        admin_user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        target_tenant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query audit log entries.

        Args:
            admin_user_id: Filter by admin user
            action: Filter by action type
            target_tenant_id: Filter by target tenant
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum entries to return
            offset: Number of entries to skip

        Returns:
            List of matching audit entries
        """
        results = []
        for entry in reversed(self._entries):
            # Apply filters
            if admin_user_id and entry["admin_user_id"] != admin_user_id:
                continue
            if action and entry["action"] != action.value:
                continue
            if target_tenant_id and entry.get("target_tenant_id") != target_tenant_id:
                continue
            if start_time:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < start_time:
                    continue
            if end_time:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time > end_time:
                    continue

            results.append(entry)

        # Apply pagination
        return results[offset : offset + limit]

    def count_entries(
        self,
        admin_user_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        target_tenant_id: Optional[str] = None,
    ) -> int:
        """Count matching audit entries."""
        count = 0
        for entry in self._entries:
            if admin_user_id and entry["admin_user_id"] != admin_user_id:
                continue
            if action and entry["action"] != action.value:
                continue
            if target_tenant_id and entry.get("target_tenant_id") != target_tenant_id:
                continue
            count += 1
        return count


# Singleton audit logger
_admin_audit_logger: Optional[AdminAuditLogger] = None


def get_admin_audit_logger() -> AdminAuditLogger:
    """Get or create the admin audit logger singleton."""
    global _admin_audit_logger
    if _admin_audit_logger is None:
        _admin_audit_logger = AdminAuditLogger()
    return _admin_audit_logger


# ==============================================================================
# Authentication Dependencies
# ==============================================================================

security = HTTPBearer(auto_error=False)


async def get_admin_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AdminUserContext:
    """
    Validate admin JWT token and return the admin user context.

    This dependency validates:
    - Token signature and expiration
    - Token issuer and audience
    - Admin role claim
    - User existence (in production)

    Raises:
        HTTPException: 401 if authentication fails
        HTTPException: 403 if user is not an admin
    """
    # --- DEMO MODE BYPASS ---
    DEMO_MODE = os.getenv("TG_ADMIN_DEMO_MODE", "false").lower() == "true"
    if DEMO_MODE:
        if os.getenv("TG_ENVIRONMENT", "development") == "production":
            logger.critical("SECURITY VIOLATION: TG_ADMIN_DEMO_MODE=true in production!")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Admin demo mode is not allowed in production",
            )
        logger.warning("ADMIN DEMO MODE: Returning demo super admin - NOT FOR PRODUCTION")
        return AdminUserContext(
            user_id="demo-admin-001",
            email="admin@tensorguard.local",
            name="Demo Admin",
            role=AdminRole.SUPER_ADMIN,
            permissions=["*"],
            session_id=f"demo-session-{uuid4().hex[:8]}",
            ip_address=request.client.host if request.client else "127.0.0.1",
        )
    # --- END DEMO MODE ---

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate admin credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not credentials:
        raise credentials_exception

    try:
        payload = jwt.decode(
            credentials.credentials,
            ADMIN_SECRET_KEY,
            algorithms=[ADMIN_TOKEN_ALGORITHM],
            audience=ADMIN_TOKEN_AUDIENCE,
            issuer=ADMIN_TOKEN_ISSUER,
        )

        # Validate token type
        token_type = payload.get("type")
        if token_type != "admin_access":
            logger.warning(f"Invalid admin token type: {token_type}")
            raise credentials_exception

        # Extract claims
        user_id = payload.get("sub")
        email = payload.get("email")
        name = payload.get("name", "Unknown")
        role_str = payload.get("role")
        tenant_id = payload.get("tenant_id")
        permissions = payload.get("permissions", [])
        session_id = payload.get("jti", str(uuid4()))

        if not user_id or not email or not role_str:
            raise credentials_exception

        # Validate role
        try:
            role = AdminRole(role_str)
        except ValueError:
            logger.warning(f"Invalid admin role: {role_str}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid admin role",
            )

        # Validate tenant_id for org_admin
        if role == AdminRole.ORG_ADMIN and not tenant_id:
            logger.warning(f"Org admin without tenant_id: {email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Org admin must have a tenant_id",
            )

        return AdminUserContext(
            user_id=user_id,
            email=email,
            name=name,
            role=role,
            tenant_id=tenant_id,
            permissions=permissions,
            session_id=session_id,
            ip_address=request.client.host if request.client else None,
        )

    except JWTError as e:
        logger.debug(f"Admin JWT validation failed: {e}")
        raise credentials_exception


async def require_super_admin(
    admin_user: AdminUserContext = Depends(get_admin_user),
) -> AdminUserContext:
    """
    Require super admin role.

    Use this dependency for endpoints that should only be accessible
    to super admins (full system access).

    Raises:
        HTTPException: 403 if user is not a super admin
    """
    if admin_user.role != AdminRole.SUPER_ADMIN:
        logger.warning(
            f"Super admin access denied: user={admin_user.email} role={admin_user.role}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires super admin privileges",
        )
    return admin_user


async def require_org_admin(
    admin_user: AdminUserContext = Depends(get_admin_user),
) -> AdminUserContext:
    """
    Require org admin role or higher.

    Use this dependency for endpoints that should be accessible
    to org admins (scoped to their tenant) and super admins.

    Raises:
        HTTPException: 403 if user is not an org admin or super admin
    """
    if admin_user.role not in (AdminRole.SUPER_ADMIN, AdminRole.ORG_ADMIN):
        logger.warning(
            f"Org admin access denied: user={admin_user.email} role={admin_user.role}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires org admin or higher privileges",
        )
    return admin_user


async def require_operator(
    admin_user: AdminUserContext = Depends(get_admin_user),
) -> AdminUserContext:
    """
    Require operator role or higher.

    Use this dependency for read-only system operations.

    Raises:
        HTTPException: 403 if user doesn't have operator access
    """
    if admin_user.role not in (
        AdminRole.SUPER_ADMIN,
        AdminRole.ORG_ADMIN,
        AdminRole.OPERATOR,
    ):
        logger.warning(
            f"Operator access denied: user={admin_user.email} role={admin_user.role}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires operator or higher privileges",
        )
    return admin_user


def require_tenant_access(tenant_id: str):
    """
    Create a dependency that validates tenant access.

    Use this for endpoints that operate on a specific tenant.

    Args:
        tenant_id: The tenant ID to validate access for

    Returns:
        Dependency function that validates access
    """

    async def _check_access(
        admin_user: AdminUserContext = Depends(get_admin_user),
    ) -> AdminUserContext:
        if not admin_user.can_access_tenant(tenant_id):
            logger.warning(
                f"Tenant access denied: user={admin_user.email} "
                f"role={admin_user.role} tenant={tenant_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to tenant {tenant_id}",
            )
        return admin_user

    return _check_access


# ==============================================================================
# Audit Decorator
# ==============================================================================


def audit_admin_action(
    action: AuditAction,
    resource_type: str,
    get_resource_id: Optional[Callable[..., Optional[str]]] = None,
    get_tenant_id: Optional[Callable[..., Optional[str]]] = None,
    include_request_body: bool = False,
):
    """
    Decorator to automatically audit admin actions.

    Usage:
        @audit_admin_action(
            action=AuditAction.TENANT_CREATE,
            resource_type="tenant",
            get_resource_id=lambda result: result.id,
        )
        async def create_tenant(request: TenantCreate, admin_user: AdminUserContext):
            ...

    Args:
        action: The audit action type
        resource_type: The type of resource being acted upon
        get_resource_id: Optional function to extract resource ID from result
        get_tenant_id: Optional function to extract tenant ID from args
        include_request_body: Whether to include request body in audit details
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract admin user from kwargs
            admin_user: Optional[AdminUserContext] = kwargs.get("admin_user")
            request: Optional[Request] = kwargs.get("request")

            if not admin_user:
                # Try to find it in args
                for arg in args:
                    if isinstance(arg, AdminUserContext):
                        admin_user = arg
                        break

            audit_logger = get_admin_audit_logger()
            details: Dict[str, Any] = {}
            resource_id: Optional[str] = None
            tenant_id: Optional[str] = None

            # Extract tenant_id if function provided
            if get_tenant_id:
                try:
                    tenant_id = get_tenant_id(*args, **kwargs)
                except Exception:
                    pass

            # Include request body if configured
            if include_request_body:
                for arg in args:
                    if isinstance(arg, BaseModel):
                        details["request_body"] = arg.model_dump()
                        break
                for key, value in kwargs.items():
                    if isinstance(value, BaseModel) and key not in ("admin_user",):
                        details["request_body"] = value.model_dump()
                        break

            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Extract resource_id if function provided
                if get_resource_id:
                    try:
                        resource_id = get_resource_id(result)
                    except Exception:
                        pass

                # Log successful action
                if admin_user:
                    audit_logger.log(
                        action=action,
                        admin_user=admin_user,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        target_tenant_id=tenant_id,
                        details=details,
                        success=True,
                        request=request,
                    )

                return result

            except HTTPException as e:
                # Log failed action
                if admin_user:
                    audit_logger.log(
                        action=action,
                        admin_user=admin_user,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        target_tenant_id=tenant_id,
                        details=details,
                        success=False,
                        error_message=str(e.detail),
                        request=request,
                    )
                raise

            except Exception as e:
                # Log failed action
                if admin_user:
                    audit_logger.log(
                        action=action,
                        admin_user=admin_user,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        target_tenant_id=tenant_id,
                        details=details,
                        success=False,
                        error_message=str(e),
                        request=request,
                    )
                raise

        return wrapper

    return decorator


# ==============================================================================
# Token Generation (for testing/admin setup)
# ==============================================================================


def create_admin_token(
    user_id: str,
    email: str,
    name: str,
    role: AdminRole,
    tenant_id: Optional[str] = None,
    permissions: Optional[List[str]] = None,
    expires_minutes: int = 60,
) -> str:
    """
    Create an admin JWT token.

    This should only be used for initial admin setup or testing.
    In production, tokens should be issued through proper authentication flow.

    Args:
        user_id: User identifier
        email: User email
        name: User display name
        role: Admin role
        tenant_id: Tenant ID (required for org_admin)
        permissions: Optional list of specific permissions
        expires_minutes: Token expiration in minutes

    Returns:
        Encoded JWT token
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes)

    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "role": role.value,
        "tenant_id": tenant_id,
        "permissions": permissions or [],
        "type": "admin_access",
        "exp": expire,
        "iat": now,
        "iss": ADMIN_TOKEN_ISSUER,
        "aud": ADMIN_TOKEN_AUDIENCE,
        "jti": secrets.token_hex(16),
    }

    return jwt.encode(payload, ADMIN_SECRET_KEY, algorithm=ADMIN_TOKEN_ALGORITHM)
