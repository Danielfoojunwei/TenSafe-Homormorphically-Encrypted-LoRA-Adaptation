"""
TensorGuard Platform Dependencies.

Systemic enforcement of tenant isolation and authentication context.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from .auth import get_current_user
from .models.core import User


async def get_current_tenant_id(
    current_user: User = Depends(get_current_user),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID")
) -> str:
    """
    Mandatory dependency to derive the current tenant context.
    
    In production, this MUST match the user's authorized tenant.
    If X-Tenant-ID is provided, it is validated against the user's scope.
    """
    # 1. Start with the tenant associated with the authenticated principal
    authorized_tenant = current_user.tenant_id

    # 2. If an explicit tenant header is used (e.g., for multi-tenant admins),
    # verify that the user has access to it.
    if x_tenant_id and x_tenant_id != authorized_tenant:
        # Note: In a full implementation, we'd check if current_user is a 'SuperAdmin'
        # For now, we enforce strict isolation.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized tenant access"
        )

    return authorized_tenant

async def require_tenant_context(
    tenant_id: str = Depends(get_current_tenant_id)
) -> str:
    """
    Dependency to ensure a valid tenant context is present and authorized.
    Used for endpoints that require multi-tenant scoping.
    """
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing tenant context"
        )
    return tenant_id
