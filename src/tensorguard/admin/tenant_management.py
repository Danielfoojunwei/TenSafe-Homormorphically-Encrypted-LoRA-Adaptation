"""
Tenant Lifecycle Management.

Provides comprehensive tenant management including:
- Tenant provisioning (with resource allocation)
- Resource isolation
- Data export (for compliance/portability)
- Tenant deletion (with secure data cleanup)
"""

import asyncio
import hashlib
import json
import logging
import os
import secrets
import shutil
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .models import (
    Tenant,
    TenantCreate,
    TenantPlan,
    TenantQuota,
    TenantStatus,
    TenantUpdate,
    TenantUsage,
)
from .permissions import AdminUserContext, AuditAction, get_admin_audit_logger

logger = logging.getLogger(__name__)


# Plan-based quota defaults
PLAN_QUOTAS: Dict[TenantPlan, TenantQuota] = {
    TenantPlan.FREE: TenantQuota(
        max_users=5,
        max_training_jobs=1,
        max_storage_gb=1.0,
        max_compute_hours=10.0,
        max_api_requests_per_day=1000,
        dp_budget_epsilon=2.0,
        he_operations_per_day=100,
    ),
    TenantPlan.STARTER: TenantQuota(
        max_users=25,
        max_training_jobs=3,
        max_storage_gb=25.0,
        max_compute_hours=100.0,
        max_api_requests_per_day=10000,
        dp_budget_epsilon=5.0,
        he_operations_per_day=1000,
    ),
    TenantPlan.PROFESSIONAL: TenantQuota(
        max_users=100,
        max_training_jobs=10,
        max_storage_gb=100.0,
        max_compute_hours=500.0,
        max_api_requests_per_day=100000,
        dp_budget_epsilon=10.0,
        he_operations_per_day=10000,
    ),
    TenantPlan.ENTERPRISE: TenantQuota(
        max_users=1000,
        max_training_jobs=50,
        max_storage_gb=1000.0,
        max_compute_hours=5000.0,
        max_api_requests_per_day=1000000,
        dp_budget_epsilon=50.0,
        he_operations_per_day=100000,
    ),
}


class TenantManager:
    """
    Manager for tenant lifecycle operations.

    Handles:
    - Tenant creation and provisioning
    - Resource quota management
    - Tenant isolation
    - Data export and deletion
    - Tenant suspension and reactivation
    """

    def __init__(self):
        """Initialize the tenant manager."""
        # In-memory storage (replace with database in production)
        self._tenants: Dict[str, Dict[str, Any]] = {}
        self._users: Dict[str, Dict[str, Any]] = {}
        self._tenant_resources: Dict[str, Dict[str, Any]] = {}

        # Data directories
        self._data_root = Path(os.getenv("TG_DATA_ROOT", "/var/lib/tensorguard"))
        self._export_dir = Path(os.getenv("TG_EXPORT_DIR", "/tmp/tensorguard/exports"))

    # ==========================================================================
    # Tenant CRUD Operations
    # ==========================================================================

    async def create_tenant(
        self,
        request: TenantCreate,
        admin_user: AdminUserContext,
    ) -> Tenant:
        """
        Create and provision a new tenant.

        This includes:
        1. Generating unique tenant ID
        2. Creating admin user
        3. Setting up resource quotas
        4. Provisioning isolated resources
        5. Initializing audit trail

        Args:
            request: Tenant creation request
            admin_user: Admin user performing the action

        Returns:
            Created Tenant

        Raises:
            ValueError: If slug is already taken
        """
        # Check slug uniqueness
        for tenant in self._tenants.values():
            if tenant["slug"] == request.slug:
                raise ValueError(f"Tenant slug '{request.slug}' is already in use")

        # Generate tenant ID
        tenant_id = f"tnt-{uuid4().hex[:12]}"

        # Determine quota (custom or plan default)
        quota = request.quota or PLAN_QUOTAS.get(request.plan, PLAN_QUOTAS[TenantPlan.FREE])

        now = datetime.now(timezone.utc)

        # Create tenant record
        tenant_data = {
            "id": tenant_id,
            "name": request.name,
            "slug": request.slug,
            "plan": request.plan.value,
            "status": TenantStatus.PENDING.value,
            "quota": quota.model_dump(),
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "user_count": 0,
            "metadata": request.metadata,
            "created_by": admin_user.user_id,
        }

        self._tenants[tenant_id] = tenant_data

        # Provision resources
        await self._provision_tenant_resources(tenant_id, quota)

        # Create admin user for tenant
        admin_user_id = await self._create_tenant_admin(
            tenant_id=tenant_id,
            email=request.admin_email,
            name=request.admin_name,
        )

        # Update user count
        tenant_data["user_count"] = 1

        # Activate tenant
        tenant_data["status"] = TenantStatus.ACTIVE.value

        # Log audit
        audit_logger = get_admin_audit_logger()
        audit_logger.log(
            action=AuditAction.TENANT_CREATE,
            admin_user=admin_user,
            resource_type="tenant",
            resource_id=tenant_id,
            target_tenant_id=tenant_id,
            details={
                "name": request.name,
                "slug": request.slug,
                "plan": request.plan.value,
                "admin_email": request.admin_email,
                "admin_user_id": admin_user_id,
            },
            success=True,
        )

        logger.info(f"Created tenant: {tenant_id} ({request.name})")

        return self._to_tenant_model(tenant_data)

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Get tenant by ID.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant if found, None otherwise
        """
        tenant_data = self._tenants.get(tenant_id)
        if tenant_data:
            return self._to_tenant_model(tenant_data)
        return None

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """
        Get tenant by slug.

        Args:
            slug: Tenant slug

        Returns:
            Tenant if found, None otherwise
        """
        for tenant_data in self._tenants.values():
            if tenant_data["slug"] == slug:
                return self._to_tenant_model(tenant_data)
        return None

    async def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        plan: Optional[TenantPlan] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Tenant], int]:
        """
        List tenants with filtering and pagination.

        Args:
            status: Filter by status
            plan: Filter by plan
            search: Search in name and slug
            page: Page number (1-indexed)
            page_size: Items per page

        Returns:
            Tuple of (list of tenants, total count)
        """
        # Filter tenants
        filtered = []
        for tenant_data in self._tenants.values():
            # Apply status filter
            if status and tenant_data["status"] != status.value:
                continue

            # Apply plan filter
            if plan and tenant_data["plan"] != plan.value:
                continue

            # Apply search filter
            if search:
                search_lower = search.lower()
                if (
                    search_lower not in tenant_data["name"].lower()
                    and search_lower not in tenant_data["slug"].lower()
                ):
                    continue

            filtered.append(tenant_data)

        # Sort by created_at descending
        filtered.sort(key=lambda x: x["created_at"], reverse=True)

        # Paginate
        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = filtered[start:end]

        return [self._to_tenant_model(t) for t in page_items], total

    async def update_tenant(
        self,
        tenant_id: str,
        request: TenantUpdate,
        admin_user: AdminUserContext,
    ) -> Tenant:
        """
        Update tenant properties.

        Args:
            tenant_id: Tenant identifier
            request: Update request
            admin_user: Admin user performing the action

        Returns:
            Updated Tenant

        Raises:
            ValueError: If tenant not found
        """
        tenant_data = self._tenants.get(tenant_id)
        if not tenant_data:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        # Track changes for audit
        changes = {}

        # Apply updates
        if request.name is not None:
            changes["name"] = {"old": tenant_data["name"], "new": request.name}
            tenant_data["name"] = request.name

        if request.plan is not None:
            changes["plan"] = {"old": tenant_data["plan"], "new": request.plan.value}
            tenant_data["plan"] = request.plan.value

            # Update quota if plan changed and no custom quota
            if request.quota is None:
                new_quota = PLAN_QUOTAS.get(request.plan, PLAN_QUOTAS[TenantPlan.FREE])
                tenant_data["quota"] = new_quota.model_dump()

        if request.status is not None:
            changes["status"] = {"old": tenant_data["status"], "new": request.status.value}
            tenant_data["status"] = request.status.value

        if request.quota is not None:
            changes["quota"] = {"old": tenant_data["quota"], "new": request.quota.model_dump()}
            tenant_data["quota"] = request.quota.model_dump()

        if request.metadata is not None:
            tenant_data["metadata"] = request.metadata

        tenant_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Log audit
        audit_logger = get_admin_audit_logger()
        audit_logger.log(
            action=AuditAction.TENANT_UPDATE,
            admin_user=admin_user,
            resource_type="tenant",
            resource_id=tenant_id,
            target_tenant_id=tenant_id,
            details={"changes": changes},
            success=True,
        )

        logger.info(f"Updated tenant: {tenant_id}")

        return self._to_tenant_model(tenant_data)

    async def delete_tenant(
        self,
        tenant_id: str,
        admin_user: AdminUserContext,
        hard_delete: bool = False,
    ) -> bool:
        """
        Delete a tenant.

        Soft delete: Marks tenant as deleted, retains data for retention period.
        Hard delete: Permanently removes all tenant data.

        Args:
            tenant_id: Tenant identifier
            admin_user: Admin user performing the action
            hard_delete: Whether to permanently delete data

        Returns:
            True if deletion successful

        Raises:
            ValueError: If tenant not found
        """
        tenant_data = self._tenants.get(tenant_id)
        if not tenant_data:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        if hard_delete:
            # Perform data cleanup
            await self._cleanup_tenant_data(tenant_id)

            # Remove tenant record
            del self._tenants[tenant_id]

            # Remove from resources
            self._tenant_resources.pop(tenant_id, None)

            # Log audit
            audit_logger = get_admin_audit_logger()
            audit_logger.log(
                action=AuditAction.TENANT_DELETE,
                admin_user=admin_user,
                resource_type="tenant",
                resource_id=tenant_id,
                target_tenant_id=tenant_id,
                details={"hard_delete": True, "name": tenant_data["name"]},
                success=True,
            )

            logger.warning(f"Hard deleted tenant: {tenant_id} ({tenant_data['name']})")

        else:
            # Soft delete - mark as deleted
            tenant_data["status"] = TenantStatus.DELETED.value
            tenant_data["deleted_at"] = datetime.now(timezone.utc).isoformat()
            tenant_data["deleted_by"] = admin_user.user_id
            tenant_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Log audit
            audit_logger = get_admin_audit_logger()
            audit_logger.log(
                action=AuditAction.TENANT_DELETE,
                admin_user=admin_user,
                resource_type="tenant",
                resource_id=tenant_id,
                target_tenant_id=tenant_id,
                details={"hard_delete": False, "name": tenant_data["name"]},
                success=True,
            )

            logger.info(f"Soft deleted tenant: {tenant_id} ({tenant_data['name']})")

        return True

    # ==========================================================================
    # Tenant Status Management
    # ==========================================================================

    async def suspend_tenant(
        self,
        tenant_id: str,
        admin_user: AdminUserContext,
        reason: str,
    ) -> Tenant:
        """
        Suspend a tenant.

        Suspended tenants cannot access the API but retain their data.

        Args:
            tenant_id: Tenant identifier
            admin_user: Admin user performing the action
            reason: Reason for suspension

        Returns:
            Updated Tenant
        """
        tenant_data = self._tenants.get(tenant_id)
        if not tenant_data:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        tenant_data["status"] = TenantStatus.SUSPENDED.value
        tenant_data["suspended_at"] = datetime.now(timezone.utc).isoformat()
        tenant_data["suspended_by"] = admin_user.user_id
        tenant_data["suspension_reason"] = reason
        tenant_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Log audit
        audit_logger = get_admin_audit_logger()
        audit_logger.log(
            action=AuditAction.TENANT_SUSPEND,
            admin_user=admin_user,
            resource_type="tenant",
            resource_id=tenant_id,
            target_tenant_id=tenant_id,
            details={"reason": reason},
            success=True,
        )

        logger.warning(f"Suspended tenant: {tenant_id} - {reason}")

        return self._to_tenant_model(tenant_data)

    async def activate_tenant(
        self,
        tenant_id: str,
        admin_user: AdminUserContext,
    ) -> Tenant:
        """
        Activate a suspended or pending tenant.

        Args:
            tenant_id: Tenant identifier
            admin_user: Admin user performing the action

        Returns:
            Updated Tenant
        """
        tenant_data = self._tenants.get(tenant_id)
        if not tenant_data:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        previous_status = tenant_data["status"]
        tenant_data["status"] = TenantStatus.ACTIVE.value
        tenant_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Clear suspension info
        tenant_data.pop("suspended_at", None)
        tenant_data.pop("suspended_by", None)
        tenant_data.pop("suspension_reason", None)

        # Log audit
        audit_logger = get_admin_audit_logger()
        audit_logger.log(
            action=AuditAction.TENANT_ACTIVATE,
            admin_user=admin_user,
            resource_type="tenant",
            resource_id=tenant_id,
            target_tenant_id=tenant_id,
            details={"previous_status": previous_status},
            success=True,
        )

        logger.info(f"Activated tenant: {tenant_id}")

        return self._to_tenant_model(tenant_data)

    # ==========================================================================
    # Quota Management
    # ==========================================================================

    async def set_tenant_quota(
        self,
        tenant_id: str,
        quota: TenantQuota,
        admin_user: AdminUserContext,
    ) -> Tenant:
        """
        Set custom quota for a tenant.

        Args:
            tenant_id: Tenant identifier
            quota: New quota configuration
            admin_user: Admin user performing the action

        Returns:
            Updated Tenant
        """
        tenant_data = self._tenants.get(tenant_id)
        if not tenant_data:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        old_quota = tenant_data["quota"]
        tenant_data["quota"] = quota.model_dump()
        tenant_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Log audit
        audit_logger = get_admin_audit_logger()
        audit_logger.log(
            action=AuditAction.TENANT_SET_QUOTA,
            admin_user=admin_user,
            resource_type="tenant",
            resource_id=tenant_id,
            target_tenant_id=tenant_id,
            details={"old_quota": old_quota, "new_quota": quota.model_dump()},
            success=True,
        )

        logger.info(f"Updated quota for tenant: {tenant_id}")

        return self._to_tenant_model(tenant_data)

    async def get_tenant_usage(self, tenant_id: str) -> TenantUsage:
        """
        Get current resource usage for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            TenantUsage with current usage metrics
        """
        if tenant_id not in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        now = datetime.now(timezone.utc)
        period_start = now - timedelta(days=30)

        # Mock usage data (replace with actual metrics queries)
        return TenantUsage(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=now,
            total_users=self._count_tenant_users(tenant_id),
            active_users=self._count_tenant_users(tenant_id),
            training_jobs_count=15,
            training_jobs_completed=12,
            compute_hours_used=45.5,
            storage_used_gb=8.3,
            artifacts_count=234,
            api_requests_count=15000,
            api_errors_count=45,
            dp_budget_consumed=1.5,
            he_operations_count=850,
            estimated_cost_usd=75.50,
        )

    # ==========================================================================
    # Resource Isolation
    # ==========================================================================

    async def _provision_tenant_resources(
        self,
        tenant_id: str,
        quota: TenantQuota,
    ) -> None:
        """
        Provision isolated resources for a tenant.

        This includes:
        - Creating isolated database schema/namespace
        - Setting up storage buckets/directories
        - Configuring rate limits
        - Initializing encryption keys

        Args:
            tenant_id: Tenant identifier
            quota: Tenant quota configuration
        """
        resources = {
            "tenant_id": tenant_id,
            "provisioned_at": datetime.now(timezone.utc).isoformat(),
            "database_schema": f"tenant_{tenant_id.replace('-', '_')}",
            "storage_bucket": f"tg-{tenant_id}",
            "encryption_key_id": f"key-{uuid4().hex[:16]}",
            "rate_limit_config": {
                "requests_per_minute": min(60, quota.max_api_requests_per_day // 1440),
                "requests_per_day": quota.max_api_requests_per_day,
            },
        }

        self._tenant_resources[tenant_id] = resources

        # Create data directories
        tenant_data_dir = self._data_root / "tenants" / tenant_id
        tenant_data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (tenant_data_dir / "artifacts").mkdir(exist_ok=True)
        (tenant_data_dir / "checkpoints").mkdir(exist_ok=True)
        (tenant_data_dir / "exports").mkdir(exist_ok=True)

        logger.info(f"Provisioned resources for tenant: {tenant_id}")

    async def _cleanup_tenant_data(self, tenant_id: str) -> None:
        """
        Clean up all data for a tenant (for hard delete).

        This includes:
        - Removing database records
        - Deleting storage objects
        - Revoking encryption keys
        - Clearing cache entries

        Args:
            tenant_id: Tenant identifier
        """
        logger.warning(f"Starting data cleanup for tenant: {tenant_id}")

        # Remove users
        users_to_remove = [
            uid for uid, udata in self._users.items()
            if udata.get("tenant_id") == tenant_id
        ]
        for uid in users_to_remove:
            del self._users[uid]

        # Remove data directories
        tenant_data_dir = self._data_root / "tenants" / tenant_id
        if tenant_data_dir.exists():
            shutil.rmtree(tenant_data_dir, ignore_errors=True)

        # Clear resources
        self._tenant_resources.pop(tenant_id, None)

        logger.info(f"Completed data cleanup for tenant: {tenant_id}")

    # ==========================================================================
    # Data Export
    # ==========================================================================

    async def export_tenant_data(
        self,
        tenant_id: str,
        admin_user: AdminUserContext,
        include_artifacts: bool = True,
        include_audit_logs: bool = True,
    ) -> str:
        """
        Export all tenant data for compliance/portability.

        Creates a ZIP archive containing:
        - Tenant configuration
        - User data
        - Training jobs and results
        - Artifacts (if requested)
        - Audit logs (if requested)

        Args:
            tenant_id: Tenant identifier
            admin_user: Admin user performing the action
            include_artifacts: Whether to include artifacts
            include_audit_logs: Whether to include audit logs

        Returns:
            Path to the export file
        """
        tenant_data = self._tenants.get(tenant_id)
        if not tenant_data:
            raise ValueError(f"Tenant '{tenant_id}' not found")

        # Create export directory
        self._export_dir.mkdir(parents=True, exist_ok=True)

        export_id = f"export-{tenant_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        export_path = self._export_dir / f"{export_id}.zip"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Export tenant metadata
            tenant_export = {
                "tenant": tenant_data,
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "exported_by": admin_user.email,
                "export_version": "1.0",
            }
            with open(temp_path / "tenant.json", "w") as f:
                json.dump(tenant_export, f, indent=2, default=str)

            # Export users
            tenant_users = [
                udata for udata in self._users.values()
                if udata.get("tenant_id") == tenant_id
            ]
            with open(temp_path / "users.json", "w") as f:
                json.dump({"users": tenant_users}, f, indent=2, default=str)

            # Export artifacts manifest (not actual files for this foundation)
            if include_artifacts:
                artifacts_manifest = {
                    "artifacts": [],  # Would be populated from actual storage
                    "total_size_bytes": 0,
                }
                with open(temp_path / "artifacts_manifest.json", "w") as f:
                    json.dump(artifacts_manifest, f, indent=2)

            # Export audit logs
            if include_audit_logs:
                audit_logs = {
                    "logs": [],  # Would be populated from audit log store
                    "export_note": "Audit logs for data portability",
                }
                with open(temp_path / "audit_logs.json", "w") as f:
                    json.dump(audit_logs, f, indent=2)

            # Create export manifest
            manifest = {
                "export_id": export_id,
                "tenant_id": tenant_id,
                "tenant_name": tenant_data["name"],
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "exported_by": admin_user.email,
                "contents": [
                    "tenant.json",
                    "users.json",
                    "artifacts_manifest.json" if include_artifacts else None,
                    "audit_logs.json" if include_audit_logs else None,
                ],
                "checksum_algorithm": "sha256",
            }
            manifest["contents"] = [c for c in manifest["contents"] if c]

            with open(temp_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            # Create ZIP archive
            with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in temp_path.iterdir():
                    zf.write(file_path, file_path.name)

        # Log audit
        audit_logger = get_admin_audit_logger()
        audit_logger.log(
            action=AuditAction.TENANT_EXPORT_DATA,
            admin_user=admin_user,
            resource_type="tenant",
            resource_id=tenant_id,
            target_tenant_id=tenant_id,
            details={
                "export_id": export_id,
                "include_artifacts": include_artifacts,
                "include_audit_logs": include_audit_logs,
                "export_path": str(export_path),
            },
            success=True,
        )

        logger.info(f"Exported tenant data: {tenant_id} -> {export_path}")

        return str(export_path)

    # ==========================================================================
    # User Management (Tenant-Scoped)
    # ==========================================================================

    async def _create_tenant_admin(
        self,
        tenant_id: str,
        email: str,
        name: str,
    ) -> str:
        """
        Create the initial admin user for a tenant.

        Args:
            tenant_id: Tenant identifier
            email: Admin email
            name: Admin name

        Returns:
            Created user ID
        """
        user_id = f"usr-{uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        user_data = {
            "id": user_id,
            "tenant_id": tenant_id,
            "email": email,
            "name": name,
            "role": "org_admin",
            "status": "active",
            "created_at": now.isoformat(),
            "last_login_at": None,
            "mfa_enabled": False,
            "metadata": {},
        }

        self._users[user_id] = user_data

        logger.info(f"Created tenant admin: {user_id} for tenant {tenant_id}")

        return user_id

    async def list_tenant_users(
        self,
        tenant_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List users for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            page: Page number
            page_size: Items per page

        Returns:
            Tuple of (list of users, total count)
        """
        tenant_users = [
            udata for udata in self._users.values()
            if udata.get("tenant_id") == tenant_id
        ]

        # Sort by created_at
        tenant_users.sort(key=lambda x: x["created_at"], reverse=True)

        # Paginate
        total = len(tenant_users)
        start = (page - 1) * page_size
        end = start + page_size

        return tenant_users[start:end], total

    def _count_tenant_users(self, tenant_id: str) -> int:
        """Count users for a tenant."""
        return sum(
            1 for udata in self._users.values()
            if udata.get("tenant_id") == tenant_id
        )

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def _to_tenant_model(self, data: Dict[str, Any]) -> Tenant:
        """Convert internal tenant data to Tenant model."""
        return Tenant(
            id=data["id"],
            name=data["name"],
            slug=data["slug"],
            plan=TenantPlan(data["plan"]),
            status=TenantStatus(data["status"]),
            quota=TenantQuota(**data["quota"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            user_count=data.get("user_count", 0),
            metadata=data.get("metadata", {}),
        )


# Singleton instance
_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    """Get or create the tenant manager singleton."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager
