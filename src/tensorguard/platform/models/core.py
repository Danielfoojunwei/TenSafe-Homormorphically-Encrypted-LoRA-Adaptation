"""Core database models for TensorGuard Platform."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Tenant(SQLModel, table=True):
    """Tenant (organization) model."""
    __tablename__ = "tenants"

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True, unique=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    metadata_json: Optional[str] = Field(default=None)


class User(SQLModel, table=True):
    """User model."""
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    email: str = Field(index=True)
    name: str
    role: str = Field(default="user")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class Fleet(SQLModel, table=True):
    """Fleet (group of compute nodes) model."""
    __tablename__ = "fleets"

    id: Optional[int] = Field(default=None, primary_key=True)
    fleet_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    config_json: Optional[str] = Field(default=None)


class Job(SQLModel, table=True):
    """Training/inference job model."""
    __tablename__ = "jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    fleet_id: Optional[str] = Field(index=True)
    name: str
    status: str = Field(default="pending")
    job_type: str = Field(default="training")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config_json: Optional[str] = Field(default=None)
    result_json: Optional[str] = Field(default=None)


class AuditLog(SQLModel, table=True):
    """Audit log model for compliance tracking."""
    __tablename__ = "audit_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    entry_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    user_id: Optional[str] = Field(index=True)
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details_json: Optional[str] = Field(default=None)
    prev_hash: Optional[str] = None
    record_hash: Optional[str] = None
