"""Identity management database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class IdentityEndpoint(SQLModel, table=True):
    """Identity service endpoint configuration."""
    __tablename__ = "identity_endpoints"

    id: Optional[int] = Field(default=None, primary_key=True)
    endpoint_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    url: str
    auth_type: str = Field(default="mtls")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class IdentityCertificate(SQLModel, table=True):
    """Certificate store for mTLS and code signing."""
    __tablename__ = "identity_certificates"

    id: Optional[int] = Field(default=None, primary_key=True)
    cert_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    subject: str
    issuer: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    fingerprint_sha256: str = Field(index=True)
    cert_pem: str
    cert_type: str = Field(default="leaf")  # leaf, intermediate, root
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_revoked: bool = Field(default=False)


class IdentityPolicy(SQLModel, table=True):
    """Access policy for identity verification."""
    __tablename__ = "identity_policies"

    id: Optional[int] = Field(default=None, primary_key=True)
    policy_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    description: Optional[str] = None
    rules_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class IdentityRenewalJob(SQLModel, table=True):
    """Certificate renewal job tracking."""
    __tablename__ = "identity_renewal_jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True, unique=True)
    cert_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class IdentityAuditLog(SQLModel, table=True):
    """Identity-specific audit log."""
    __tablename__ = "identity_audit_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    entry_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    action: str
    subject_id: Optional[str] = None
    resource_type: str
    resource_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details_json: Optional[str] = None


class IdentityAgent(SQLModel, table=True):
    """Registered identity agent (workload/service)."""
    __tablename__ = "identity_agents"

    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    agent_type: str = Field(default="workload")
    spiffe_id: Optional[str] = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen_at: Optional[datetime] = None
    is_active: bool = Field(default=True)
    metadata_json: Optional[str] = None
