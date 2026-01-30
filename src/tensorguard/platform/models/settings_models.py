"""System settings database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class SystemSetting(SQLModel, table=True):
    """System-wide configuration settings."""
    __tablename__ = "system_settings"

    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(index=True, unique=True)
    value: str
    value_type: str = Field(default="string")  # string, int, bool, json
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class KMSKey(SQLModel, table=True):
    """KMS key reference."""
    __tablename__ = "kms_keys"

    id: Optional[int] = Field(default=None, primary_key=True)
    key_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    key_type: str  # kek, dek, signing
    kms_provider: str  # aws, gcp, azure, vault
    kms_key_ref: str  # Provider-specific key reference
    algorithm: str = Field(default="AES-256-GCM")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    rotated_at: Optional[datetime] = None
    is_active: bool = Field(default=True)


class KMSRotationLog(SQLModel, table=True):
    """KMS key rotation audit log."""
    __tablename__ = "kms_rotation_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    rotation_id: str = Field(index=True, unique=True)
    key_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    old_key_ref: str
    new_key_ref: str
    rotated_at: datetime = Field(default_factory=datetime.utcnow)
    reason: str = Field(default="scheduled")
    initiated_by: Optional[str] = None
