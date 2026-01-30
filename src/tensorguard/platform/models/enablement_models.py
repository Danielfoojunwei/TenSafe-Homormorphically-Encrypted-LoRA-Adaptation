"""Enablement/onboarding database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class OnboardingProgress(SQLModel, table=True):
    """Track tenant onboarding progress."""
    __tablename__ = "onboarding_progress"

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True, unique=True)
    current_step: str = Field(default="welcome")
    completed_steps_json: str = Field(default="[]")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata_json: Optional[str] = None


class FeatureFlag(SQLModel, table=True):
    """Feature flags for gradual rollout."""
    __tablename__ = "feature_flags"

    id: Optional[int] = Field(default=None, primary_key=True)
    flag_id: str = Field(index=True, unique=True)
    name: str
    description: Optional[str] = None
    is_enabled: bool = Field(default=False)
    tenant_ids_json: Optional[str] = None  # Specific tenants
    percentage: int = Field(default=0)  # Percentage rollout
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
