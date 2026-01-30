"""Federated MoE (Mixture of Experts) database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class FedMoEExpert(SQLModel, table=True):
    """Federated MoE expert model."""
    __tablename__ = "fedmoe_experts"

    id: Optional[int] = Field(default=None, primary_key=True)
    expert_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    description: Optional[str] = None
    domain: str  # e.g., "medical", "legal", "finance"
    model_ref: str
    adapter_artifact_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    metrics_json: Optional[str] = None


class SkillEvidence(SQLModel, table=True):
    """Evidence of expert skill/capability."""
    __tablename__ = "skill_evidence"

    id: Optional[int] = Field(default=None, primary_key=True)
    evidence_id: str = Field(index=True, unique=True)
    expert_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    skill_type: str
    benchmark_name: str
    score: float
    evaluation_date: datetime = Field(default_factory=datetime.utcnow)
    details_json: Optional[str] = None
