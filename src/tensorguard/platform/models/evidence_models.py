"""Evidence/compliance database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class EvidenceRecord(SQLModel, table=True):
    """Compliance evidence record."""
    __tablename__ = "evidence_records"

    id: Optional[int] = Field(default=None, primary_key=True)
    record_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    evidence_type: str = Field(index=True)
    content_hash: str
    content_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    prev_hash: Optional[str] = None
    record_hash: str


class ComplianceReport(SQLModel, table=True):
    """Generated compliance report."""
    __tablename__ = "compliance_reports"

    id: Optional[int] = Field(default=None, primary_key=True)
    report_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    report_type: str
    period_start: datetime
    period_end: datetime
    status: str = Field(default="draft")
    content_json: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finalized_at: Optional[datetime] = None
