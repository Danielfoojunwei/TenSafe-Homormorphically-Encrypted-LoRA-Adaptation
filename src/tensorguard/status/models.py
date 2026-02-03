"""Status Page Models for TenSafe.

Data models for component status, incidents, maintenance windows, and uptime metrics.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, EmailStr
from sqlmodel import Field as SQLField, SQLModel


class ComponentState(str, Enum):
    """Component operational states."""

    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    MAJOR_OUTAGE = "major_outage"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class IncidentSeverity(str, Enum):
    """Incident severity levels (P1-P4)."""

    P1_CRITICAL = "p1_critical"  # Complete service outage
    P2_MAJOR = "p2_major"  # Major functionality impaired
    P3_MINOR = "p3_minor"  # Minor functionality impaired
    P4_LOW = "p4_low"  # Cosmetic or non-urgent issues


class IncidentStatus(str, Enum):
    """Incident lifecycle status."""

    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    POSTMORTEM = "postmortem"


class MaintenanceStatus(str, Enum):
    """Maintenance window status."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# ==============================================================================
# Pydantic API Models
# ==============================================================================


class ComponentStatus(BaseModel):
    """Status of a single system component."""

    id: str = Field(..., description="Unique component identifier")
    name: str = Field(..., description="Human-readable component name")
    description: Optional[str] = Field(None, description="Component description")
    state: ComponentState = Field(default=ComponentState.OPERATIONAL)
    last_check: datetime = Field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = Field(None, description="Last response time in ms")
    uptime_percentage: Optional[float] = Field(None, description="Uptime percentage (0-100)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class IncidentUpdate(BaseModel):
    """Update to an incident."""

    id: str = Field(..., description="Update ID")
    incident_id: str = Field(..., description="Parent incident ID")
    status: IncidentStatus = Field(..., description="Status at time of update")
    message: str = Field(..., description="Update message")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="User who created update")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Incident(BaseModel):
    """Incident record."""

    id: str = Field(..., description="Unique incident identifier")
    title: str = Field(..., description="Incident title")
    severity: IncidentSeverity = Field(..., description="Severity level")
    status: IncidentStatus = Field(default=IncidentStatus.INVESTIGATING)
    affected_components: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = Field(None)
    updates: List[IncidentUpdate] = Field(default_factory=list)
    postmortem_url: Optional[str] = Field(None, description="Link to post-mortem")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    @property
    def is_active(self) -> bool:
        """Check if incident is still active."""
        return self.status not in [IncidentStatus.RESOLVED, IncidentStatus.POSTMORTEM]

    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate incident duration in minutes."""
        end_time = self.resolved_at or datetime.utcnow()
        delta = end_time - self.started_at
        return delta.total_seconds() / 60


class MaintenanceWindow(BaseModel):
    """Scheduled maintenance window."""

    id: str = Field(..., description="Unique maintenance ID")
    title: str = Field(..., description="Maintenance title")
    description: str = Field(..., description="Detailed description")
    affected_components: List[str] = Field(default_factory=list)
    scheduled_start: datetime = Field(..., description="Scheduled start time")
    scheduled_end: datetime = Field(..., description="Scheduled end time")
    actual_start: Optional[datetime] = Field(None)
    actual_end: Optional[datetime] = Field(None)
    status: MaintenanceStatus = Field(default=MaintenanceStatus.SCHEDULED)
    impact: str = Field(default="minimal", description="Expected impact level")
    created_by: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    @property
    def is_active(self) -> bool:
        """Check if maintenance is currently active."""
        return self.status == MaintenanceStatus.IN_PROGRESS

    @property
    def is_upcoming(self) -> bool:
        """Check if maintenance is upcoming."""
        return (
            self.status == MaintenanceStatus.SCHEDULED
            and self.scheduled_start > datetime.utcnow()
        )


class UptimeMetrics(BaseModel):
    """Uptime metrics for a component or system."""

    component_id: str = Field(..., description="Component identifier or 'system'")
    period: str = Field(..., description="Time period (daily, monthly, yearly)")

    # Availability metrics
    uptime_percentage: float = Field(..., ge=0, le=100, description="Uptime percentage")
    total_minutes: int = Field(..., description="Total minutes in period")
    downtime_minutes: float = Field(..., description="Total downtime minutes")

    # Incident metrics
    incident_count: int = Field(default=0, description="Number of incidents")
    mttr_minutes: Optional[float] = Field(None, description="Mean time to recovery")
    mtbf_hours: Optional[float] = Field(None, description="Mean time between failures")

    # Response time metrics
    avg_response_time_ms: Optional[float] = Field(None)
    p50_response_time_ms: Optional[float] = Field(None)
    p95_response_time_ms: Optional[float] = Field(None)
    p99_response_time_ms: Optional[float] = Field(None)

    # SLA tracking
    sla_target: float = Field(default=99.9, description="SLA target percentage")
    sla_met: bool = Field(default=True, description="Whether SLA was met")
    sla_breach_minutes: float = Field(default=0, description="Minutes over SLA limit")

    period_start: datetime = Field(...)
    period_end: datetime = Field(...)
    calculated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class StatusSubscriber(BaseModel):
    """Status update subscriber."""

    id: str = Field(..., description="Subscriber ID")
    email: EmailStr = Field(..., description="Subscriber email")
    components: List[str] = Field(default_factory=list, description="Subscribed components")
    notify_on: List[str] = Field(
        default=["incidents", "maintenance"],
        description="Event types to notify on"
    )
    verified: bool = Field(default=False)
    verification_token: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    unsubscribe_token: Optional[str] = Field(None)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SystemStatus(BaseModel):
    """Overall system status summary."""

    status: ComponentState = Field(..., description="Overall system status")
    status_message: str = Field(..., description="Human-readable status message")
    components: List[ComponentStatus] = Field(default_factory=list)
    active_incidents: List[Incident] = Field(default_factory=list)
    active_maintenance: List[MaintenanceWindow] = Field(default_factory=list)
    upcoming_maintenance: List[MaintenanceWindow] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ==============================================================================
# SQLModel Database Models
# ==============================================================================


class ComponentStatusDB(SQLModel, table=True):
    """Database model for component status history."""

    __tablename__ = "status_component_history"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    component_id: str = SQLField(index=True)
    state: str = SQLField(index=True)
    response_time_ms: Optional[float] = None
    check_timestamp: datetime = SQLField(default_factory=datetime.utcnow, index=True)
    metadata_json: Optional[str] = None


class IncidentDB(SQLModel, table=True):
    """Database model for incidents."""

    __tablename__ = "status_incidents"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    incident_id: str = SQLField(index=True, unique=True)
    title: str
    severity: str
    status: str = SQLField(index=True)
    affected_components_json: str = SQLField(default="[]")
    started_at: datetime = SQLField(default_factory=datetime.utcnow, index=True)
    resolved_at: Optional[datetime] = None
    postmortem_url: Optional[str] = None
    metadata_json: Optional[str] = None
    created_by: Optional[str] = None


class IncidentUpdateDB(SQLModel, table=True):
    """Database model for incident updates."""

    __tablename__ = "status_incident_updates"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    update_id: str = SQLField(index=True, unique=True)
    incident_id: str = SQLField(index=True)
    status: str
    message: str
    created_at: datetime = SQLField(default_factory=datetime.utcnow, index=True)
    created_by: Optional[str] = None


class MaintenanceWindowDB(SQLModel, table=True):
    """Database model for maintenance windows."""

    __tablename__ = "status_maintenance_windows"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    maintenance_id: str = SQLField(index=True, unique=True)
    title: str
    description: str
    affected_components_json: str = SQLField(default="[]")
    scheduled_start: datetime = SQLField(index=True)
    scheduled_end: datetime
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    status: str = SQLField(default="scheduled", index=True)
    impact: str = SQLField(default="minimal")
    created_by: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


class StatusSubscriberDB(SQLModel, table=True):
    """Database model for status subscribers."""

    __tablename__ = "status_subscribers"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    subscriber_id: str = SQLField(index=True, unique=True)
    email: str = SQLField(index=True)
    components_json: str = SQLField(default="[]")
    notify_on_json: str = SQLField(default='["incidents", "maintenance"]')
    verified: bool = SQLField(default=False)
    verification_token: Optional[str] = None
    unsubscribe_token: Optional[str] = None
    created_at: datetime = SQLField(default_factory=datetime.utcnow)


class UptimeRecordDB(SQLModel, table=True):
    """Database model for uptime records (aggregated)."""

    __tablename__ = "status_uptime_records"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    component_id: str = SQLField(index=True)
    period: str = SQLField(index=True)  # daily, monthly, yearly
    period_start: datetime = SQLField(index=True)
    period_end: datetime
    uptime_percentage: float
    total_minutes: int
    downtime_minutes: float
    incident_count: int = SQLField(default=0)
    mttr_minutes: Optional[float] = None
    avg_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None
    p99_response_time_ms: Optional[float] = None
    calculated_at: datetime = SQLField(default_factory=datetime.utcnow)


# ==============================================================================
# API Request/Response Models
# ==============================================================================


class CreateIncidentRequest(BaseModel):
    """Request to create a new incident."""

    title: str = Field(..., min_length=5, max_length=200)
    severity: IncidentSeverity
    affected_components: List[str] = Field(default_factory=list)
    initial_message: str = Field(..., min_length=10)


class UpdateIncidentRequest(BaseModel):
    """Request to update an incident."""

    status: Optional[IncidentStatus] = None
    message: str = Field(..., min_length=10)


class CreateMaintenanceRequest(BaseModel):
    """Request to schedule maintenance."""

    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10)
    affected_components: List[str] = Field(default_factory=list)
    scheduled_start: datetime
    scheduled_end: datetime
    impact: str = Field(default="minimal")


class SubscribeRequest(BaseModel):
    """Request to subscribe to status updates."""

    email: EmailStr
    components: List[str] = Field(default_factory=list)
    notify_on: List[str] = Field(default=["incidents", "maintenance"])


class StatusSummaryResponse(BaseModel):
    """Status summary response for API."""

    status: ComponentState
    message: str
    components: Dict[str, ComponentState]
    active_incidents_count: int
    active_maintenance_count: int
    last_incident: Optional[datetime]
    uptime_30d: Optional[float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
