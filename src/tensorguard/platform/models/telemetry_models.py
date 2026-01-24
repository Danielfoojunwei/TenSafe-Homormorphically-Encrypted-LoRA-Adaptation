"""
Telemetry Models for Production Continual Learning Control Plane

These models support real telemetry ingestion from edge agents, replacing
all simulated/mock data with persisted, queryable records.

All tables include:
- Multi-tenant isolation (tenant_id)
- Fleet scoping (fleet_id)
- Device attribution (device_id)
- Proper indexing for query patterns
- Timestamps for time-range queries
"""

from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Index, text
from datetime import datetime
from enum import Enum
import uuid


# =============================================================================
# Enums
# =============================================================================

class PipelineStage(str, Enum):
    """Canonical pipeline stages for telemetry tracking."""
    CAPTURE = "capture"
    EMBED = "embed"
    GATE = "gate"
    PEFT = "peft"
    SHIELD = "shield"
    SYNC = "sync"
    PULL = "pull"


class StageStatus(str, Enum):
    """Status values for pipeline stage events."""
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class EventSeverity(str, Enum):
    """Severity levels for forensics events."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ForensicsEventType(str, Enum):
    """Types of forensics events."""
    SAFETY_VIOLATION = "safety_violation"
    PRIVACY_BREACH = "privacy_breach"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MODEL_ANOMALY = "model_anomaly"
    CONSTRAINT_HIT = "constraint_hit"
    OPERATOR_INTERVENTION = "operator_intervention"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    COMPATIBILITY_FAILURE = "compatibility_failure"


# =============================================================================
# FleetDevice - Device Registry with Version Tracking
# =============================================================================

class FleetDevice(SQLModel, table=True):
    """
    Registry of devices in a fleet with version tracking.

    Enables compatibility checks for deployment rollouts by tracking
    agent/runtime/firmware versions and sensor configurations.
    """
    __tablename__ = "fleet_device"
    __table_args__ = (
        Index('ix_fleet_device_tenant_fleet', 'tenant_id', 'fleet_id'),
        Index('ix_fleet_device_fleet_lastseen', 'fleet_id', 'last_seen_at'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
        description="Internal UUID"
    )
    device_id: str = Field(
        index=True,
        unique=True,
        description="External device identifier (from agent)"
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True,
        description="Owning tenant"
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True,
        description="Fleet membership"
    )

    # Version tracking for compatibility management
    agent_version: Optional[str] = Field(
        default=None,
        description="TensorGuard agent version (semver)"
    )
    runtime_version: Optional[str] = Field(
        default=None,
        description="Runtime/framework version"
    )
    ros_distro: Optional[str] = Field(
        default=None,
        description="ROS distribution (humble, iron, etc.)"
    )
    firmware_version: Optional[str] = Field(
        default=None,
        description="Device firmware version"
    )
    sensor_manifest_hash: Optional[str] = Field(
        default=None,
        description="SHA256 hash of sensor configuration manifest"
    )

    # Status tracking
    last_seen_at: Optional[datetime] = Field(
        default=None,
        index=True,
        description="Last heartbeat/telemetry timestamp"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Device registration timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )


# =============================================================================
# TelemetryStageEvent - Pipeline Stage Telemetry
# =============================================================================

class TelemetryStageEvent(SQLModel, table=True):
    """
    Records pipeline stage execution events from edge agents.

    Replaces simulated telemetry with real persisted data for:
    - p50/p90/p99 latency computation
    - Error rate tracking
    - Safe mode determination
    """
    __tablename__ = "telemetry_stage_event"
    __table_args__ = (
        Index('ix_telemetry_stage_fleet_ts', 'fleet_id', 'ts'),
        Index('ix_telemetry_stage_fleet_stage_ts', 'fleet_id', 'stage', 'ts'),
        Index('ix_telemetry_stage_device_ts', 'device_id', 'ts'),
        Index('ix_telemetry_stage_tenant_ts', 'tenant_id', 'ts'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )
    device_id: str = Field(
        index=True,
        description="Device identifier (external)"
    )

    # Pipeline context
    run_id: Optional[str] = Field(
        default=None,
        index=True,
        description="Training/inference run identifier"
    )
    stage: str = Field(
        index=True,
        description="Pipeline stage (capture/embed/gate/peft/shield/sync/pull)"
    )
    status: str = Field(
        default=StageStatus.OK.value,
        description="Stage execution status (ok/degraded/error)"
    )

    # Metrics
    latency_ms: float = Field(
        description="Stage execution latency in milliseconds"
    )
    metadata_json: str = Field(
        default="{}",
        description="Stage-specific metadata (JSON)"
    )

    # Timestamp
    ts: datetime = Field(
        default_factory=datetime.utcnow,
        index=True,
        description="Event timestamp"
    )


# =============================================================================
# TelemetrySystemEvent - System Resource Telemetry
# =============================================================================

class TelemetrySystemEvent(SQLModel, table=True):
    """
    Records system resource utilization from edge agents.

    Provides infrastructure monitoring without simulation.
    """
    __tablename__ = "telemetry_system_event"
    __table_args__ = (
        Index('ix_telemetry_system_fleet_ts', 'fleet_id', 'ts'),
        Index('ix_telemetry_system_device_ts', 'device_id', 'ts'),
        Index('ix_telemetry_system_tenant_ts', 'tenant_id', 'ts'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )
    device_id: str = Field(
        index=True,
        description="Device identifier (external)"
    )

    # CPU/Memory metrics
    cpu_pct: float = Field(
        description="CPU utilization percentage (0-100)"
    )
    mem_pct: float = Field(
        description="Memory utilization percentage (0-100)"
    )

    # GPU metrics (optional - not all devices have GPUs)
    gpu_pct: Optional[float] = Field(
        default=None,
        description="GPU utilization percentage (0-100)"
    )
    temp_c: Optional[float] = Field(
        default=None,
        description="Temperature in Celsius"
    )

    # Network metrics (optional)
    bandwidth_up_bps: Optional[int] = Field(
        default=None,
        description="Upload bandwidth in bits per second"
    )
    bandwidth_down_bps: Optional[int] = Field(
        default=None,
        description="Download bandwidth in bits per second"
    )

    # Queue/Processing metrics
    dropped_frames: int = Field(
        default=0,
        description="Number of dropped frames/samples"
    )
    queue_latency_ms: Optional[float] = Field(
        default=None,
        description="Average queue wait time in milliseconds"
    )

    # Timestamp
    ts: datetime = Field(
        default_factory=datetime.utcnow,
        index=True,
        description="Event timestamp"
    )


# =============================================================================
# TelemetryModelBehaviorEvent - Model Decision Telemetry (Shadow/A-B)
# =============================================================================

class TelemetryModelBehaviorEvent(SQLModel, table=True):
    """
    Records model behavior for shadow mode and A/B testing.

    Enables comparison of model decisions between variants without
    affecting production actions (shadow mode) or with controlled
    rollout (A/B testing).
    """
    __tablename__ = "telemetry_model_behavior_event"
    __table_args__ = (
        Index('ix_telemetry_model_fleet_ts', 'fleet_id', 'ts'),
        Index('ix_telemetry_model_device_ts', 'device_id', 'ts'),
        Index('ix_telemetry_model_version_ts', 'model_version', 'ts'),
        Index('ix_telemetry_model_shadow', 'is_shadow', 'ts'),
        Index('ix_telemetry_model_tenant_ts', 'tenant_id', 'ts'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )
    device_id: str = Field(
        index=True,
        description="Device identifier (external)"
    )

    # Model identity
    model_version: str = Field(
        index=True,
        description="Model version identifier"
    )
    adapter_id: Optional[str] = Field(
        default=None,
        index=True,
        description="LoRA/PEFT adapter identifier"
    )

    # Decision metrics
    decision_hash: str = Field(
        description="Hash of model decision for comparison"
    )
    action_distribution_json: Optional[str] = Field(
        default=None,
        description="Action probability distribution (JSON)"
    )
    refusal_rate: Optional[float] = Field(
        default=None,
        description="Fraction of refused actions (0-1)"
    )

    # Safety metrics
    tool_call_failures: int = Field(
        default=0,
        description="Count of failed tool/action calls"
    )
    policy_constraint_hits: int = Field(
        default=0,
        description="Count of policy constraint violations"
    )

    # Shadow/A-B mode flag
    is_shadow: bool = Field(
        default=False,
        index=True,
        description="True if running in shadow mode (not acting)"
    )

    # Timestamp
    ts: datetime = Field(
        default_factory=datetime.utcnow,
        index=True,
        description="Event timestamp"
    )


# =============================================================================
# ForensicsEvent - Security and Safety Events
# =============================================================================

class ForensicsEvent(SQLModel, table=True):
    """
    Records forensics-grade events for incident investigation.

    Replaces mock incident data with real persisted events from
    edge agents and platform services.
    """
    __tablename__ = "forensics_event"
    __table_args__ = (
        Index('ix_forensics_fleet_ts', 'fleet_id', 'ts'),
        Index('ix_forensics_device_ts', 'device_id', 'ts'),
        Index('ix_forensics_severity_ts', 'severity', 'ts'),
        Index('ix_forensics_type_ts', 'event_type', 'ts'),
        Index('ix_forensics_tenant_ts', 'tenant_id', 'ts'),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: str = Field(
        foreign_key="tenant.id",
        index=True
    )
    fleet_id: str = Field(
        foreign_key="fleet.id",
        index=True
    )
    device_id: Optional[str] = Field(
        default=None,
        index=True,
        description="Device identifier (external), null for platform events"
    )

    # Event classification
    severity: str = Field(
        index=True,
        description="Event severity (critical/high/medium/low/info)"
    )
    event_type: str = Field(
        index=True,
        description="Event type classification"
    )

    # Evidence storage (pointer-based for security)
    evidence_ref: Optional[str] = Field(
        default=None,
        description="Reference/hash to encrypted evidence blob"
    )
    details_json: str = Field(
        default="{}",
        description="Event details (JSON, no raw sensitive data)"
    )

    # Timestamp
    ts: datetime = Field(
        default_factory=datetime.utcnow,
        index=True,
        description="Event timestamp"
    )


# =============================================================================
# Retention Policy Metadata
# =============================================================================

class TelemetryRetentionPolicy(SQLModel, table=True):
    """
    Defines retention policies for high-volume telemetry tables.

    Policies are evaluated by a background job to purge old data.
    """
    __tablename__ = "telemetry_retention_policy"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True
    )
    tenant_id: Optional[str] = Field(
        default=None,
        foreign_key="tenant.id",
        index=True,
        description="Tenant-specific policy, null for global"
    )
    table_name: str = Field(
        index=True,
        unique=True,
        description="Target table name"
    )
    retention_days: int = Field(
        default=30,
        description="Days to retain data"
    )
    enabled: bool = Field(
        default=True,
        description="Policy active status"
    )
    last_purge_at: Optional[datetime] = Field(
        default=None,
        description="Last purge execution timestamp"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow
    )
