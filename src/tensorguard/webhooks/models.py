"""
TenSafe Webhook Database Models.

SQLModel definitions for webhooks and delivery tracking.
Provides multi-tenant webhook management with comprehensive audit trails.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlmodel import JSON, Column, Field, Relationship, SQLModel


def generate_webhook_id() -> str:
    """Generate a unique webhook ID."""
    return f"wh-{uuid.uuid4()}"


def generate_delivery_id() -> str:
    """Generate a unique delivery ID."""
    return f"whd-{uuid.uuid4()}"


class WebhookEventType(str, Enum):
    """Supported webhook event types."""

    # Training lifecycle events
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"

    # Checkpoint events
    CHECKPOINT_SAVED = "checkpoint.saved"

    # Quota/usage events
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"

    # Security events
    SECURITY_ALERT = "security.alert"
    KEY_ROTATED = "key.rotated"

    # Adapter events (TGSP)
    ADAPTER_LOADED = "adapter.loaded"
    ADAPTER_ACTIVATED = "adapter.activated"
    INFERENCE_COMPLETED = "inference.completed"


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class Webhook(SQLModel, table=True):
    """
    Webhook registration model.

    Stores webhook configurations per tenant with:
    - Target URL for delivery
    - Event subscriptions
    - HMAC secret for signature verification
    - Active/inactive status
    """

    __tablename__ = "webhooks"

    # Primary key
    id: str = Field(default_factory=generate_webhook_id, primary_key=True)

    # Tenant association (multi-tenant support)
    tenant_id: str = Field(index=True)

    # Webhook configuration
    url: str = Field(max_length=2048)  # Target URL for webhook delivery
    description: Optional[str] = Field(default=None, max_length=500)

    # Event subscriptions (list of WebhookEventType values)
    events: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Security
    secret: str = Field(max_length=256)  # HMAC secret for signature generation

    # Status
    active: bool = Field(default=True)

    # Metadata
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Headers to include with webhook requests
    custom_headers: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))

    # Rate limiting
    max_retries: int = Field(default=3)
    timeout_seconds: int = Field(default=30)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Last successful delivery
    last_triggered_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None

    # Statistics
    total_deliveries: int = Field(default=0)
    successful_deliveries: int = Field(default=0)
    failed_deliveries: int = Field(default=0)

    # Relationships
    deliveries: List["WebhookDelivery"] = Relationship(back_populates="webhook")

    def is_subscribed_to(self, event_type: str) -> bool:
        """Check if this webhook is subscribed to a specific event type."""
        return event_type in self.events or "*" in self.events


class WebhookDelivery(SQLModel, table=True):
    """
    Webhook delivery attempt tracking.

    Records each delivery attempt with:
    - Full payload for audit
    - Response status and body
    - Retry tracking
    - Timing information
    """

    __tablename__ = "webhook_deliveries"

    # Primary key
    id: str = Field(default_factory=generate_delivery_id, primary_key=True)

    # Foreign key to webhook
    webhook_id: str = Field(foreign_key="webhooks.id", index=True)

    # Tenant association (denormalized for efficient queries)
    tenant_id: str = Field(index=True)

    # Event information
    event_type: str = Field(index=True)
    event_id: str = Field(index=True)  # Unique ID for this specific event

    # Payload
    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Delivery status
    status: str = Field(default=DeliveryStatus.PENDING.value, index=True)

    # Retry tracking
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)
    next_retry_at: Optional[datetime] = None

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    first_attempt_at: Optional[datetime] = None
    last_attempt_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    # Response information
    response_status_code: Optional[int] = None
    response_body: Optional[str] = Field(default=None, max_length=10000)
    response_headers: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    response_time_ms: Optional[int] = None

    # Error tracking
    error_message: Optional[str] = Field(default=None, max_length=2000)
    error_code: Optional[str] = Field(default=None, max_length=100)

    # Signature used for this delivery
    signature: Optional[str] = Field(default=None, max_length=256)

    # Idempotency key for deduplication
    idempotency_key: str = Field(index=True)

    # Request metadata
    request_headers: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))

    # Relationship
    webhook: Optional[Webhook] = Relationship(back_populates="deliveries")

    def can_retry(self) -> bool:
        """Check if this delivery can be retried."""
        return (
            self.status in (DeliveryStatus.FAILED.value, DeliveryStatus.RETRYING.value)
            and self.attempts < self.max_attempts
        )

    def get_retry_delay_seconds(self) -> int:
        """
        Calculate exponential backoff delay for retry.

        Base delay: 5 seconds
        Formula: base * (2 ^ (attempts - 1))
        Max delay: 5 minutes (300 seconds)
        """
        base_delay = 5
        max_delay = 300
        delay = base_delay * (2 ** (self.attempts - 1))
        return min(delay, max_delay)


class WebhookEvent(SQLModel, table=True):
    """
    Webhook event log.

    Records all events that trigger webhooks for audit and replay purposes.
    """

    __tablename__ = "webhook_events"

    # Primary key
    id: str = Field(default_factory=lambda: f"whe-{uuid.uuid4()}", primary_key=True)

    # Tenant association
    tenant_id: str = Field(index=True)

    # Event information
    event_type: str = Field(index=True)
    event_data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Source information
    source_type: Optional[str] = None  # e.g., "training_client", "job", "adapter"
    source_id: Optional[str] = None  # ID of the source resource

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Processing status
    processed: bool = Field(default=False)
    processed_at: Optional[datetime] = None

    # Number of webhooks triggered by this event
    webhooks_triggered: int = Field(default=0)
