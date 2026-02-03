"""
TenSafe Webhook API Routes.

FastAPI routes for webhook management including:
- CRUD operations for webhooks
- Test event triggering
- Delivery history and status

All endpoints require authentication and are tenant-scoped.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field, HttpUrl, field_validator

from .models import DeliveryStatus, WebhookEventType
from .webhooks import (
    WebhookService,
    WebhookValidationError,
    get_webhook_service,
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ==============================================================================
# Request/Response Models
# ==============================================================================


class WebhookCreateRequest(BaseModel):
    """Request to create a new webhook."""

    url: str = Field(
        ...,
        description="HTTPS URL to receive webhook events",
        examples=["https://api.example.com/webhooks/tensafe"],
    )
    events: List[str] = Field(
        ...,
        description="List of event types to subscribe to. Use '*' for all events.",
        examples=[["training.completed", "checkpoint.saved"]],
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional description for this webhook",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom metadata to store with the webhook",
    )
    custom_headers: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Custom headers to include in webhook requests",
    )
    max_retries: Optional[int] = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed deliveries",
    )
    timeout_seconds: Optional[int] = Field(
        default=30,
        ge=5,
        le=120,
        description="Request timeout in seconds",
    )

    @field_validator("events")
    @classmethod
    def validate_events_not_empty(cls, v):
        if not v:
            raise ValueError("At least one event type must be specified")
        return v


class WebhookUpdateRequest(BaseModel):
    """Request to update an existing webhook."""

    url: Optional[str] = Field(
        None,
        description="New URL for the webhook",
    )
    events: Optional[List[str]] = Field(
        None,
        description="New list of event types",
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="New description",
    )
    active: Optional[bool] = Field(
        None,
        description="Enable/disable the webhook",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="New metadata",
    )
    custom_headers: Optional[Dict[str, str]] = Field(
        None,
        description="New custom headers",
    )


class WebhookResponse(BaseModel):
    """Response containing webhook details."""

    id: str
    tenant_id: str
    url: str
    events: List[str]
    description: Optional[str]
    active: bool
    metadata: Dict[str, Any]
    custom_headers: Dict[str, str]
    max_retries: int
    timeout_seconds: int
    created_at: datetime
    updated_at: datetime
    last_triggered_at: Optional[datetime]
    last_success_at: Optional[datetime]
    last_failure_at: Optional[datetime]

    # Statistics (secret is never exposed)
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int


class WebhookSecretResponse(BaseModel):
    """Response containing webhook with secret (only on create)."""

    webhook: WebhookResponse
    secret: str = Field(
        ...,
        description="Webhook signing secret. Store securely - this is only shown once.",
    )


class DeliveryResponse(BaseModel):
    """Response containing delivery details."""

    id: str
    webhook_id: str
    tenant_id: str
    event_type: str
    event_id: str
    status: str
    attempts: int
    max_attempts: int
    created_at: datetime
    first_attempt_at: Optional[datetime]
    last_attempt_at: Optional[datetime]
    delivered_at: Optional[datetime]
    next_retry_at: Optional[datetime]
    response_status_code: Optional[int]
    response_time_ms: Optional[int]
    error_message: Optional[str]
    error_code: Optional[str]


class DeliveryDetailResponse(DeliveryResponse):
    """Response containing full delivery details including payload."""

    payload: Dict[str, Any]
    response_body: Optional[str]
    response_headers: Dict[str, str]
    request_headers: Dict[str, str]


class TestEventResponse(BaseModel):
    """Response from sending a test event."""

    success: bool
    delivery_id: str
    status: str
    response_status_code: Optional[int]
    response_time_ms: Optional[int]
    error_message: Optional[str]


class WebhookListResponse(BaseModel):
    """Response containing list of webhooks."""

    webhooks: List[WebhookResponse]
    total: int
    has_more: bool


class DeliveryListResponse(BaseModel):
    """Response containing list of deliveries."""

    deliveries: List[DeliveryResponse]
    total: int
    has_more: bool


class EventTypesResponse(BaseModel):
    """Response containing available event types."""

    event_types: List[Dict[str, str]]


class RotateSecretResponse(BaseModel):
    """Response from rotating webhook secret."""

    webhook_id: str
    new_secret: str = Field(
        ...,
        description="New webhook signing secret. Store securely.",
    )


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: ErrorDetail


# ==============================================================================
# Dependency: Get tenant from API key
# ==============================================================================


async def get_tenant_id(
    authorization: str = Header(..., description="Bearer token"),
) -> str:
    """
    Extract tenant ID from authorization header.

    In production, this would validate the token and extract the tenant.
    For demo purposes, derives tenant ID from token hash.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "code": "AUTHENTICATION_REQUIRED",
                    "message": "Invalid authorization header",
                }
            },
        )

    token = authorization[7:]

    # In production, validate token and extract tenant
    # For demo, derive tenant from token hash
    tenant_id = f"tenant-{hashlib.sha256(token.encode()).hexdigest()[:8]}"
    return tenant_id


def get_service() -> WebhookService:
    """Get webhook service instance."""
    return get_webhook_service()


# ==============================================================================
# Webhook CRUD Endpoints
# ==============================================================================


@router.post(
    "",
    response_model=WebhookSecretResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def create_webhook(
    request: WebhookCreateRequest,
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> WebhookSecretResponse:
    """
    Create a new webhook.

    Creates a webhook subscription that will receive events at the specified URL.
    The webhook secret is returned only once - store it securely for signature verification.

    Supported event types:
    - training.started - Training job started
    - training.completed - Training job completed successfully
    - training.failed - Training job failed
    - checkpoint.saved - Model checkpoint saved
    - quota.warning - Approaching usage quota
    - quota.exceeded - Usage quota exceeded
    - security.alert - Security-related event
    - key.rotated - Encryption key rotated
    - adapter.loaded - TGSP adapter loaded
    - adapter.activated - TGSP adapter activated
    - inference.completed - Inference request completed

    Use "*" to subscribe to all events.
    """
    try:
        webhook = service.create_webhook(
            tenant_id=tenant_id,
            url=request.url,
            events=request.events,
            description=request.description,
            metadata=request.metadata,
            custom_headers=request.custom_headers,
            max_retries=request.max_retries,
            timeout_seconds=request.timeout_seconds,
        )

        webhook_response = WebhookResponse(
            id=webhook.id,
            tenant_id=webhook.tenant_id,
            url=webhook.url,
            events=webhook.events,
            description=webhook.description,
            active=webhook.active,
            metadata=webhook.metadata_json,
            custom_headers=webhook.custom_headers,
            max_retries=webhook.max_retries,
            timeout_seconds=webhook.timeout_seconds,
            created_at=webhook.created_at,
            updated_at=webhook.updated_at,
            last_triggered_at=webhook.last_triggered_at,
            last_success_at=webhook.last_success_at,
            last_failure_at=webhook.last_failure_at,
            total_deliveries=webhook.total_deliveries,
            successful_deliveries=webhook.successful_deliveries,
            failed_deliveries=webhook.failed_deliveries,
        )

        return WebhookSecretResponse(
            webhook=webhook_response,
            secret=webhook.secret,
        )

    except WebhookValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(e),
                }
            },
        )


@router.get(
    "",
    response_model=WebhookListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_webhooks(
    active_only: bool = Query(False, description="Only return active webhooks"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> WebhookListResponse:
    """
    List all webhooks for the tenant.

    Returns webhooks ordered by creation date (newest first).
    Use pagination parameters for large result sets.
    """
    webhooks = service.list_webhooks(
        tenant_id=tenant_id,
        active_only=active_only,
        limit=limit + 1,  # Fetch one extra to check has_more
        offset=offset,
    )

    has_more = len(webhooks) > limit
    if has_more:
        webhooks = webhooks[:limit]

    return WebhookListResponse(
        webhooks=[
            WebhookResponse(
                id=wh.id,
                tenant_id=wh.tenant_id,
                url=wh.url,
                events=wh.events,
                description=wh.description,
                active=wh.active,
                metadata=wh.metadata_json,
                custom_headers=wh.custom_headers,
                max_retries=wh.max_retries,
                timeout_seconds=wh.timeout_seconds,
                created_at=wh.created_at,
                updated_at=wh.updated_at,
                last_triggered_at=wh.last_triggered_at,
                last_success_at=wh.last_success_at,
                last_failure_at=wh.last_failure_at,
                total_deliveries=wh.total_deliveries,
                successful_deliveries=wh.successful_deliveries,
                failed_deliveries=wh.failed_deliveries,
            )
            for wh in webhooks
        ],
        total=len(webhooks),
        has_more=has_more,
    )


@router.get(
    "/{webhook_id}",
    response_model=WebhookResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Webhook not found"},
    },
)
async def get_webhook(
    webhook_id: str,
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> WebhookResponse:
    """
    Get details of a specific webhook.

    Note: The webhook secret is not included in this response for security.
    Use the rotate secret endpoint if you need a new secret.
    """
    webhook = service.get_webhook(webhook_id, tenant_id)
    if webhook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "WEBHOOK_NOT_FOUND",
                    "message": f"Webhook '{webhook_id}' not found",
                    "details": {"webhook_id": webhook_id},
                }
            },
        )

    return WebhookResponse(
        id=webhook.id,
        tenant_id=webhook.tenant_id,
        url=webhook.url,
        events=webhook.events,
        description=webhook.description,
        active=webhook.active,
        metadata=webhook.metadata_json,
        custom_headers=webhook.custom_headers,
        max_retries=webhook.max_retries,
        timeout_seconds=webhook.timeout_seconds,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
        last_triggered_at=webhook.last_triggered_at,
        last_success_at=webhook.last_success_at,
        last_failure_at=webhook.last_failure_at,
        total_deliveries=webhook.total_deliveries,
        successful_deliveries=webhook.successful_deliveries,
        failed_deliveries=webhook.failed_deliveries,
    )


@router.patch(
    "/{webhook_id}",
    response_model=WebhookResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Webhook not found"},
    },
)
async def update_webhook(
    webhook_id: str,
    request: WebhookUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> WebhookResponse:
    """
    Update a webhook.

    Allows updating URL, events, description, active status, metadata, and custom headers.
    All fields are optional - only provided fields will be updated.
    """
    try:
        webhook = service.update_webhook(
            webhook_id=webhook_id,
            tenant_id=tenant_id,
            url=request.url,
            events=request.events,
            description=request.description,
            active=request.active,
            metadata=request.metadata,
            custom_headers=request.custom_headers,
        )

        if webhook is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "WEBHOOK_NOT_FOUND",
                        "message": f"Webhook '{webhook_id}' not found",
                    }
                },
            )

        return WebhookResponse(
            id=webhook.id,
            tenant_id=webhook.tenant_id,
            url=webhook.url,
            events=webhook.events,
            description=webhook.description,
            active=webhook.active,
            metadata=webhook.metadata_json,
            custom_headers=webhook.custom_headers,
            max_retries=webhook.max_retries,
            timeout_seconds=webhook.timeout_seconds,
            created_at=webhook.created_at,
            updated_at=webhook.updated_at,
            last_triggered_at=webhook.last_triggered_at,
            last_success_at=webhook.last_success_at,
            last_failure_at=webhook.last_failure_at,
            total_deliveries=webhook.total_deliveries,
            successful_deliveries=webhook.successful_deliveries,
            failed_deliveries=webhook.failed_deliveries,
        )

    except WebhookValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(e),
                }
            },
        )


@router.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Webhook not found"},
    },
)
async def delete_webhook(
    webhook_id: str,
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
):
    """
    Delete a webhook.

    This permanently removes the webhook and stops all event delivery.
    Pending deliveries will be cancelled.
    """
    success = service.delete_webhook(webhook_id, tenant_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "WEBHOOK_NOT_FOUND",
                    "message": f"Webhook '{webhook_id}' not found",
                }
            },
        )


# ==============================================================================
# Test and Secret Management Endpoints
# ==============================================================================


@router.post(
    "/{webhook_id}/test",
    response_model=TestEventResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Webhook not found"},
    },
)
async def send_test_event(
    webhook_id: str,
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> TestEventResponse:
    """
    Send a test event to a webhook.

    Sends a test payload to verify the webhook endpoint is working correctly.
    The test event has type "test" and includes a test message.
    """
    delivery = await service.send_test_event(webhook_id, tenant_id)

    if delivery is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "WEBHOOK_NOT_FOUND",
                    "message": f"Webhook '{webhook_id}' not found",
                }
            },
        )

    return TestEventResponse(
        success=delivery.status == DeliveryStatus.DELIVERED.value,
        delivery_id=delivery.id,
        status=delivery.status,
        response_status_code=delivery.response_status_code,
        response_time_ms=delivery.response_time_ms,
        error_message=delivery.error_message,
    )


@router.post(
    "/{webhook_id}/rotate-secret",
    response_model=RotateSecretResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Webhook not found"},
    },
)
async def rotate_webhook_secret(
    webhook_id: str,
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> RotateSecretResponse:
    """
    Rotate the webhook signing secret.

    Generates a new secret for the webhook. The new secret is returned only once.
    Update your webhook handler to use the new secret for signature verification.

    Note: There is no grace period - the old secret becomes invalid immediately.
    """
    new_secret = service.rotate_secret(webhook_id, tenant_id)

    if new_secret is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "WEBHOOK_NOT_FOUND",
                    "message": f"Webhook '{webhook_id}' not found",
                }
            },
        )

    return RotateSecretResponse(
        webhook_id=webhook_id,
        new_secret=new_secret,
    )


# ==============================================================================
# Delivery History Endpoints
# ==============================================================================


@router.get(
    "/{webhook_id}/deliveries",
    response_model=DeliveryListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Webhook not found"},
    },
)
async def list_deliveries(
    webhook_id: str,
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by delivery status",
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> DeliveryListResponse:
    """
    Get delivery history for a webhook.

    Returns delivery attempts ordered by creation date (newest first).
    Use status filter to find failed deliveries for debugging.
    """
    # Verify webhook exists and belongs to tenant
    webhook = service.get_webhook(webhook_id, tenant_id)
    if webhook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "WEBHOOK_NOT_FOUND",
                    "message": f"Webhook '{webhook_id}' not found",
                }
            },
        )

    deliveries = service.list_deliveries(
        tenant_id=tenant_id,
        webhook_id=webhook_id,
        status=status_filter,
        limit=limit + 1,
        offset=offset,
    )

    has_more = len(deliveries) > limit
    if has_more:
        deliveries = deliveries[:limit]

    return DeliveryListResponse(
        deliveries=[
            DeliveryResponse(
                id=d.id,
                webhook_id=d.webhook_id,
                tenant_id=d.tenant_id,
                event_type=d.event_type,
                event_id=d.event_id,
                status=d.status,
                attempts=d.attempts,
                max_attempts=d.max_attempts,
                created_at=d.created_at,
                first_attempt_at=d.first_attempt_at,
                last_attempt_at=d.last_attempt_at,
                delivered_at=d.delivered_at,
                next_retry_at=d.next_retry_at,
                response_status_code=d.response_status_code,
                response_time_ms=d.response_time_ms,
                error_message=d.error_message,
                error_code=d.error_code,
            )
            for d in deliveries
        ],
        total=len(deliveries),
        has_more=has_more,
    )


@router.get(
    "/{webhook_id}/deliveries/{delivery_id}",
    response_model=DeliveryDetailResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Delivery not found"},
    },
)
async def get_delivery(
    webhook_id: str,
    delivery_id: str,
    tenant_id: str = Depends(get_tenant_id),
    service: WebhookService = Depends(get_service),
) -> DeliveryDetailResponse:
    """
    Get detailed information about a specific delivery.

    Includes full payload, response body, and headers for debugging.
    """
    delivery = service.get_delivery(delivery_id, tenant_id)

    if delivery is None or delivery.webhook_id != webhook_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "DELIVERY_NOT_FOUND",
                    "message": f"Delivery '{delivery_id}' not found",
                }
            },
        )

    return DeliveryDetailResponse(
        id=delivery.id,
        webhook_id=delivery.webhook_id,
        tenant_id=delivery.tenant_id,
        event_type=delivery.event_type,
        event_id=delivery.event_id,
        status=delivery.status,
        attempts=delivery.attempts,
        max_attempts=delivery.max_attempts,
        created_at=delivery.created_at,
        first_attempt_at=delivery.first_attempt_at,
        last_attempt_at=delivery.last_attempt_at,
        delivered_at=delivery.delivered_at,
        next_retry_at=delivery.next_retry_at,
        response_status_code=delivery.response_status_code,
        response_time_ms=delivery.response_time_ms,
        error_message=delivery.error_message,
        error_code=delivery.error_code,
        payload=delivery.payload,
        response_body=delivery.response_body,
        response_headers=delivery.response_headers,
        request_headers=delivery.request_headers,
    )


# ==============================================================================
# Event Types Endpoint
# ==============================================================================


@router.get(
    "/event-types",
    response_model=EventTypesResponse,
    tags=["webhooks"],
)
async def list_event_types() -> EventTypesResponse:
    """
    List all available webhook event types.

    Returns the complete list of events that can be subscribed to.
    """
    event_descriptions = {
        WebhookEventType.TRAINING_STARTED: "Triggered when a training job starts",
        WebhookEventType.TRAINING_COMPLETED: "Triggered when a training job completes successfully",
        WebhookEventType.TRAINING_FAILED: "Triggered when a training job fails",
        WebhookEventType.CHECKPOINT_SAVED: "Triggered when a model checkpoint is saved",
        WebhookEventType.QUOTA_WARNING: "Triggered when usage approaches quota limit",
        WebhookEventType.QUOTA_EXCEEDED: "Triggered when usage quota is exceeded",
        WebhookEventType.SECURITY_ALERT: "Triggered for security-related events",
        WebhookEventType.KEY_ROTATED: "Triggered when an encryption key is rotated",
        WebhookEventType.ADAPTER_LOADED: "Triggered when a TGSP adapter is loaded",
        WebhookEventType.ADAPTER_ACTIVATED: "Triggered when a TGSP adapter is activated",
        WebhookEventType.INFERENCE_COMPLETED: "Triggered when an inference request completes",
    }

    return EventTypesResponse(
        event_types=[
            {
                "type": event.value,
                "description": event_descriptions.get(event, ""),
            }
            for event in WebhookEventType
        ]
    )
