"""
TenSafe Core Webhook Service.

Provides the main webhook management functionality including:
- Webhook registration and configuration
- Event triggering and routing
- Delivery status tracking
- Retry logic with exponential backoff

This module follows industry standards similar to Stripe and GitHub webhooks.
"""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import httpx

from .models import (
    DeliveryStatus,
    Webhook,
    WebhookDelivery,
    WebhookEvent,
    WebhookEventType,
    generate_delivery_id,
    generate_webhook_id,
)
from .signatures import (
    DELIVERY_ID_HEADER_NAME,
    EVENT_ID_HEADER_NAME,
    EVENT_TYPE_HEADER_NAME,
    SIGNATURE_HEADER_NAME,
    TIMESTAMP_HEADER_NAME,
    WEBHOOK_ID_HEADER_NAME,
    WebhookSigner,
    generate_secret,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 5
DEFAULT_MAX_RETRY_DELAY = 300  # 5 minutes
MAX_PAYLOAD_SIZE_BYTES = 256 * 1024  # 256 KB


@dataclass
class WebhookConfig:
    """Configuration for webhook service."""

    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_base_delay: int = DEFAULT_RETRY_BASE_DELAY
    max_retry_delay: int = DEFAULT_MAX_RETRY_DELAY
    max_payload_size: int = MAX_PAYLOAD_SIZE_BYTES
    allowed_schemes: Set[str] = field(default_factory=lambda: {"https", "http"})
    blocked_hosts: Set[str] = field(default_factory=lambda: {"localhost", "127.0.0.1", "0.0.0.0"})
    verify_ssl: bool = True
    user_agent: str = "TenSafe-Webhooks/1.0"


@dataclass
class DeliveryResult:
    """Result of a webhook delivery attempt."""

    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    response_headers: Dict[str, str] = field(default_factory=dict)
    response_time_ms: int = 0
    error_message: Optional[str] = None
    error_code: Optional[str] = None


@dataclass
class WebhookPayload:
    """Structured webhook payload."""

    id: str  # Unique event ID
    type: str  # Event type (e.g., "training.completed")
    created: int  # Unix timestamp
    data: Dict[str, Any]  # Event-specific data
    tenant_id: str
    api_version: str = "2024-01"
    livemode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "object": "event",
            "type": self.type,
            "created": self.created,
            "data": self.data,
            "tenant_id": self.tenant_id,
            "api_version": self.api_version,
            "livemode": self.livemode,
        }

    def to_bytes(self) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


class WebhookValidationError(Exception):
    """Raised when webhook validation fails."""

    pass


class WebhookDeliveryError(Exception):
    """Raised when webhook delivery fails."""

    def __init__(self, message: str, code: Optional[str] = None, retriable: bool = True):
        super().__init__(message)
        self.code = code
        self.retriable = retriable


class WebhookService:
    """
    Core webhook service for TenSafe.

    Provides:
    - Webhook registration with URL validation
    - Event subscription management
    - Secure payload delivery with HMAC-SHA256 signatures
    - Retry logic with exponential backoff
    - Delivery status tracking
    """

    def __init__(
        self,
        config: Optional[WebhookConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize webhook service.

        Args:
            config: Webhook configuration
            http_client: Optional HTTP client for testing
        """
        self.config = config or WebhookConfig()
        self._http_client = http_client
        self._webhooks: Dict[str, Webhook] = {}  # In-memory storage (use DB in production)
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._events: Dict[str, WebhookEvent] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                verify=self.config.verify_ssl,
                follow_redirects=False,  # Don't follow redirects for security
            )
        return self._http_client

    def validate_url(self, url: str) -> None:
        """
        Validate webhook URL.

        Args:
            url: URL to validate

        Raises:
            WebhookValidationError: If URL is invalid or blocked
        """
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise WebhookValidationError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme not in self.config.allowed_schemes:
            raise WebhookValidationError(
                f"URL scheme '{parsed.scheme}' not allowed. "
                f"Allowed schemes: {', '.join(self.config.allowed_schemes)}"
            )

        # Check host
        if not parsed.netloc:
            raise WebhookValidationError("URL must have a host")

        hostname = parsed.hostname or ""

        # Block internal hosts in production
        if hostname.lower() in self.config.blocked_hosts:
            raise WebhookValidationError(f"Host '{hostname}' is not allowed")

        # Block internal IP ranges
        if hostname.startswith("10.") or hostname.startswith("192.168."):
            raise WebhookValidationError("Internal IP addresses are not allowed")

        # Block localhost variants
        if "localhost" in hostname.lower() or hostname == "::1":
            raise WebhookValidationError("Localhost URLs are not allowed")

    def validate_events(self, events: List[str]) -> None:
        """
        Validate event types.

        Args:
            events: List of event types to validate

        Raises:
            WebhookValidationError: If any event type is invalid
        """
        valid_events = {e.value for e in WebhookEventType}
        valid_events.add("*")  # Wildcard for all events

        for event in events:
            if event not in valid_events:
                raise WebhookValidationError(
                    f"Invalid event type: '{event}'. "
                    f"Valid types: {', '.join(sorted(valid_events))}"
                )

    def create_webhook(
        self,
        tenant_id: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Webhook:
        """
        Register a new webhook.

        Args:
            tenant_id: Tenant ID
            url: Webhook URL
            events: List of event types to subscribe to
            secret: Webhook secret (generated if not provided)
            description: Optional description
            metadata: Optional metadata
            custom_headers: Custom headers to include in requests
            max_retries: Max retry attempts (default: 3)
            timeout_seconds: Request timeout (default: 30)

        Returns:
            Created Webhook

        Raises:
            WebhookValidationError: If validation fails
        """
        # Validate URL
        self.validate_url(url)

        # Validate events
        self.validate_events(events)

        # Generate secret if not provided
        if secret is None:
            secret = generate_secret()

        # Create webhook
        webhook = Webhook(
            id=generate_webhook_id(),
            tenant_id=tenant_id,
            url=url,
            events=events,
            secret=secret,
            description=description,
            metadata_json=metadata or {},
            custom_headers=custom_headers or {},
            max_retries=max_retries or self.config.max_retries,
            timeout_seconds=timeout_seconds or self.config.timeout_seconds,
            active=True,
        )

        # Store webhook
        self._webhooks[webhook.id] = webhook

        logger.info(
            f"Created webhook {webhook.id} for tenant {tenant_id} "
            f"with events: {events}"
        )

        return webhook

    def get_webhook(self, webhook_id: str, tenant_id: str) -> Optional[Webhook]:
        """
        Get a webhook by ID.

        Args:
            webhook_id: Webhook ID
            tenant_id: Tenant ID (for access control)

        Returns:
            Webhook if found and belongs to tenant, None otherwise
        """
        webhook = self._webhooks.get(webhook_id)
        if webhook and webhook.tenant_id == tenant_id:
            return webhook
        return None

    def list_webhooks(
        self,
        tenant_id: str,
        active_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Webhook]:
        """
        List webhooks for a tenant.

        Args:
            tenant_id: Tenant ID
            active_only: If True, only return active webhooks
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of webhooks
        """
        webhooks = [
            wh for wh in self._webhooks.values()
            if wh.tenant_id == tenant_id
            and (not active_only or wh.active)
        ]

        # Sort by created_at descending
        webhooks.sort(key=lambda w: w.created_at, reverse=True)

        return webhooks[offset:offset + limit]

    def update_webhook(
        self,
        webhook_id: str,
        tenant_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        description: Optional[str] = None,
        active: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Webhook]:
        """
        Update a webhook.

        Args:
            webhook_id: Webhook ID
            tenant_id: Tenant ID
            url: New URL (optional)
            events: New event subscriptions (optional)
            description: New description (optional)
            active: New active status (optional)
            metadata: New metadata (optional)
            custom_headers: New custom headers (optional)

        Returns:
            Updated webhook if found, None otherwise
        """
        webhook = self.get_webhook(webhook_id, tenant_id)
        if webhook is None:
            return None

        # Validate new values
        if url is not None:
            self.validate_url(url)
            webhook.url = url

        if events is not None:
            self.validate_events(events)
            webhook.events = events

        if description is not None:
            webhook.description = description

        if active is not None:
            webhook.active = active

        if metadata is not None:
            webhook.metadata_json = metadata

        if custom_headers is not None:
            webhook.custom_headers = custom_headers

        webhook.updated_at = datetime.utcnow()

        logger.info(f"Updated webhook {webhook_id}")

        return webhook

    def delete_webhook(self, webhook_id: str, tenant_id: str) -> bool:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook ID
            tenant_id: Tenant ID

        Returns:
            True if deleted, False if not found
        """
        webhook = self.get_webhook(webhook_id, tenant_id)
        if webhook is None:
            return False

        del self._webhooks[webhook_id]

        logger.info(f"Deleted webhook {webhook_id}")

        return True

    def rotate_secret(self, webhook_id: str, tenant_id: str) -> Optional[str]:
        """
        Rotate webhook secret.

        Args:
            webhook_id: Webhook ID
            tenant_id: Tenant ID

        Returns:
            New secret if successful, None if webhook not found
        """
        webhook = self.get_webhook(webhook_id, tenant_id)
        if webhook is None:
            return None

        new_secret = generate_secret()
        webhook.secret = new_secret
        webhook.updated_at = datetime.utcnow()

        logger.info(f"Rotated secret for webhook {webhook_id}")

        return new_secret

    def get_webhooks_for_event(
        self, tenant_id: str, event_type: str
    ) -> List[Webhook]:
        """
        Get all active webhooks subscribed to an event type.

        Args:
            tenant_id: Tenant ID
            event_type: Event type

        Returns:
            List of matching webhooks
        """
        return [
            wh for wh in self._webhooks.values()
            if wh.tenant_id == tenant_id
            and wh.active
            and wh.is_subscribed_to(event_type)
        ]

    def create_payload(
        self,
        event_type: str,
        data: Dict[str, Any],
        tenant_id: str,
        event_id: Optional[str] = None,
    ) -> WebhookPayload:
        """
        Create a structured webhook payload.

        Args:
            event_type: Event type
            data: Event-specific data
            tenant_id: Tenant ID
            event_id: Optional event ID (generated if not provided)

        Returns:
            WebhookPayload
        """
        return WebhookPayload(
            id=event_id or f"evt-{secrets.token_hex(16)}",
            type=event_type,
            created=int(time.time()),
            data=data,
            tenant_id=tenant_id,
        )

    async def deliver(
        self,
        webhook: Webhook,
        payload: WebhookPayload,
        delivery_id: Optional[str] = None,
    ) -> DeliveryResult:
        """
        Deliver a webhook payload to the target URL.

        Args:
            webhook: Webhook configuration
            payload: Payload to deliver
            delivery_id: Optional delivery ID (generated if not provided)

        Returns:
            DeliveryResult with outcome details
        """
        delivery_id = delivery_id or generate_delivery_id()
        payload_bytes = payload.to_bytes()

        # Check payload size
        if len(payload_bytes) > self.config.max_payload_size:
            return DeliveryResult(
                success=False,
                error_message=f"Payload size ({len(payload_bytes)} bytes) exceeds limit "
                f"({self.config.max_payload_size} bytes)",
                error_code="PAYLOAD_TOO_LARGE",
            )

        # Sign payload
        signer = WebhookSigner(webhook.secret)
        signature_headers = signer.sign(payload.to_dict())

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self.config.user_agent,
            SIGNATURE_HEADER_NAME: signature_headers["X-TenSafe-Signature"],
            TIMESTAMP_HEADER_NAME: signature_headers["X-TenSafe-Timestamp"],
            EVENT_ID_HEADER_NAME: payload.id,
            EVENT_TYPE_HEADER_NAME: payload.type,
            WEBHOOK_ID_HEADER_NAME: webhook.id,
            DELIVERY_ID_HEADER_NAME: delivery_id,
        }

        # Add custom headers
        headers.update(webhook.custom_headers)

        # Send request
        start_time = time.perf_counter()
        try:
            client = await self._get_http_client()
            response = await client.post(
                webhook.url,
                content=payload_bytes,
                headers=headers,
                timeout=webhook.timeout_seconds,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            # Check for success (2xx status codes)
            success = 200 <= response.status_code < 300

            # Read response body (limited size)
            response_body = response.text[:10000] if response.text else None

            result = DeliveryResult(
                success=success,
                status_code=response.status_code,
                response_body=response_body,
                response_headers=dict(response.headers),
                response_time_ms=elapsed_ms,
            )

            if not success:
                result.error_message = f"Non-2xx status code: {response.status_code}"
                result.error_code = f"HTTP_{response.status_code}"

            return result

        except httpx.TimeoutException:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return DeliveryResult(
                success=False,
                response_time_ms=elapsed_ms,
                error_message=f"Request timed out after {webhook.timeout_seconds} seconds",
                error_code="TIMEOUT",
            )

        except httpx.ConnectError as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return DeliveryResult(
                success=False,
                response_time_ms=elapsed_ms,
                error_message=f"Connection failed: {str(e)}",
                error_code="CONNECTION_ERROR",
            )

        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.exception(f"Webhook delivery failed: {e}")
            return DeliveryResult(
                success=False,
                response_time_ms=elapsed_ms,
                error_message=f"Delivery failed: {str(e)}",
                error_code="DELIVERY_ERROR",
            )

    async def deliver_with_retry(
        self,
        webhook: Webhook,
        payload: WebhookPayload,
        max_retries: Optional[int] = None,
    ) -> WebhookDelivery:
        """
        Deliver a webhook with automatic retry on failure.

        Uses exponential backoff for retry delays.

        Args:
            webhook: Webhook configuration
            payload: Payload to deliver
            max_retries: Maximum retry attempts (default: from webhook config)

        Returns:
            WebhookDelivery with final status
        """
        max_retries = max_retries or webhook.max_retries
        delivery_id = generate_delivery_id()

        # Create delivery record
        idempotency_key = hashlib.sha256(
            f"{webhook.id}:{payload.id}".encode()
        ).hexdigest()

        delivery = WebhookDelivery(
            id=delivery_id,
            webhook_id=webhook.id,
            tenant_id=webhook.tenant_id,
            event_type=payload.type,
            event_id=payload.id,
            payload=payload.to_dict(),
            status=DeliveryStatus.PENDING.value,
            max_attempts=max_retries,
            idempotency_key=idempotency_key,
        )

        self._deliveries[delivery_id] = delivery

        # Attempt delivery with retries
        for attempt in range(max_retries + 1):
            delivery.attempts = attempt + 1

            if attempt == 0:
                delivery.first_attempt_at = datetime.utcnow()
            delivery.last_attempt_at = datetime.utcnow()

            logger.info(
                f"Delivering webhook {webhook.id} to {webhook.url} "
                f"(attempt {attempt + 1}/{max_retries + 1})"
            )

            delivery.status = DeliveryStatus.IN_PROGRESS.value
            result = await self.deliver(webhook, payload, delivery_id)

            # Update delivery record
            delivery.response_status_code = result.status_code
            delivery.response_body = result.response_body
            delivery.response_headers = result.response_headers
            delivery.response_time_ms = result.response_time_ms
            delivery.error_message = result.error_message
            delivery.error_code = result.error_code

            if result.success:
                delivery.status = DeliveryStatus.DELIVERED.value
                delivery.delivered_at = datetime.utcnow()

                # Update webhook stats
                webhook.last_triggered_at = datetime.utcnow()
                webhook.last_success_at = datetime.utcnow()
                webhook.total_deliveries += 1
                webhook.successful_deliveries += 1

                logger.info(
                    f"Webhook {webhook.id} delivered successfully "
                    f"(status: {result.status_code}, time: {result.response_time_ms}ms)"
                )

                return delivery

            # Check if we should retry
            if attempt < max_retries:
                # Calculate backoff delay
                delay = self.config.retry_base_delay * (2 ** attempt)
                delay = min(delay, self.config.max_retry_delay)

                delivery.status = DeliveryStatus.RETRYING.value
                delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

                logger.warning(
                    f"Webhook {webhook.id} delivery failed "
                    f"(attempt {attempt + 1}, error: {result.error_code}). "
                    f"Retrying in {delay} seconds..."
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        delivery.status = DeliveryStatus.FAILED.value

        # Update webhook stats
        webhook.last_triggered_at = datetime.utcnow()
        webhook.last_failure_at = datetime.utcnow()
        webhook.total_deliveries += 1
        webhook.failed_deliveries += 1

        logger.error(
            f"Webhook {webhook.id} delivery failed after {max_retries + 1} attempts"
        )

        return delivery

    async def trigger_event(
        self,
        tenant_id: str,
        event_type: str,
        data: Dict[str, Any],
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        """
        Trigger an event and deliver to all subscribed webhooks.

        Args:
            tenant_id: Tenant ID
            event_type: Event type
            data: Event data
            source_type: Type of source that triggered the event
            source_id: ID of the source resource

        Returns:
            List of delivery results
        """
        # Create event record
        payload = self.create_payload(event_type, data, tenant_id)

        event = WebhookEvent(
            id=payload.id,
            tenant_id=tenant_id,
            event_type=event_type,
            event_data=data,
            source_type=source_type,
            source_id=source_id,
        )
        self._events[event.id] = event

        # Find subscribed webhooks
        webhooks = self.get_webhooks_for_event(tenant_id, event_type)

        if not webhooks:
            logger.debug(f"No webhooks subscribed to event {event_type}")
            return []

        logger.info(
            f"Triggering event {event_type} for tenant {tenant_id}, "
            f"{len(webhooks)} webhooks subscribed"
        )

        # Deliver to all webhooks concurrently
        deliveries = await asyncio.gather(
            *[self.deliver_with_retry(wh, payload) for wh in webhooks],
            return_exceptions=True,
        )

        # Filter out exceptions and update event
        results = []
        for delivery in deliveries:
            if isinstance(delivery, WebhookDelivery):
                results.append(delivery)
            else:
                logger.error(f"Webhook delivery exception: {delivery}")

        # Update event record
        event.processed = True
        event.processed_at = datetime.utcnow()
        event.webhooks_triggered = len(results)

        return results

    def get_delivery(
        self, delivery_id: str, tenant_id: str
    ) -> Optional[WebhookDelivery]:
        """
        Get a delivery by ID.

        Args:
            delivery_id: Delivery ID
            tenant_id: Tenant ID

        Returns:
            WebhookDelivery if found, None otherwise
        """
        delivery = self._deliveries.get(delivery_id)
        if delivery and delivery.tenant_id == tenant_id:
            return delivery
        return None

    def list_deliveries(
        self,
        tenant_id: str,
        webhook_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[WebhookDelivery]:
        """
        List deliveries with filtering.

        Args:
            tenant_id: Tenant ID
            webhook_id: Filter by webhook ID (optional)
            status: Filter by status (optional)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of deliveries
        """
        deliveries = [
            d for d in self._deliveries.values()
            if d.tenant_id == tenant_id
            and (webhook_id is None or d.webhook_id == webhook_id)
            and (status is None or d.status == status)
        ]

        # Sort by created_at descending
        deliveries.sort(key=lambda d: d.created_at, reverse=True)

        return deliveries[offset:offset + limit]

    async def send_test_event(
        self, webhook_id: str, tenant_id: str
    ) -> Optional[WebhookDelivery]:
        """
        Send a test event to a webhook.

        Args:
            webhook_id: Webhook ID
            tenant_id: Tenant ID

        Returns:
            Delivery result if webhook found, None otherwise
        """
        webhook = self.get_webhook(webhook_id, tenant_id)
        if webhook is None:
            return None

        # Create test payload
        payload = self.create_payload(
            event_type="test",
            data={
                "message": "This is a test webhook event from TenSafe",
                "webhook_id": webhook_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
            tenant_id=tenant_id,
        )

        # Deliver without retry for test events
        delivery = await self.deliver_with_retry(webhook, payload, max_retries=0)

        return delivery

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


# Global webhook service instance
_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Get the global webhook service instance."""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service


def set_webhook_service(service: WebhookService) -> None:
    """Set the global webhook service instance."""
    global _webhook_service
    _webhook_service = service
