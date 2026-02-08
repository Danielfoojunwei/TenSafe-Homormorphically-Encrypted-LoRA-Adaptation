"""
Service Telemetry Collector

Collects and aggregates telemetry data from MSS and HAS services.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TelemetryEventType(str, Enum):
    """Types of telemetry events."""
    # Request lifecycle
    REQUEST_START = "request_start"
    REQUEST_COMPLETE = "request_complete"
    REQUEST_ERROR = "request_error"

    # Token processing
    TOKEN_PREFILL = "token_prefill"
    TOKEN_DECODE = "token_decode"

    # HE operations
    HE_ENCRYPT = "he_encrypt"
    HE_COMPUTE = "he_compute"
    HE_DECRYPT = "he_decrypt"
    HE_ROTATION = "he_rotation"
    HE_KEYSWITCH = "he_keyswitch"
    HE_RESCALE = "he_rescale"

    # Service health
    HEALTH_CHECK = "health_check"
    ADAPTER_LOAD = "adapter_load"
    ADAPTER_UNLOAD = "adapter_unload"


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_type: TelemetryEventType
    timestamp: float
    request_id: Optional[str] = None
    adapter_id: Optional[str] = None

    # Event-specific data
    duration_us: int = 0
    token_idx: int = -1
    layer_idx: int = -1
    projection_type: str = ""

    # HE operation counts
    rotations: int = 0
    keyswitches: int = 0
    rescales: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """Aggregated metrics for a single request."""
    request_id: str
    adapter_id: str
    start_time: float
    end_time: Optional[float] = None

    # Token counts
    prefill_tokens: int = 0
    decode_tokens: int = 0

    # Timing
    total_time_ms: float = 0.0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    he_time_ms: float = 0.0

    # HE operation totals
    total_rotations: int = 0
    total_keyswitches: int = 0
    total_rescales: int = 0
    total_encryptions: int = 0
    total_decryptions: int = 0

    # Timing breakdowns
    total_encrypt_time_us: int = 0
    total_compute_time_us: int = 0
    total_decrypt_time_us: int = 0

    # Error tracking
    errors: List[str] = field(default_factory=list)


class ServiceTelemetryCollector:
    """
    Collects and aggregates telemetry from MSS and HAS services.

    Thread-safe collector that can be used from multiple request handlers.
    Supports real-time metrics, aggregation, and export.
    """

    def __init__(
        self,
        max_events: int = 100000,
        aggregation_interval_s: float = 10.0,
    ):
        """
        Initialize collector.

        Args:
            max_events: Maximum events to retain
            aggregation_interval_s: Interval for periodic aggregation
        """
        self._max_events = max_events
        self._aggregation_interval = aggregation_interval_s

        # Thread safety
        self._lock = threading.RLock()

        # Event storage
        self._events: List[TelemetryEvent] = []
        self._event_idx = 0

        # Per-request metrics
        self._request_metrics: Dict[str, RequestMetrics] = {}

        # Aggregated metrics
        self._aggregated: Dict[str, Any] = {}
        self._last_aggregation = time.time()

        # Callbacks for real-time monitoring
        self._callbacks: List[Callable[[TelemetryEvent], None]] = []

        # Running statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0

    def record_event(self, event: TelemetryEvent) -> None:
        """
        Record a telemetry event.

        Args:
            event: Event to record
        """
        with self._lock:
            # Store event
            if len(self._events) < self._max_events:
                self._events.append(event)
            else:
                # Ring buffer behavior
                self._events[self._event_idx % self._max_events] = event
            self._event_idx += 1

            # Update request metrics
            if event.request_id:
                self._update_request_metrics(event)

            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Telemetry callback error: {e}")

    def _update_request_metrics(self, event: TelemetryEvent) -> None:
        """Update metrics for a request."""
        req_id = event.request_id

        if req_id not in self._request_metrics:
            self._request_metrics[req_id] = RequestMetrics(
                request_id=req_id,
                adapter_id=event.adapter_id or "",
                start_time=event.timestamp,
            )

        metrics = self._request_metrics[req_id]

        if event.event_type == TelemetryEventType.REQUEST_START:
            metrics.start_time = event.timestamp
            self._total_requests += 1

        elif event.event_type == TelemetryEventType.REQUEST_COMPLETE:
            metrics.end_time = event.timestamp
            metrics.total_time_ms = (event.timestamp - metrics.start_time) * 1000

        elif event.event_type == TelemetryEventType.REQUEST_ERROR:
            metrics.errors.append(event.metadata.get('error', 'unknown'))
            self._total_errors += 1

        elif event.event_type == TelemetryEventType.TOKEN_PREFILL:
            metrics.prefill_tokens += 1
            metrics.prefill_time_ms += event.duration_us / 1000
            self._total_tokens += 1

        elif event.event_type == TelemetryEventType.TOKEN_DECODE:
            metrics.decode_tokens += 1
            metrics.decode_time_ms += event.duration_us / 1000
            self._total_tokens += 1

        elif event.event_type == TelemetryEventType.HE_ENCRYPT:
            metrics.total_encryptions += 1
            metrics.total_encrypt_time_us += event.duration_us

        elif event.event_type == TelemetryEventType.HE_COMPUTE:
            metrics.total_compute_time_us += event.duration_us
            metrics.total_rotations += event.rotations
            metrics.total_keyswitches += event.keyswitches
            metrics.total_rescales += event.rescales

        elif event.event_type == TelemetryEventType.HE_DECRYPT:
            metrics.total_decryptions += 1
            metrics.total_decrypt_time_us += event.duration_us

        elif event.event_type == TelemetryEventType.HE_ROTATION:
            metrics.total_rotations += 1

        elif event.event_type == TelemetryEventType.HE_KEYSWITCH:
            metrics.total_keyswitches += 1

        elif event.event_type == TelemetryEventType.HE_RESCALE:
            metrics.total_rescales += 1

        # Update HE total time
        metrics.he_time_ms = (
            metrics.total_encrypt_time_us +
            metrics.total_compute_time_us +
            metrics.total_decrypt_time_us
        ) / 1000

    def start_request(
        self,
        request_id: str,
        adapter_id: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record request start."""
        self.record_event(TelemetryEvent(
            event_type=TelemetryEventType.REQUEST_START,
            timestamp=time.time(),
            request_id=request_id,
            adapter_id=adapter_id,
            metadata=metadata or {},
        ))

    def end_request(
        self,
        request_id: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record request completion."""
        event_type = TelemetryEventType.REQUEST_COMPLETE if success else TelemetryEventType.REQUEST_ERROR
        self.record_event(TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            request_id=request_id,
            metadata={'error': error} if error else {},
        ))

    def record_token(
        self,
        request_id: str,
        token_idx: int,
        is_prefill: bool,
        duration_us: int,
    ) -> None:
        """Record token processing."""
        event_type = TelemetryEventType.TOKEN_PREFILL if is_prefill else TelemetryEventType.TOKEN_DECODE
        self.record_event(TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            request_id=request_id,
            token_idx=token_idx,
            duration_us=duration_us,
        ))

    def record_he_operation(
        self,
        request_id: str,
        operation: str,
        layer_idx: int,
        projection_type: str,
        duration_us: int,
        rotations: int = 0,
        keyswitches: int = 0,
        rescales: int = 0,
    ) -> None:
        """Record HE operation."""
        op_map = {
            'encrypt': TelemetryEventType.HE_ENCRYPT,
            'compute': TelemetryEventType.HE_COMPUTE,
            'decrypt': TelemetryEventType.HE_DECRYPT,
            'rotation': TelemetryEventType.HE_ROTATION,
            'keyswitch': TelemetryEventType.HE_KEYSWITCH,
            'rescale': TelemetryEventType.HE_RESCALE,
        }

        event_type = op_map.get(operation, TelemetryEventType.HE_COMPUTE)
        self.record_event(TelemetryEvent(
            event_type=event_type,
            timestamp=time.time(),
            request_id=request_id,
            layer_idx=layer_idx,
            projection_type=projection_type,
            duration_us=duration_us,
            rotations=rotations,
            keyswitches=keyswitches,
            rescales=rescales,
        ))

    def get_request_metrics(self, request_id: str) -> Optional[RequestMetrics]:
        """Get metrics for a specific request."""
        with self._lock:
            return self._request_metrics.get(request_id)

    def get_aggregated_metrics(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get aggregated metrics across all requests.

        Args:
            force_refresh: Force recalculation even if recent

        Returns:
            Dict of aggregated metrics
        """
        with self._lock:
            now = time.time()
            if not force_refresh and (now - self._last_aggregation) < self._aggregation_interval:
                return self._aggregated

            # Compute aggregates
            completed_requests = [
                m for m in self._request_metrics.values()
                if m.end_time is not None
            ]

            if not completed_requests:
                return self._aggregated

            total_time = sum(m.total_time_ms for m in completed_requests)
            total_tokens = sum(m.prefill_tokens + m.decode_tokens for m in completed_requests)
            total_he_time = sum(m.he_time_ms for m in completed_requests)
            total_rotations = sum(m.total_rotations for m in completed_requests)
            total_keyswitches = sum(m.total_keyswitches for m in completed_requests)
            total_rescales = sum(m.total_rescales for m in completed_requests)

            self._aggregated = {
                # Request counts
                'total_requests': len(completed_requests),
                'total_errors': self._total_errors,
                'error_rate': self._total_errors / max(1, len(completed_requests)),

                # Throughput
                'total_tokens': total_tokens,
                'tokens_per_second': total_tokens / max(0.001, total_time / 1000),
                'requests_per_second': len(completed_requests) / max(0.001, total_time / 1000),

                # Latency
                'avg_request_time_ms': total_time / max(1, len(completed_requests)),
                'avg_time_per_token_ms': total_time / max(1, total_tokens),

                # HE metrics
                'avg_he_time_ms': total_he_time / max(1, len(completed_requests)),
                'he_time_percentage': (total_he_time / max(0.001, total_time)) * 100,
                'avg_rotations_per_token': total_rotations / max(1, total_tokens),
                'avg_keyswitches_per_token': total_keyswitches / max(1, total_tokens),
                'avg_rescales_per_token': total_rescales / max(1, total_tokens),

                # Totals
                'total_rotations': total_rotations,
                'total_keyswitches': total_keyswitches,
                'total_rescales': total_rescales,
            }

            self._last_aggregation = now
            return self._aggregated

    def add_callback(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Add a callback for real-time event processing."""
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Remove a callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def clear_request(self, request_id: str) -> None:
        """Clear metrics for a completed request."""
        with self._lock:
            if request_id in self._request_metrics:
                del self._request_metrics[request_id]

    def clear_all(self) -> None:
        """Clear all telemetry data."""
        with self._lock:
            self._events.clear()
            self._event_idx = 0
            self._request_metrics.clear()
            self._aggregated.clear()
            self._total_requests = 0
            self._total_tokens = 0
            self._total_errors = 0

    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[TelemetryEventType] = None,
    ) -> List[TelemetryEvent]:
        """Get recent events."""
        with self._lock:
            events = self._events[-count:] if count < len(self._events) else self._events[:]
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return events

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collector state."""
        with self._lock:
            return {
                'total_events': self._event_idx,
                'stored_events': len(self._events),
                'active_requests': len(self._request_metrics),
                'total_requests': self._total_requests,
                'total_tokens': self._total_tokens,
                'total_errors': self._total_errors,
                'callback_count': len(self._callbacks),
            }
