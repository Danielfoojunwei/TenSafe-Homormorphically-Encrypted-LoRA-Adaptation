"""
KPI Definition and Enforcement

Defines and enforces Key Performance Indicators for HE-LoRA services.
Includes rotation budgets, latency SLAs, and operational limits.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .collector import ServiceTelemetryCollector, TelemetryEvent, TelemetryEventType

logger = logging.getLogger(__name__)


class KPISeverity(str, Enum):
    """Severity level for KPI violations."""
    INFO = "info"         # Informational, no action needed
    WARNING = "warning"   # Worth noting, may need attention
    ERROR = "error"       # Significant issue, needs attention
    CRITICAL = "critical" # Service-impacting, immediate attention


class KPICategory(str, Enum):
    """Category of KPI."""
    ROTATION = "rotation"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR = "error"
    RESOURCE = "resource"


@dataclass
class KPIDefinition:
    """
    Definition of a Key Performance Indicator.

    Attributes:
        name: KPI identifier
        category: KPI category
        description: Human-readable description
        threshold: Threshold value
        comparison: How to compare ("lt", "gt", "le", "ge", "eq")
        severity: Severity when violated
        per_token: Whether this is a per-token metric
    """
    name: str
    category: KPICategory
    description: str
    threshold: float
    comparison: str = "le"  # less than or equal by default
    severity: KPISeverity = KPISeverity.WARNING
    per_token: bool = True
    enabled: bool = True

    def check(self, value: float) -> bool:
        """
        Check if value satisfies the KPI.

        Returns:
            True if KPI is satisfied, False if violated
        """
        if self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "le":
            return value <= self.threshold
        elif self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "ge":
            return value >= self.threshold
        elif self.comparison == "eq":
            return value == self.threshold
        return True


@dataclass
class KPIViolation:
    """Record of a KPI violation."""
    kpi_name: str
    category: KPICategory
    severity: KPISeverity
    threshold: float
    actual_value: float
    timestamp: float
    request_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ServiceKPIs:
    """
    Standard KPI definitions for HE-LoRA services.

    Defines rotation budgets, latency targets, and operational limits
    aligned with the HE-LoRA microkernel's MOAI-inspired design.
    """

    # Rotation budget KPIs (MOAI-aligned)
    ROTATIONS_PER_TOKEN = KPIDefinition(
        name="rotations_per_token",
        category=KPICategory.ROTATION,
        description="Maximum rotations per token (MOAI budget)",
        threshold=16.0,
        comparison="le",
        severity=KPISeverity.ERROR,
        per_token=True,
    )

    KEYSWITCHES_PER_TOKEN = KPIDefinition(
        name="keyswitches_per_token",
        category=KPICategory.ROTATION,
        description="Maximum keyswitches per token",
        threshold=16.0,
        comparison="le",
        severity=KPISeverity.ERROR,
        per_token=True,
    )

    RESCALES_PER_TOKEN = KPIDefinition(
        name="rescales_per_token",
        category=KPICategory.ROTATION,
        description="Maximum rescales per token",
        threshold=8.0,
        comparison="le",
        severity=KPISeverity.WARNING,
        per_token=True,
    )

    # Latency KPIs
    TOKEN_LATENCY_MS = KPIDefinition(
        name="token_latency_ms",
        category=KPICategory.LATENCY,
        description="Maximum latency per token in milliseconds",
        threshold=100.0,  # 100ms per token
        comparison="le",
        severity=KPISeverity.WARNING,
        per_token=True,
    )

    REQUEST_LATENCY_MS = KPIDefinition(
        name="request_latency_ms",
        category=KPICategory.LATENCY,
        description="Maximum request latency in milliseconds",
        threshold=30000.0,  # 30 seconds
        comparison="le",
        severity=KPISeverity.WARNING,
        per_token=False,
    )

    HE_TIME_PERCENTAGE = KPIDefinition(
        name="he_time_percentage",
        category=KPICategory.LATENCY,
        description="Maximum percentage of time spent on HE operations",
        threshold=80.0,  # 80%
        comparison="le",
        severity=KPISeverity.WARNING,
        per_token=False,
    )

    # Throughput KPIs
    MIN_TOKENS_PER_SECOND = KPIDefinition(
        name="min_tokens_per_second",
        category=KPICategory.THROUGHPUT,
        description="Minimum tokens per second throughput",
        threshold=1.0,  # At least 1 tok/s
        comparison="ge",
        severity=KPISeverity.ERROR,
        per_token=False,
    )

    # Error KPIs
    ERROR_RATE = KPIDefinition(
        name="error_rate",
        category=KPICategory.ERROR,
        description="Maximum error rate",
        threshold=0.05,  # 5%
        comparison="le",
        severity=KPISeverity.CRITICAL,
        per_token=False,
    )

    @classmethod
    def get_all_kpis(cls) -> List[KPIDefinition]:
        """Get all KPI definitions."""
        return [
            cls.ROTATIONS_PER_TOKEN,
            cls.KEYSWITCHES_PER_TOKEN,
            cls.RESCALES_PER_TOKEN,
            cls.TOKEN_LATENCY_MS,
            cls.REQUEST_LATENCY_MS,
            cls.HE_TIME_PERCENTAGE,
            cls.MIN_TOKENS_PER_SECOND,
            cls.ERROR_RATE,
        ]

    @classmethod
    def get_rotation_kpis(cls) -> List[KPIDefinition]:
        """Get rotation budget KPIs."""
        return [
            cls.ROTATIONS_PER_TOKEN,
            cls.KEYSWITCHES_PER_TOKEN,
            cls.RESCALES_PER_TOKEN,
        ]


class KPIEnforcer:
    """
    Enforces KPIs and tracks violations.

    Integrates with ServiceTelemetryCollector to check KPIs
    in real-time and record violations.
    """

    def __init__(
        self,
        collector: Optional[ServiceTelemetryCollector] = None,
        kpis: Optional[List[KPIDefinition]] = None,
        on_violation: Optional[Callable[[KPIViolation], None]] = None,
    ):
        """
        Initialize enforcer.

        Args:
            collector: Telemetry collector to use
            kpis: KPIs to enforce (default: all)
            on_violation: Callback for violations
        """
        self._collector = collector
        self._kpis = {kpi.name: kpi for kpi in (kpis or ServiceKPIs.get_all_kpis())}
        self._on_violation = on_violation

        # Violation tracking
        self._violations: List[KPIViolation] = []
        self._violation_counts: Dict[str, int] = {}

        # Per-request tracking
        self._request_state: Dict[str, Dict[str, float]] = {}

        # Register callback if collector provided
        if self._collector:
            self._collector.add_callback(self._on_telemetry_event)

    def _on_telemetry_event(self, event: TelemetryEvent) -> None:
        """Process telemetry event and check KPIs."""
        if event.request_id:
            self._update_request_state(event)

        # Check rotation KPIs on HE operations
        if event.event_type == TelemetryEventType.HE_COMPUTE:
            self._check_rotation_kpis(event)

        # Check latency on token completion
        if event.event_type in (TelemetryEventType.TOKEN_PREFILL, TelemetryEventType.TOKEN_DECODE):
            self._check_token_latency(event)

        # Check request-level KPIs on completion
        if event.event_type == TelemetryEventType.REQUEST_COMPLETE:
            self._check_request_kpis(event.request_id)

    def _update_request_state(self, event: TelemetryEvent) -> None:
        """Update per-request state."""
        req_id = event.request_id
        if req_id not in self._request_state:
            self._request_state[req_id] = {
                'tokens': 0,
                'rotations': 0,
                'keyswitches': 0,
                'rescales': 0,
                'total_time_us': 0,
                'he_time_us': 0,
            }

        state = self._request_state[req_id]

        if event.event_type in (TelemetryEventType.TOKEN_PREFILL, TelemetryEventType.TOKEN_DECODE):
            state['tokens'] += 1
            state['total_time_us'] += event.duration_us

        if event.event_type == TelemetryEventType.HE_COMPUTE:
            state['rotations'] += event.rotations
            state['keyswitches'] += event.keyswitches
            state['rescales'] += event.rescales
            state['he_time_us'] += event.duration_us

        if event.event_type in (TelemetryEventType.HE_ENCRYPT, TelemetryEventType.HE_DECRYPT):
            state['he_time_us'] += event.duration_us

    def _check_rotation_kpis(self, event: TelemetryEvent) -> None:
        """Check rotation budget KPIs."""
        kpi = self._kpis.get('rotations_per_token')
        if kpi and kpi.enabled and event.rotations > 0:
            if not kpi.check(event.rotations):
                self._record_violation(KPIViolation(
                    kpi_name=kpi.name,
                    category=kpi.category,
                    severity=kpi.severity,
                    threshold=kpi.threshold,
                    actual_value=event.rotations,
                    timestamp=event.timestamp,
                    request_id=event.request_id,
                    details={
                        'layer_idx': event.layer_idx,
                        'projection_type': event.projection_type,
                    },
                ))

        kpi = self._kpis.get('keyswitches_per_token')
        if kpi and kpi.enabled and event.keyswitches > 0:
            if not kpi.check(event.keyswitches):
                self._record_violation(KPIViolation(
                    kpi_name=kpi.name,
                    category=kpi.category,
                    severity=kpi.severity,
                    threshold=kpi.threshold,
                    actual_value=event.keyswitches,
                    timestamp=event.timestamp,
                    request_id=event.request_id,
                ))

    def _check_token_latency(self, event: TelemetryEvent) -> None:
        """Check token latency KPI."""
        kpi = self._kpis.get('token_latency_ms')
        if kpi and kpi.enabled:
            latency_ms = event.duration_us / 1000
            if not kpi.check(latency_ms):
                self._record_violation(KPIViolation(
                    kpi_name=kpi.name,
                    category=kpi.category,
                    severity=kpi.severity,
                    threshold=kpi.threshold,
                    actual_value=latency_ms,
                    timestamp=event.timestamp,
                    request_id=event.request_id,
                    details={'token_idx': event.token_idx},
                ))

    def _check_request_kpis(self, request_id: Optional[str]) -> None:
        """Check request-level KPIs on completion."""
        if not request_id or request_id not in self._request_state:
            return

        state = self._request_state[request_id]
        tokens = max(1, state['tokens'])

        # Check average rotations per token
        kpi = self._kpis.get('rotations_per_token')
        if kpi and kpi.enabled:
            avg_rotations = state['rotations'] / tokens
            if not kpi.check(avg_rotations):
                self._record_violation(KPIViolation(
                    kpi_name=f"{kpi.name}_avg",
                    category=kpi.category,
                    severity=kpi.severity,
                    threshold=kpi.threshold,
                    actual_value=avg_rotations,
                    timestamp=time.time(),
                    request_id=request_id,
                ))

        # Check HE time percentage
        kpi = self._kpis.get('he_time_percentage')
        if kpi and kpi.enabled and state['total_time_us'] > 0:
            he_percentage = (state['he_time_us'] / state['total_time_us']) * 100
            if not kpi.check(he_percentage):
                self._record_violation(KPIViolation(
                    kpi_name=kpi.name,
                    category=kpi.category,
                    severity=kpi.severity,
                    threshold=kpi.threshold,
                    actual_value=he_percentage,
                    timestamp=time.time(),
                    request_id=request_id,
                ))

        # Clean up request state
        del self._request_state[request_id]

    def _record_violation(self, violation: KPIViolation) -> None:
        """Record a KPI violation."""
        self._violations.append(violation)
        self._violation_counts[violation.kpi_name] = (
            self._violation_counts.get(violation.kpi_name, 0) + 1
        )

        # Log violation
        log_msg = (
            f"KPI violation: {violation.kpi_name} "
            f"({violation.actual_value:.2f} vs {violation.threshold:.2f})"
        )
        if violation.severity == KPISeverity.CRITICAL:
            logger.error(log_msg)
        elif violation.severity == KPISeverity.ERROR:
            logger.error(log_msg)
        elif violation.severity == KPISeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Trigger callback
        if self._on_violation:
            try:
                self._on_violation(violation)
            except Exception as e:
                logger.warning(f"Violation callback error: {e}")

    def check_value(self, kpi_name: str, value: float, request_id: Optional[str] = None) -> bool:
        """
        Manually check a KPI value.

        Args:
            kpi_name: Name of KPI to check
            value: Value to check
            request_id: Optional request ID for context

        Returns:
            True if KPI is satisfied
        """
        kpi = self._kpis.get(kpi_name)
        if kpi is None or not kpi.enabled:
            return True

        if kpi.check(value):
            return True

        self._record_violation(KPIViolation(
            kpi_name=kpi_name,
            category=kpi.category,
            severity=kpi.severity,
            threshold=kpi.threshold,
            actual_value=value,
            timestamp=time.time(),
            request_id=request_id,
        ))
        return False

    def get_violations(
        self,
        category: Optional[KPICategory] = None,
        severity: Optional[KPISeverity] = None,
        since: Optional[float] = None,
    ) -> List[KPIViolation]:
        """Get recorded violations with optional filtering."""
        violations = self._violations

        if category:
            violations = [v for v in violations if v.category == category]
        if severity:
            violations = [v for v in violations if v.severity == severity]
        if since:
            violations = [v for v in violations if v.timestamp >= since]

        return violations

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of violations."""
        return {
            'total_violations': len(self._violations),
            'by_kpi': dict(self._violation_counts),
            'by_severity': {
                sev.value: len([v for v in self._violations if v.severity == sev])
                for sev in KPISeverity
            },
            'by_category': {
                cat.value: len([v for v in self._violations if v.category == cat])
                for cat in KPICategory
            },
        }

    def clear_violations(self) -> None:
        """Clear all recorded violations."""
        self._violations.clear()
        self._violation_counts.clear()

    def update_kpi(self, name: str, threshold: Optional[float] = None, enabled: Optional[bool] = None) -> bool:
        """Update KPI threshold or enable state."""
        if name not in self._kpis:
            return False

        kpi = self._kpis[name]
        if threshold is not None:
            kpi.threshold = threshold
        if enabled is not None:
            kpi.enabled = enabled

        return True

    def get_kpi(self, name: str) -> Optional[KPIDefinition]:
        """Get KPI definition by name."""
        return self._kpis.get(name)

    def get_all_kpis(self) -> Dict[str, KPIDefinition]:
        """Get all KPI definitions."""
        return dict(self._kpis)
