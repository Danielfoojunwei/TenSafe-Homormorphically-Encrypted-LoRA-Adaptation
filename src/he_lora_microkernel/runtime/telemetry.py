"""
Telemetry for HE-LoRA Microkernel

This module provides comprehensive telemetry for monitoring
HE-LoRA execution performance and validating rotation budgets.

Key metrics tracked:
  - Rotations per token (CRITICAL KPI)
  - Key switches per token
  - Rescales per token
  - HE time as percentage of total inference
  - Throughput (tokens/second)

All metrics are designed for CI enforcement of performance invariants.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time
import json
from collections import deque
import threading


# =============================================================================
# METRIC TYPES
# =============================================================================

class MetricType(Enum):
    """Types of metrics tracked."""
    COUNTER = "counter"      # Monotonically increasing
    GAUGE = "gauge"          # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution
    TIMER = "timer"          # Duration


@dataclass
class MetricValue:
    """Single metric value with timestamp."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


# =============================================================================
# TELEMETRY COLLECTOR
# =============================================================================

class TelemetryCollector:
    """
    Collects and aggregates telemetry from HE-LoRA execution.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        flush_callback: Optional[Callable[[List[MetricValue]], None]] = None,
    ):
        """
        Initialize telemetry collector.

        Args:
            buffer_size: Maximum metrics to buffer
            flush_callback: Optional callback when buffer is flushed
        """
        self._buffer: deque = deque(maxlen=buffer_size)
        self._flush_callback = flush_callback
        self._lock = threading.Lock()

        # Aggregated counters
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}

        # Per-token history (for averaging)
        self._token_history: deque = deque(maxlen=1000)

        # Session info
        self._session_start = time.time()
        self._tokens_processed = 0

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional tags for filtering
        """
        metric = MetricValue(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metric_type=metric_type,
        )

        with self._lock:
            self._buffer.append(metric)

            if metric_type == MetricType.COUNTER:
                self._counters[name] = self._counters.get(name, 0) + value
            elif metric_type == MetricType.GAUGE:
                self._gauges[name] = value

    def record_token_metrics(
        self,
        rotations: int,
        keyswitches: int,
        rescales: int,
        he_time_ms: float,
        total_time_ms: float,
    ) -> None:
        """
        Record metrics for a single token.

        Args:
            rotations: Rotation count
            keyswitches: Key switch count
            rescales: Rescale count
            he_time_ms: HE computation time
            total_time_ms: Total token time
        """
        self.record('rotations_per_token', rotations, MetricType.GAUGE)
        self.record('keyswitches_per_token', keyswitches, MetricType.GAUGE)
        self.record('rescales_per_token', rescales, MetricType.GAUGE)
        self.record('he_time_ms', he_time_ms, MetricType.TIMER)
        self.record('total_time_ms', total_time_ms, MetricType.TIMER)

        if total_time_ms > 0:
            he_percentage = (he_time_ms / total_time_ms) * 100
            self.record('he_time_percentage', he_percentage, MetricType.GAUGE)

        with self._lock:
            self._tokens_processed += 1
            self._token_history.append({
                'rotations': rotations,
                'keyswitches': keyswitches,
                'rescales': rescales,
                'he_time_ms': he_time_ms,
                'total_time_ms': total_time_ms,
            })

    def get_counter(self, name: str) -> float:
        """Get counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get gauge value."""
        with self._lock:
            return self._gauges.get(name, 0)

    def get_token_averages(self) -> Dict[str, float]:
        """Get average metrics per token."""
        with self._lock:
            if not self._token_history:
                return {}

            history = list(self._token_history)

        n = len(history)
        return {
            'avg_rotations_per_token': sum(h['rotations'] for h in history) / n,
            'avg_keyswitches_per_token': sum(h['keyswitches'] for h in history) / n,
            'avg_rescales_per_token': sum(h['rescales'] for h in history) / n,
            'avg_he_time_ms': sum(h['he_time_ms'] for h in history) / n,
            'avg_total_time_ms': sum(h['total_time_ms'] for h in history) / n,
            'token_count': n,
        }

    def get_throughput(self) -> Dict[str, float]:
        """Get throughput metrics."""
        with self._lock:
            elapsed = time.time() - self._session_start
            tokens = self._tokens_processed

        return {
            'tokens_processed': tokens,
            'elapsed_seconds': elapsed,
            'tokens_per_second': tokens / max(elapsed, 0.001),
            'ms_per_token': (elapsed * 1000) / max(tokens, 1),
        }

    def flush(self) -> List[MetricValue]:
        """Flush buffer and optionally call callback."""
        with self._lock:
            metrics = list(self._buffer)
            self._buffer.clear()

        if self._flush_callback:
            self._flush_callback(metrics)

        return metrics

    def reset(self) -> None:
        """Reset all telemetry."""
        with self._lock:
            self._buffer.clear()
            self._counters.clear()
            self._gauges.clear()
            self._token_history.clear()
            self._session_start = time.time()
            self._tokens_processed = 0


# =============================================================================
# PERFORMANCE REPORTER
# =============================================================================

class PerformanceReporter:
    """
    Generates performance reports for HE-LoRA execution.

    Reports are designed for:
      - CI validation
      - Benchmark documentation
      - Performance debugging
    """

    def __init__(self, collector: TelemetryCollector):
        """
        Initialize reporter.

        Args:
            collector: Telemetry collector
        """
        self._collector = collector

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns:
            Performance report dictionary
        """
        averages = self._collector.get_token_averages()
        throughput = self._collector.get_throughput()

        return {
            'summary': {
                'tokens_processed': throughput['tokens_processed'],
                'total_time_seconds': throughput['elapsed_seconds'],
                'tokens_per_second': throughput['tokens_per_second'],
            },
            'per_token_metrics': {
                'avg_rotations': averages.get('avg_rotations_per_token', 0),
                'avg_keyswitches': averages.get('avg_keyswitches_per_token', 0),
                'avg_rescales': averages.get('avg_rescales_per_token', 0),
                'avg_he_time_ms': averages.get('avg_he_time_ms', 0),
                'avg_total_time_ms': averages.get('avg_total_time_ms', 0),
            },
            'throughput': throughput,
            'timestamp': time.time(),
        }

    def generate_ci_report(
        self,
        rotation_budget: int,
        keyswitch_budget: int,
        rescale_budget: int,
    ) -> Dict[str, Any]:
        """
        Generate CI validation report.

        Args:
            rotation_budget: Maximum allowed rotations per token
            keyswitch_budget: Maximum allowed keyswitches per token
            rescale_budget: Maximum allowed rescales per token

        Returns:
            CI report with pass/fail status
        """
        averages = self._collector.get_token_averages()

        rotation_pass = averages.get('avg_rotations_per_token', 0) <= rotation_budget
        keyswitch_pass = averages.get('avg_keyswitches_per_token', 0) <= keyswitch_budget
        rescale_pass = averages.get('avg_rescales_per_token', 0) <= rescale_budget

        all_pass = rotation_pass and keyswitch_pass and rescale_pass

        return {
            'passed': all_pass,
            'checks': {
                'rotation_budget': {
                    'passed': rotation_pass,
                    'actual': averages.get('avg_rotations_per_token', 0),
                    'budget': rotation_budget,
                },
                'keyswitch_budget': {
                    'passed': keyswitch_pass,
                    'actual': averages.get('avg_keyswitches_per_token', 0),
                    'budget': keyswitch_budget,
                },
                'rescale_budget': {
                    'passed': rescale_pass,
                    'actual': averages.get('avg_rescales_per_token', 0),
                    'budget': rescale_budget,
                },
            },
            'metrics': averages,
        }

    def generate_benchmark_report(
        self,
        config_name: str,
        batch_sizes: List[int],
        hidden_sizes: List[int],
        ranks: List[int],
    ) -> Dict[str, Any]:
        """
        Generate benchmark report format.

        Args:
            config_name: Configuration name
            batch_sizes: Batch sizes tested
            hidden_sizes: Hidden sizes tested
            ranks: Ranks tested

        Returns:
            Benchmark report
        """
        base_report = self.generate_report()

        return {
            'config_name': config_name,
            'parameters': {
                'batch_sizes': batch_sizes,
                'hidden_sizes': hidden_sizes,
                'ranks': ranks,
            },
            'results': base_report,
            'format_version': '1.0',
        }

    def to_json(self) -> str:
        """Export report as JSON."""
        return json.dumps(self.generate_report(), indent=2)


# =============================================================================
# INVARIANT CHECKER
# =============================================================================

class InvariantChecker:
    """
    Checks performance invariants for CI enforcement.

    Invariants are hard constraints that must be satisfied.
    Failure causes CI to fail.
    """

    def __init__(
        self,
        max_rotations_per_token: int = 16,
        max_keyswitches_per_token: int = 16,
        max_rescales_per_token: int = 8,
        max_he_percentage: float = 95.0,
    ):
        """
        Initialize invariant checker.

        Args:
            max_rotations_per_token: R_max budget
            max_keyswitches_per_token: K_max budget
            max_rescales_per_token: S_max budget
            max_he_percentage: Maximum HE time as % of total
        """
        self.max_rotations = max_rotations_per_token
        self.max_keyswitches = max_keyswitches_per_token
        self.max_rescales = max_rescales_per_token
        self.max_he_percentage = max_he_percentage

        self._violations: List[str] = []

    def check_token(
        self,
        rotations: int,
        keyswitches: int,
        rescales: int,
        he_time_ms: float,
        total_time_ms: float,
    ) -> bool:
        """
        Check invariants for a single token.

        Args:
            rotations: Rotation count
            keyswitches: Key switch count
            rescales: Rescale count
            he_time_ms: HE time
            total_time_ms: Total time

        Returns:
            True if all invariants pass
        """
        passed = True

        if rotations > self.max_rotations:
            self._violations.append(
                f"Rotation invariant violated: {rotations} > {self.max_rotations}"
            )
            passed = False

        if keyswitches > self.max_keyswitches:
            self._violations.append(
                f"Keyswitch invariant violated: {keyswitches} > {self.max_keyswitches}"
            )
            passed = False

        if rescales > self.max_rescales:
            self._violations.append(
                f"Rescale invariant violated: {rescales} > {self.max_rescales}"
            )
            passed = False

        if total_time_ms > 0:
            he_percentage = (he_time_ms / total_time_ms) * 100
            if he_percentage > self.max_he_percentage:
                self._violations.append(
                    f"HE time invariant violated: {he_percentage:.1f}% > "
                    f"{self.max_he_percentage}%"
                )
                passed = False

        return passed

    def check_determinism(
        self,
        schedule_hash_1: str,
        schedule_hash_2: str,
    ) -> bool:
        """
        Check schedule determinism invariant.

        Args:
            schedule_hash_1: First compilation hash
            schedule_hash_2: Second compilation hash

        Returns:
            True if hashes match
        """
        if schedule_hash_1 != schedule_hash_2:
            self._violations.append(
                f"Determinism invariant violated: hashes differ "
                f"({schedule_hash_1[:8]}... vs {schedule_hash_2[:8]}...)"
            )
            return False
        return True

    @property
    def violations(self) -> List[str]:
        """Get list of violations."""
        return self._violations.copy()

    @property
    def has_violations(self) -> bool:
        """Check if any violations occurred."""
        return len(self._violations) > 0

    def reset(self) -> None:
        """Reset violation list."""
        self._violations.clear()

    def get_ci_result(self) -> Dict[str, Any]:
        """
        Get CI-formatted result.

        Returns:
            Dict with 'passed' and 'violations' keys
        """
        return {
            'passed': not self.has_violations,
            'violations': self.violations,
            'invariants': {
                'max_rotations_per_token': self.max_rotations,
                'max_keyswitches_per_token': self.max_keyswitches,
                'max_rescales_per_token': self.max_rescales,
                'max_he_percentage': self.max_he_percentage,
            },
        }


# =============================================================================
# GLOBAL TELEMETRY INSTANCE
# =============================================================================

_global_collector: Optional[TelemetryCollector] = None


def get_global_collector() -> TelemetryCollector:
    """Get or create global telemetry collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = TelemetryCollector()
    return _global_collector


def reset_global_collector() -> None:
    """Reset global telemetry collector."""
    global _global_collector
    if _global_collector is not None:
        _global_collector.reset()
