"""
Metrics Export

Exports metrics to monitoring systems (Prometheus, etc.)
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .collector import ServiceTelemetryCollector

logger = logging.getLogger(__name__)


class MetricsExporter(ABC):
    """Base class for metrics exporters."""

    @abstractmethod
    def export(self, metrics: Dict[str, Any]) -> None:
        """Export metrics."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start exporter."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop exporter."""
        pass


@dataclass
class MetricDefinition:
    """Definition of a metric to export."""
    name: str
    description: str
    metric_type: str  # "counter", "gauge", "histogram", "summary"
    labels: List[str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = []


class PrometheusExporter(MetricsExporter):
    """
    Export metrics in Prometheus format.

    Exposes an HTTP endpoint for Prometheus scraping.
    """

    # Standard HE-LoRA metrics
    METRICS = [
        MetricDefinition(
            name="helora_requests_total",
            description="Total number of requests",
            metric_type="counter",
            labels=["status"],
        ),
        MetricDefinition(
            name="helora_tokens_total",
            description="Total number of tokens processed",
            metric_type="counter",
            labels=["type"],
        ),
        MetricDefinition(
            name="helora_request_duration_seconds",
            description="Request duration in seconds",
            metric_type="histogram",
            labels=["adapter_id"],
        ),
        MetricDefinition(
            name="helora_token_duration_seconds",
            description="Token processing duration in seconds",
            metric_type="histogram",
            labels=["type"],
        ),
        MetricDefinition(
            name="helora_he_operations_total",
            description="Total HE operations",
            metric_type="counter",
            labels=["operation"],
        ),
        MetricDefinition(
            name="helora_rotations_per_token",
            description="Rotations per token",
            metric_type="gauge",
        ),
        MetricDefinition(
            name="helora_keyswitches_per_token",
            description="Keyswitches per token",
            metric_type="gauge",
        ),
        MetricDefinition(
            name="helora_rescales_per_token",
            description="Rescales per token",
            metric_type="gauge",
        ),
        MetricDefinition(
            name="helora_tokens_per_second",
            description="Throughput in tokens per second",
            metric_type="gauge",
        ),
        MetricDefinition(
            name="helora_he_time_ratio",
            description="Ratio of time spent on HE operations",
            metric_type="gauge",
        ),
        MetricDefinition(
            name="helora_active_requests",
            description="Number of active requests",
            metric_type="gauge",
        ),
        MetricDefinition(
            name="helora_loaded_adapters",
            description="Number of loaded adapters",
            metric_type="gauge",
        ),
        MetricDefinition(
            name="helora_kpi_violations_total",
            description="Total KPI violations",
            metric_type="counter",
            labels=["kpi", "severity"],
        ),
    ]

    def __init__(
        self,
        collector: ServiceTelemetryCollector,
        port: int = 9090,
        path: str = "/metrics",
    ):
        """
        Initialize Prometheus exporter.

        Args:
            collector: Telemetry collector
            port: HTTP port for metrics endpoint
            path: URL path for metrics
        """
        self._collector = collector
        self._port = port
        self._path = path

        # Metric values
        self._counters: Dict[str, Dict[str, float]] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

        # HTTP server
        self._server = None
        self._server_thread = None

        # Initialize counters
        for metric in self.METRICS:
            if metric.metric_type == "counter":
                self._counters[metric.name] = {}
            elif metric.metric_type == "histogram":
                self._histograms[metric.name] = []

    def start(self) -> None:
        """Start the HTTP server for Prometheus scraping."""
        try:
            from http.server import BaseHTTPRequestHandler, HTTPServer
        except ImportError:
            logger.warning("HTTP server not available")
            return

        exporter = self

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == exporter._path:
                    metrics = exporter._format_metrics()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(metrics.encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        try:
            self._server = HTTPServer(('', self._port), MetricsHandler)
            self._server_thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
            )
            self._server_thread.start()
            logger.info(f"Prometheus metrics available at http://localhost:{self._port}{self._path}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._server_thread:
            self._server_thread.join(timeout=5)
            self._server_thread = None

    def export(self, metrics: Dict[str, Any]) -> None:
        """Update metrics from aggregated data."""
        # Update gauges
        self._gauges['helora_tokens_per_second'] = metrics.get('tokens_per_second', 0)
        self._gauges['helora_rotations_per_token'] = metrics.get('avg_rotations_per_token', 0)
        self._gauges['helora_keyswitches_per_token'] = metrics.get('avg_keyswitches_per_token', 0)
        self._gauges['helora_rescales_per_token'] = metrics.get('avg_rescales_per_token', 0)
        self._gauges['helora_he_time_ratio'] = metrics.get('he_time_percentage', 0) / 100

        # Update counters incrementally
        total_requests = metrics.get('total_requests', 0)
        if 'success' not in self._counters.get('helora_requests_total', {}):
            self._counters['helora_requests_total'] = {'success': 0, 'error': 0}

        total_tokens = metrics.get('total_tokens', 0)
        if 'prefill' not in self._counters.get('helora_tokens_total', {}):
            self._counters['helora_tokens_total'] = {'prefill': 0, 'decode': 0}

    def _format_metrics(self) -> str:
        """Format metrics in Prometheus text format."""
        # Refresh from collector
        aggregated = self._collector.get_aggregated_metrics(force_refresh=True)
        self.export(aggregated)

        summary = self._collector.get_summary()

        lines = []

        # Helper to format metric
        def format_metric(name: str, help_text: str, metric_type: str, value: float, labels: Dict[str, str] = None):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            if labels:
                label_str = ','.join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        # Format gauges
        for name, value in self._gauges.items():
            metric_def = next((m for m in self.METRICS if m.name == name), None)
            if metric_def:
                format_metric(name, metric_def.description, "gauge", value)

        # Format counters
        for name, label_values in self._counters.items():
            metric_def = next((m for m in self.METRICS if m.name == name), None)
            if metric_def:
                lines.append(f"# HELP {name} {metric_def.description}")
                lines.append(f"# TYPE {name} counter")
                for label_value, count in label_values.items():
                    label_name = metric_def.labels[0] if metric_def.labels else "label"
                    lines.append(f'{name}{{{label_name}="{label_value}"}} {count}')

        # Add summary metrics
        format_metric(
            "helora_active_requests",
            "Number of active requests",
            "gauge",
            summary.get('active_requests', 0),
        )

        format_metric(
            "helora_total_events",
            "Total telemetry events",
            "counter",
            summary.get('total_events', 0),
        )

        return '\n'.join(lines) + '\n'

    def get_metrics_text(self) -> str:
        """Get metrics as text (for testing)."""
        return self._format_metrics()


class JSONExporter(MetricsExporter):
    """Export metrics as JSON (for logging or custom endpoints)."""

    def __init__(self, collector: ServiceTelemetryCollector):
        self._collector = collector
        self._last_export: Dict[str, Any] = {}

    def start(self) -> None:
        """No-op for JSON exporter."""
        pass

    def stop(self) -> None:
        """No-op for JSON exporter."""
        pass

    def export(self, metrics: Dict[str, Any]) -> None:
        """Store metrics for retrieval."""
        self._last_export = {
            'timestamp': time.time(),
            'metrics': metrics,
            'summary': self._collector.get_summary(),
        }

    def get_json(self) -> Dict[str, Any]:
        """Get latest metrics as JSON."""
        return self._last_export


class LoggingExporter(MetricsExporter):
    """Export metrics to logging."""

    def __init__(
        self,
        collector: ServiceTelemetryCollector,
        interval_seconds: float = 60.0,
        log_level: int = logging.INFO,
    ):
        self._collector = collector
        self._interval = interval_seconds
        self._log_level = log_level
        self._timer: Optional[threading.Timer] = None
        self._running = False

    def start(self) -> None:
        """Start periodic logging."""
        self._running = True
        self._schedule_next()

    def stop(self) -> None:
        """Stop periodic logging."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _schedule_next(self) -> None:
        """Schedule next log."""
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._log_metrics)
        self._timer.daemon = True
        self._timer.start()

    def _log_metrics(self) -> None:
        """Log current metrics."""
        metrics = self._collector.get_aggregated_metrics(force_refresh=True)
        self.export(metrics)
        self._schedule_next()

    def export(self, metrics: Dict[str, Any]) -> None:
        """Log metrics."""
        logger.log(
            self._log_level,
            f"HE-LoRA Metrics: "
            f"requests={metrics.get('total_requests', 0)}, "
            f"tokens={metrics.get('total_tokens', 0)}, "
            f"tok/s={metrics.get('tokens_per_second', 0):.2f}, "
            f"rot/tok={metrics.get('avg_rotations_per_token', 0):.2f}, "
            f"HE%={metrics.get('he_time_percentage', 0):.1f}"
        )
