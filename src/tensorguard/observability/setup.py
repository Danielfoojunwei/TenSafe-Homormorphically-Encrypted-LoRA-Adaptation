"""OpenTelemetry Setup for TenSafe.

Configures comprehensive observability stack with privacy-aware telemetry.
"""

from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import logging
import os
import time
import threading

logger = logging.getLogger(__name__)

# Conditional OpenTelemetry imports
OTEL_AVAILABLE = False
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    logger.warning("OpenTelemetry not installed. Install with: pip install opentelemetry-sdk")

# Optional exporters
OTLP_AVAILABLE = False
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except ImportError:
    pass

PROMETHEUS_AVAILABLE = False
try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import start_http_server, Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ObservabilityConfig:
    """Configuration for observability stack."""
    service_name: str = "tensafe"
    service_version: str = "3.0.0"
    environment: str = "development"

    # Tracing
    tracing_enabled: bool = True
    tracing_endpoint: str = "http://localhost:4317"
    tracing_sampling_rate: float = 1.0

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    # OTLP
    otlp_endpoint: Optional[str] = None

    # Privacy
    redact_sensitive_data: bool = True
    sensitive_fields: list = field(default_factory=lambda: [
        "password", "token", "api_key", "secret", "credential",
        "ssn", "credit_card", "private_key"
    ])


class TenSafeMetrics:
    """TenSafe-specific metrics collection.

    Provides both OpenTelemetry and Prometheus-compatible metrics for:
    - Inference latency and throughput
    - Training progress and privacy budget
    - HE-LoRA computation overhead
    - System resource utilization
    """

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize metrics collector.

        Args:
            config: Observability configuration
        """
        self.config = config or ObservabilityConfig()
        self._lock = threading.Lock()

        # Internal metric storage (fallback when OTEL not available)
        self._counters: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        self._gauges: Dict[str, float] = {}

        # OpenTelemetry meter
        self._meter = None
        self._otel_metrics: Dict[str, Any] = {}

        # Prometheus metrics
        self._prom_metrics: Dict[str, Any] = {}

        self._setup_metrics()

    def _setup_metrics(self):
        """Setup metric instruments."""
        if OTEL_AVAILABLE:
            self._meter = metrics.get_meter("tensafe")
            self._setup_otel_metrics()

        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()

    def _setup_otel_metrics(self):
        """Setup OpenTelemetry metrics."""
        if not self._meter:
            return

        # Inference metrics
        self._otel_metrics["inference_latency"] = self._meter.create_histogram(
            name="tensafe_inference_latency_seconds",
            description="Inference latency in seconds",
            unit="s",
        )

        self._otel_metrics["inference_tokens"] = self._meter.create_counter(
            name="tensafe_inference_tokens_total",
            description="Total tokens generated",
            unit="tokens",
        )

        self._otel_metrics["inference_requests"] = self._meter.create_counter(
            name="tensafe_inference_requests_total",
            description="Total inference requests",
            unit="requests",
        )

        # Training metrics
        self._otel_metrics["training_loss"] = self._meter.create_histogram(
            name="tensafe_training_loss",
            description="Training loss values",
        )

        self._otel_metrics["training_steps"] = self._meter.create_counter(
            name="tensafe_training_steps_total",
            description="Total training steps",
        )

        # Privacy metrics
        self._otel_metrics["privacy_epsilon"] = self._meter.create_histogram(
            name="tensafe_privacy_epsilon",
            description="Privacy budget epsilon",
        )

        # HE-LoRA metrics
        self._otel_metrics["he_lora_latency"] = self._meter.create_histogram(
            name="tensafe_he_lora_latency_seconds",
            description="HE-LoRA computation latency",
            unit="s",
        )

        self._otel_metrics["he_lora_operations"] = self._meter.create_counter(
            name="tensafe_he_lora_operations_total",
            description="Total HE-LoRA operations",
        )

        # Queue metrics
        self._otel_metrics["request_queue_depth"] = self._meter.create_up_down_counter(
            name="tensafe_request_queue_depth",
            description="Current request queue depth",
        )

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # Inference
        self._prom_metrics["inference_latency"] = Histogram(
            "tensafe_inference_latency_seconds",
            "Inference latency in seconds",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        self._prom_metrics["inference_tokens"] = Counter(
            "tensafe_inference_tokens_total",
            "Total tokens generated",
        )

        self._prom_metrics["inference_requests"] = Counter(
            "tensafe_inference_requests_total",
            "Total inference requests",
            ["status"],
        )

        # Training
        self._prom_metrics["training_loss"] = Histogram(
            "tensafe_training_loss",
            "Training loss values",
        )

        # Privacy
        self._prom_metrics["privacy_epsilon"] = Gauge(
            "tensafe_privacy_epsilon",
            "Current privacy budget epsilon",
            ["training_id"],
        )

        self._prom_metrics["privacy_delta"] = Gauge(
            "tensafe_privacy_delta",
            "Current privacy budget delta",
            ["training_id"],
        )

        # HE-LoRA
        self._prom_metrics["he_lora_latency"] = Histogram(
            "tensafe_he_lora_latency_seconds",
            "HE-LoRA computation latency",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
        )

        self._prom_metrics["he_lora_operations"] = Counter(
            "tensafe_he_lora_operations_total",
            "Total HE-LoRA operations",
            ["layer"],
        )

        # Queue
        self._prom_metrics["request_queue_depth"] = Gauge(
            "tensafe_request_queue_depth",
            "Current request queue depth",
        )

        # TTFT
        self._prom_metrics["time_to_first_token"] = Histogram(
            "tensafe_time_to_first_token_seconds",
            "Time to first token",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

    def record_inference_latency(self, latency_seconds: float, **attributes):
        """Record inference latency."""
        with self._lock:
            if "inference_latency" not in self._histograms:
                self._histograms["inference_latency"] = []
            self._histograms["inference_latency"].append(latency_seconds)

        if OTEL_AVAILABLE and "inference_latency" in self._otel_metrics:
            self._otel_metrics["inference_latency"].record(latency_seconds, attributes)

        if PROMETHEUS_AVAILABLE and "inference_latency" in self._prom_metrics:
            self._prom_metrics["inference_latency"].observe(latency_seconds)

    def record_inference_tokens(self, count: int, **attributes):
        """Record tokens generated."""
        with self._lock:
            self._counters["inference_tokens"] = self._counters.get("inference_tokens", 0) + count

        if OTEL_AVAILABLE and "inference_tokens" in self._otel_metrics:
            self._otel_metrics["inference_tokens"].add(count, attributes)

        if PROMETHEUS_AVAILABLE and "inference_tokens" in self._prom_metrics:
            self._prom_metrics["inference_tokens"].inc(count)

    def record_inference_request(self, status: str = "success", **attributes):
        """Record inference request."""
        with self._lock:
            key = f"inference_requests_{status}"
            self._counters[key] = self._counters.get(key, 0) + 1

        if OTEL_AVAILABLE and "inference_requests" in self._otel_metrics:
            self._otel_metrics["inference_requests"].add(1, {"status": status, **attributes})

        if PROMETHEUS_AVAILABLE and "inference_requests" in self._prom_metrics:
            self._prom_metrics["inference_requests"].labels(status=status).inc()

    def record_training_loss(self, loss: float, **attributes):
        """Record training loss."""
        with self._lock:
            if "training_loss" not in self._histograms:
                self._histograms["training_loss"] = []
            self._histograms["training_loss"].append(loss)

        if OTEL_AVAILABLE and "training_loss" in self._otel_metrics:
            self._otel_metrics["training_loss"].record(loss, attributes)

        if PROMETHEUS_AVAILABLE and "training_loss" in self._prom_metrics:
            self._prom_metrics["training_loss"].observe(loss)

    def record_privacy_budget(self, epsilon: float, delta: float, training_id: str = "default"):
        """Record privacy budget consumption."""
        with self._lock:
            self._gauges[f"privacy_epsilon_{training_id}"] = epsilon
            self._gauges[f"privacy_delta_{training_id}"] = delta

        if PROMETHEUS_AVAILABLE:
            if "privacy_epsilon" in self._prom_metrics:
                self._prom_metrics["privacy_epsilon"].labels(training_id=training_id).set(epsilon)
            if "privacy_delta" in self._prom_metrics:
                self._prom_metrics["privacy_delta"].labels(training_id=training_id).set(delta)

    def record_he_lora_latency(self, latency_seconds: float, layer: str = "default"):
        """Record HE-LoRA computation latency."""
        with self._lock:
            if "he_lora_latency" not in self._histograms:
                self._histograms["he_lora_latency"] = []
            self._histograms["he_lora_latency"].append(latency_seconds)

        if OTEL_AVAILABLE and "he_lora_latency" in self._otel_metrics:
            self._otel_metrics["he_lora_latency"].record(latency_seconds, {"layer": layer})

        if PROMETHEUS_AVAILABLE and "he_lora_latency" in self._prom_metrics:
            self._prom_metrics["he_lora_latency"].observe(latency_seconds)

    def record_he_lora_operation(self, layer: str = "default"):
        """Record HE-LoRA operation."""
        with self._lock:
            key = f"he_lora_operations_{layer}"
            self._counters[key] = self._counters.get(key, 0) + 1

        if OTEL_AVAILABLE and "he_lora_operations" in self._otel_metrics:
            self._otel_metrics["he_lora_operations"].add(1, {"layer": layer})

        if PROMETHEUS_AVAILABLE and "he_lora_operations" in self._prom_metrics:
            self._prom_metrics["he_lora_operations"].labels(layer=layer).inc()

    def set_queue_depth(self, depth: int):
        """Set current queue depth."""
        with self._lock:
            self._gauges["request_queue_depth"] = depth

        if PROMETHEUS_AVAILABLE and "request_queue_depth" in self._prom_metrics:
            self._prom_metrics["request_queue_depth"].set(depth)

    def record_ttft(self, seconds: float):
        """Record time to first token."""
        with self._lock:
            if "ttft" not in self._histograms:
                self._histograms["ttft"] = []
            self._histograms["ttft"].append(seconds)

        if PROMETHEUS_AVAILABLE and "time_to_first_token" in self._prom_metrics:
            self._prom_metrics["time_to_first_token"].observe(seconds)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            summary = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }

            for name, values in self._histograms.items():
                if values:
                    summary["histograms"][name] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                    }

            return summary


def setup_observability(
    app=None,
    config: Optional[ObservabilityConfig] = None,
) -> tuple:
    """Initialize OpenTelemetry observability stack.

    Args:
        app: Optional FastAPI app to instrument
        config: Observability configuration

    Returns:
        Tuple of (tracer_provider, meter_provider, metrics)
    """
    config = config or ObservabilityConfig()

    logger.info(f"Setting up observability for {config.service_name}")

    tracer_provider = None
    meter_provider = None

    if OTEL_AVAILABLE:
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: config.service_name,
            SERVICE_VERSION: config.service_version,
            "deployment.environment": config.environment,
        })

        # Setup tracing
        if config.tracing_enabled:
            tracer_provider = TracerProvider(resource=resource)

            if OTLP_AVAILABLE and config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=config.otlp_endpoint,
                    insecure=True,
                )
                tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

            trace.set_tracer_provider(tracer_provider)
            logger.info("Tracing enabled")

        # Setup metrics
        if config.metrics_enabled:
            readers = []

            if PROMETHEUS_AVAILABLE:
                readers.append(PrometheusMetricReader())

            if OTLP_AVAILABLE and config.otlp_endpoint:
                otlp_metric_exporter = OTLPMetricExporter(
                    endpoint=config.otlp_endpoint,
                    insecure=True,
                )
                readers.append(PeriodicExportingMetricReader(otlp_metric_exporter))

            if readers:
                meter_provider = MeterProvider(resource=resource, metric_readers=readers)
                metrics.set_meter_provider(meter_provider)

            logger.info("Metrics enabled")

    # Start Prometheus metrics server
    if PROMETHEUS_AVAILABLE and config.metrics_enabled:
        try:
            start_http_server(config.metrics_port)
            logger.info(f"Prometheus metrics server started on port {config.metrics_port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")

    # Instrument FastAPI if provided
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI instrumentation enabled")
        except ImportError:
            logger.warning("FastAPI instrumentation not available")

    # Create metrics collector
    tensafe_metrics = TenSafeMetrics(config)

    return tracer_provider, meter_provider, tensafe_metrics
