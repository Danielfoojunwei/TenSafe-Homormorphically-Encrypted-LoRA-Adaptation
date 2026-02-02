# Observability Guide

**Version**: 4.0.0
**Last Updated**: 2026-02-02

This guide covers setting up comprehensive observability for TenSafe using OpenTelemetry.

## Overview

TenSafe's observability stack provides:
- **Distributed Tracing**: Request tracing with Jaeger/Tempo
- **Metrics**: Privacy-aware metrics with Prometheus
- **Logging**: Structured JSON logs with sensitive data redaction
- **Privacy-Specific Metrics**: DP epsilon, gradient norms, HE operations

## Prerequisites

```bash
# Install TenSafe with observability support
pip install tensafe[observability]>=4.0.0

# Or install dependencies separately
pip install opentelemetry-sdk>=1.20.0
pip install opentelemetry-exporter-otlp>=1.20.0
pip install opentelemetry-instrumentation-fastapi>=0.40b0
pip install prometheus-client>=0.17.0
```

## Quick Start

### Basic Setup

```python
from tensorguard.observability import setup_observability

# Initialize observability
setup_observability(
    service_name="tensafe-api",
    otlp_endpoint="http://otel-collector:4317",
    enable_metrics=True,
    enable_tracing=True,
    enable_logging=True,
)

# Now all TenSafe operations are automatically instrumented
```

### With FastAPI

```python
from fastapi import FastAPI
from tensorguard.observability import setup_observability
from tensorguard.observability.middleware import TenSafeTracingMiddleware

# Initialize observability
tracer, meter, logger = setup_observability(
    service_name="tensafe-api",
    otlp_endpoint="http://otel-collector:4317",
)

# Create FastAPI app
app = FastAPI()

# Add tracing middleware
app.add_middleware(
    TenSafeTracingMiddleware,
    service_name="tensafe-api",
    redact_sensitive=True,
)

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

## Configuration

### setup_observability()

```python
def setup_observability(
    service_name: str,                    # Service name for spans/metrics
    otlp_endpoint: Optional[str] = None,  # OTEL collector endpoint
    otlp_headers: Dict[str, str] = None,  # Auth headers

    # Exporters
    enable_metrics: bool = True,          # Enable metrics
    enable_tracing: bool = True,          # Enable tracing
    enable_logging: bool = True,          # Enable logging

    # Metrics
    prometheus_port: int = 9090,          # Prometheus scrape port
    metric_export_interval: int = 60,     # Export interval (seconds)

    # Tracing
    trace_sample_rate: float = 1.0,       # Sampling rate (1.0 = all)

    # Logging
    log_level: str = "INFO",              # Log level
    log_format: str = "json",             # "json" or "text"

    # Privacy
    redact_sensitive_data: bool = True,   # Auto-redact secrets
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_SERVICE_NAME` | Service name | `tensafe` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector endpoint | - |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers | - |
| `OTEL_TRACES_SAMPLER` | Sampler type | `parentbased_always_on` |
| `OTEL_TRACES_SAMPLER_ARG` | Sample rate | `1.0` |
| `TENSAFE_LOG_LEVEL` | Log level | `INFO` |
| `TENSAFE_LOG_FORMAT` | Log format | `json` |

## Distributed Tracing

### Automatic Instrumentation

TenSafe automatically instruments:
- HTTP requests (FastAPI/Starlette)
- Database operations (SQLAlchemy)
- Redis operations
- HE computations
- DP-SGD steps

```python
# All operations automatically traced
from tensorguard.backends.vllm import TenSafeAsyncEngine

engine = TenSafeAsyncEngine(config)

# This generates spans for:
# - HE-LoRA forward pass
# - Token generation
# - KV cache operations
async for output in engine.generate(prompt, request_id):
    print(output.text)
```

### Manual Spans

```python
from tensorguard.observability.middleware import create_span_decorator

# Create decorator
trace_function = create_span_decorator("tensafe-custom")

@trace_function("my_operation", attributes={"custom.key": "value"})
def my_function(x, y):
    return x + y

# Or use context manager
from opentelemetry import trace

tracer = trace.get_tracer("tensafe-custom")

def process_batch(batch):
    with tracer.start_as_current_span("process_batch") as span:
        span.set_attribute("batch.size", len(batch))
        result = do_processing(batch)
        span.set_attribute("result.success", True)
        return result
```

### Span Attributes

Standard TenSafe span attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `tensafe.operation` | string | Operation type |
| `tensafe.model` | string | Model name |
| `dp.epsilon_spent` | float | Privacy budget used |
| `dp.noise_multiplier` | float | Noise multiplier |
| `he.operation_count` | int | HE operations in span |
| `training.step` | int | Training step |
| `training.loss` | float | Training loss |

### Context Propagation

```python
# Headers automatically propagate trace context
# Client
import httpx
from opentelemetry.propagate import inject

headers = {}
inject(headers)

response = httpx.post(
    "http://tensafe-api:8000/v1/train",
    headers=headers,
    json=data,
)

# Server (automatic with middleware)
# Trace context extracted from incoming headers
```

## Metrics

### TenSafe Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `tensafe_dp_epsilon_spent` | Gauge | Total privacy budget consumed |
| `tensafe_dp_epsilon_per_step` | Histogram | Epsilon per training step |
| `tensafe_training_loss` | Histogram | Training loss distribution |
| `tensafe_training_steps_total` | Counter | Total training steps |
| `tensafe_inference_latency_seconds` | Histogram | Inference latency |
| `tensafe_inference_tokens_generated` | Counter | Tokens generated |
| `tensafe_he_operations_total` | Counter | HE operations count |
| `tensafe_he_operation_latency_seconds` | Histogram | HE operation latency |
| `tensafe_gradient_norm` | Histogram | Gradient norm after clipping |
| `tensafe_model_load_seconds` | Histogram | Model load time |

### Custom Metrics

```python
from opentelemetry import metrics

meter = metrics.get_meter("tensafe-custom")

# Counter
request_counter = meter.create_counter(
    "custom_requests_total",
    description="Total custom requests",
)
request_counter.add(1, {"endpoint": "/v1/train"})

# Histogram
latency_histogram = meter.create_histogram(
    "custom_latency_seconds",
    description="Custom operation latency",
)
latency_histogram.record(0.5, {"operation": "encrypt"})

# Gauge (using callback)
def get_queue_depth():
    return len(job_queue)

meter.create_observable_gauge(
    "custom_queue_depth",
    callbacks=[lambda: [metrics.Observation(get_queue_depth())]],
    description="Current queue depth",
)
```

### Prometheus Scraping

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tensafe'
    static_configs:
      - targets: ['tensafe-api:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

```python
# Expose metrics endpoint
from prometheus_client import make_wsgi_app
from starlette.middleware.wsgi import WSGIMiddleware

app.mount("/metrics", WSGIMiddleware(make_wsgi_app()))
```

## Logging

### Structured Logging

```python
import logging
from tensorguard.observability import setup_observability

setup_observability(
    service_name="tensafe",
    log_format="json",
    log_level="INFO",
)

logger = logging.getLogger("tensafe")

# Structured log output
logger.info(
    "Training step completed",
    extra={
        "step": 100,
        "loss": 0.5,
        "epsilon_spent": 0.1,
    }
)
# Output: {"timestamp": "...", "level": "INFO", "message": "Training step completed", "step": 100, "loss": 0.5, "epsilon_spent": 0.1}
```

### Sensitive Data Redaction

```python
# Automatic redaction of sensitive fields
logger.info(
    "API request",
    extra={
        "endpoint": "/v1/train",
        "api_key": "sk_live_xxx",  # Automatically redacted
        "password": "secret123",   # Automatically redacted
        "user_id": "user123",      # Not redacted
    }
)
# Output: {"endpoint": "/v1/train", "api_key": "[REDACTED]", "password": "[REDACTED]", "user_id": "user123"}
```

### Log Correlation

```python
# Logs automatically include trace context
from opentelemetry import trace

tracer = trace.get_tracer("tensafe")

with tracer.start_as_current_span("my_operation") as span:
    logger.info("Operation started")
    # Log includes: {"trace_id": "abc123", "span_id": "def456", ...}
```

## Tracing Middleware

### TenSafeTracingMiddleware

```python
from tensorguard.observability.middleware import TenSafeTracingMiddleware

app.add_middleware(
    TenSafeTracingMiddleware,
    service_name="tensafe-api",
    redact_sensitive=True,
    exclude_paths=["/health", "/readiness", "/metrics"],
)
```

Features:
- Automatic span creation for each request
- HTTP method, URL, status code attributes
- Request/response header capture (with redaction)
- Error tracking and exception recording
- Correlation ID propagation

### Privacy-Aware Attributes

```python
# Sensitive patterns automatically detected
SENSITIVE_PATTERNS = [
    "password",
    "token",
    "api_key",
    "secret",
    "credential",
    "authorization",
]

# Headers matching these patterns are redacted
# span.set_attribute("http.header.authorization", "[REDACTED]")
```

## Backend Configuration

### OpenTelemetry Collector

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024

  # Filter sensitive attributes
  attributes:
    actions:
      - key: http.header.authorization
        action: delete
      - key: api_key
        action: delete

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  prometheus:
    endpoint: 0.0.0.0:8889

  loki:
    endpoint: http://loki:3100/loki/api/v1/push

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, attributes]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [batch, attributes]
      exporters: [loki]
```

### Jaeger Setup

```yaml
# docker-compose.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:1.50
    ports:
      - "16686:16686"  # UI
      - "14250:14250"  # gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

### Grafana Dashboards

```json
// tensafe-dashboard.json
{
  "title": "TenSafe Observability",
  "panels": [
    {
      "title": "Privacy Budget (Epsilon)",
      "type": "gauge",
      "targets": [
        {
          "expr": "tensafe_dp_epsilon_spent",
          "legendFormat": "ε spent"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "max": 10,
          "thresholds": {
            "steps": [
              {"color": "green", "value": 0},
              {"color": "yellow", "value": 5},
              {"color": "red", "value": 8}
            ]
          }
        }
      }
    },
    {
      "title": "Inference Latency P95",
      "type": "timeseries",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(tensafe_inference_latency_seconds_bucket[5m]))",
          "legendFormat": "P95"
        }
      ]
    },
    {
      "title": "Training Loss",
      "type": "timeseries",
      "targets": [
        {
          "expr": "tensafe_training_loss",
          "legendFormat": "Loss"
        }
      ]
    }
  ]
}
```

## Alerting

### Prometheus Alerts

```yaml
# alerts.yml
groups:
  - name: tensafe
    rules:
      - alert: HighPrivacyBudgetUsage
        expr: tensafe_dp_epsilon_spent > 7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High privacy budget usage"
          description: "Epsilon spent is {{ $value }}, approaching limit"

      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, rate(tensafe_inference_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "P95 latency is {{ $value }}s"

      - alert: TrainingStalled
        expr: increase(tensafe_training_steps_total[10m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Training stalled"
          description: "No training steps in the last 10 minutes"
```

## Performance Considerations

### Sampling

```python
# Reduce trace volume with sampling
setup_observability(
    service_name="tensafe",
    trace_sample_rate=0.1,  # Sample 10% of traces
)
```

### Batch Export

```python
# Configure batch export
from opentelemetry.sdk.trace.export import BatchSpanProcessor

processor = BatchSpanProcessor(
    exporter,
    max_queue_size=2048,
    max_export_batch_size=512,
    schedule_delay_millis=5000,
)
```

### Metric Cardinality

```python
# Avoid high-cardinality labels
# Bad: user_id as label (high cardinality)
meter.create_counter("requests", {"user_id": user_id})

# Good: use bounded labels
meter.create_counter("requests", {"endpoint": endpoint, "status": status_code})
```

## Troubleshooting

### Debug Logging

```python
import logging

# Enable OTEL debug logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
logging.getLogger("tensorguard.observability").setLevel(logging.DEBUG)
```

### Verify Exports

```bash
# Check OTLP endpoint
curl -v http://otel-collector:4318/v1/traces

# Check Prometheus metrics
curl http://tensafe:9090/metrics | grep tensafe

# Check Jaeger traces
curl http://jaeger:16686/api/traces?service=tensafe
```

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [kubernetes.md](kubernetes.md) - Kubernetes deployment with monitoring
- [mlops.md](mlops.md) - MLOps integration for experiment tracking
