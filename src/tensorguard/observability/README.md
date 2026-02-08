# TenSafe Observability Stack

The observability component provides a unified, privacy-aware monitoring foundation using OpenTelemetry (OTEL).

## Component Overview

- **`setup_observability`**: Global initialization for tracing, metrics, and logging.
- **Privacy redaction**: Middleware that automatically identifies and masks sensitive fields (API keys, tokens, session IDs) in traces and logs.
- **TenSafe Metrics**: Specialized meters for DP budget consumption, gradient norms, and HE operation overhead.

## Key Features

1. **Distributed Tracing**: End-to-end request tracing from Client SDK to Backend and HE Microkernel.
2. **Privacy Redaction Middleware**: Hook into FastAPI/Starlette to sanitize HTTP headers and request bodies before they reach the OTEL collector.
3. **Structured Context Logging**: Correlation IDs that link logs, spans, and metrics to specific training or inference requests.

## Usage

```python
from tensorguard.observability import setup_observability

setup_observability(
    service_name="tensafe-api-prod",
    otlp_endpoint="http://otel-collector:4317",
    redact_sensitive_data=True
)
```

## Instrumented Fields

- `dp.epsilon_spent`: Gauge for consumed epsilon.
- `he.op_count`: Counter for encrypted matrix operations.
- `tssp.verify_status`: Boolean status of package verification.
