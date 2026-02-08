# Observability Guide

**Version**: 4.1.0
**Last Updated**: 2026-02-08

This guide explains how to monitor, trace, and debug the TenSafe platform using OpenTelemetry (OTEL) and integrated privacy metrics.

## Getting Started

TenSafe is instrumented with OpenTelemetry by default. To start collecting data, you need an OTEL Collector endpoint (e.g., Jaeger for traces, Prometheus for metrics).

```python
from tensorguard.observability import setup_observability

setup_observability(
    service_name="tensafe-compute-node",
    otlp_endpoint="http://otel-collector:4317",
    enable_tracing=True,
    enable_metrics=True
)
```

## Privacy-Aware Metrics

TenSafe exposes specialized metrics to track the state and overhead of privacy-preserving operations.

| Metric | Type | Description |
| :--- | :--- | :--- |
| `dp.epsilon_spent` | Gauge | Cumulative privacy budget consumed (ε). |
| `dp.delta_spent` | Gauge | Cumulative δ spent. |
| `he.op_latency` | Histogram | Latency (ms) of encrypted matrix multiplications. |
| `he.rotation_count` | Counter | Number of HE rotations executed (should be 0 in MOAI mode). |
| `tssp.verify_latency` | Histogram | Time taken to verify TSSP package signatures. |

### Visualizing Privacy Budget
Use the provided Grafana dashboard (`/dashboards/tensafe-privacy.json`) to visualize the ε-vs-Loss curve in real-time.

---

## Distributed Tracing

TenSafe automatically propagates trace contexts across service boundaries (Client SDK -> Gateway -> Ray Worker -> vLLM Engine).

### Identifying HE Bottlenecks
In your tracing UI (Jaeger/Zipkin), look for spans with the `he.` prefix. These spans provide deep visibility into the HE-LoRA microkernel:
- `he.load_weights`: Time to decrypt and load LoRA ranks into memory.
- `he.encrypt_input`: Time to pack and encrypt hidden states.
- `he.forward_pass`: Time for encrypted matrix multiplication.

---

## v4.1 Feature: Sensitive Data Redaction

In version 4.1, TenSafe introduces a **Privacy-First Redaction Middleware**.

- **Automatic Masking**: The OTEL exporter automatically identifies and masks sensitive fields like `api_key`, `tssp_secret`, and `user_id` in trace attributes and log messages.
- **Header Sanitization**: HTTP headers (Authorization, Cookie, etc.) are stripped before being logged or exported to OTEL.

---

## Logging

TenSafe uses structured JSON logging. This allows you to correlate logs with traces using the `trace_id` and `span_id` fields.

```json
{
  "timestamp": "2026-02-08T12:00:00Z",
  "level": "INFO",
  "message": "Enforcing Zero-Rotation MOAI contract",
  "trace_id": "5b2a...",
  "span_id": "c1d2...",
  "component": "he_microkernel"
}
```

## Related Documentation

- [MLOps Guide](mlops.md) - For higher-level experiment tracking (W&B, MLflow).
- [vLLM Guide](vllm-integration.md) - Integrating OTEL with serving endpoints.
