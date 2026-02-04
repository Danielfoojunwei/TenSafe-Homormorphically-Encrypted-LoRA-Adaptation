"""
Prometheus Metrics Export Example

Demonstrates exporting TenSafe metrics to Prometheus for monitoring.

Requirements:
- TenSafe account and API key
- Prometheus server

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python prometheus_metrics.py
"""

import os
from tensafe import TenSafeClient
from tensafe.observability import PrometheusExporter


def main():
    # Initialize Prometheus exporter
    exporter = PrometheusExporter(port=9090)
    exporter.start()

    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={"rank": 16, "alpha": 32.0},
        dp_config={"enabled": True, "target_epsilon": 8.0},
        observability={"prometheus": True},
    )

    print("Metrics available at http://localhost:9090/metrics")
    print()
    print("Training with metrics export...")

    sample_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[2, 3, 4, 5, 6]] * 8}

    for step in range(100):
        fb = tc.forward_backward(batch=sample_batch).result()
        opt = tc.optim_step(apply_dp_noise=True).result()

        # Metrics automatically exported
        if (step + 1) % 20 == 0:
            print(f"Step {step+1}: loss={fb.get('loss', 0):.4f}")

    print("\nExported metrics:")
    print("  tensafe_training_loss")
    print("  tensafe_dp_epsilon_total")
    print("  tensafe_forward_latency_seconds")
    print("  tensafe_gradient_norm")


if __name__ == "__main__":
    main()
