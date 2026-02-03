"""
Privacy Budget Tracking Example

Demonstrates monitoring and managing differential privacy budget
consumption during training.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python dp_budget_tracking.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        dp_config={
            "enabled": True,
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
        },
    )

    # Training with budget monitoring
    budget_limit = 8.0
    warning_threshold = 0.8 * budget_limit

    sample_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[2, 3, 4, 5, 6]] * 8}

    print(f"Privacy budget: {budget_limit} epsilon")
    print(f"Warning threshold: {warning_threshold}")
    print("-" * 40)

    step = 0
    while True:
        tc.forward_backward(batch=sample_batch).result()
        tc.optim_step(apply_dp_noise=True).result()
        step += 1

        metrics = tc.get_dp_metrics()
        current_epsilon = metrics["total_epsilon"]

        if step % 20 == 0:
            remaining = budget_limit - current_epsilon
            pct = (current_epsilon / budget_limit) * 100
            print(f"Step {step}: ε={current_epsilon:.4f} ({pct:.1f}% used), remaining={remaining:.4f}")

        if current_epsilon >= warning_threshold and current_epsilon < budget_limit:
            print(f"WARNING: Approaching privacy budget limit!")

        if current_epsilon >= budget_limit:
            print(f"STOP: Privacy budget exhausted at step {step}")
            break

    print(f"\nFinal: ε={metrics['total_epsilon']:.4f}, δ={metrics['delta']:.2e}")


if __name__ == "__main__":
    main()
