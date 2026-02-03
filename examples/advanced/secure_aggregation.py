"""
Secure Gradient Aggregation Example

Demonstrates secure multi-party computation for gradient aggregation.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python secure_aggregation.py
"""

import os
from tensafe import TenSafeClient
from tensafe.crypto import SecureMPCConfig


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Configure secure aggregation
    mpc_config = SecureMPCConfig(
        protocol="pairwise_masking",  # Efficient pairwise masking
        num_parties=4,
        threshold=3,  # Need 3 of 4 parties
        dropout_tolerance=0.25,  # Tolerate up to 25% dropout
    )

    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={"rank": 16, "alpha": 32.0},
        dp_config={"enabled": True, "target_epsilon": 8.0},
        secure_aggregation_config=mpc_config,
    )

    print("Secure Aggregation Configuration:")
    print(f"  Protocol: {mpc_config.protocol}")
    print(f"  Parties: {mpc_config.num_parties}")
    print(f"  Threshold: {mpc_config.threshold}")
    print()

    # Training with secure aggregation
    sample_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[2, 3, 4, 5, 6]] * 8}

    for step in range(50):
        fb = tc.forward_backward(batch=sample_batch).result()
        # Gradients are securely aggregated across parties
        opt = tc.optim_step(apply_dp_noise=True, secure_aggregate=True).result()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: loss={fb.get('loss', 0):.4f}, secure_agg={opt.get('secure_aggregation_used', False)}")

    print("\nSecure aggregation training complete!")


if __name__ == "__main__":
    main()
