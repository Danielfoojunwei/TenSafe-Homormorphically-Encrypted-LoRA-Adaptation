"""
DP Hyperparameter Tuning Example

Demonstrates tuning differential privacy hyperparameters for optimal privacy-utility tradeoff.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python dp_hyperparameters.py
"""

import os
from tensafe import TenSafeClient
from tensafe.privacy import compute_dp_sgd_privacy


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Analyze privacy-utility tradeoffs
    print("DP Hyperparameter Analysis")
    print("=" * 60)

    configs = [
        {"noise_multiplier": 0.5, "max_grad_norm": 1.0, "epochs": 10},
        {"noise_multiplier": 1.0, "max_grad_norm": 1.0, "epochs": 10},
        {"noise_multiplier": 1.5, "max_grad_norm": 1.0, "epochs": 10},
        {"noise_multiplier": 1.0, "max_grad_norm": 0.5, "epochs": 10},
        {"noise_multiplier": 1.0, "max_grad_norm": 2.0, "epochs": 10},
    ]

    sample_size = 10000
    batch_size = 64
    delta = 1e-5

    for config in configs:
        epsilon = compute_dp_sgd_privacy(
            n=sample_size,
            batch_size=batch_size,
            noise_multiplier=config["noise_multiplier"],
            epochs=config["epochs"],
            delta=delta,
        )
        print(f"noise={config['noise_multiplier']}, clip={config['max_grad_norm']}, epochs={config['epochs']}")
        print(f"  -> (ε={epsilon:.2f}, δ={delta:.0e})-DP")
        print()

    # Train with optimal config
    print("Training with optimal config (ε≈8)...")
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        dp_config={
            "enabled": True,
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
        },
    )

    sample_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 64, "attention_mask": [[1] * 5] * 64, "labels": [[2, 3, 4, 5, 6]] * 64}

    for step in range(50):
        tc.forward_backward(batch=sample_batch).result()
        tc.optim_step(apply_dp_noise=True).result()

    final = tc.get_dp_metrics()
    print(f"\nFinal privacy: (ε={final['total_epsilon']:.4f}, δ={final['delta']:.0e})-DP")


if __name__ == "__main__":
    main()
