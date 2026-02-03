"""
Basic Differential Privacy Training

Demonstrates training with differential privacy (DP-SGD) guarantees.

This example shows how to:
1. Configure differential privacy parameters
2. Train a model with DP-SGD
3. Monitor privacy budget consumption
4. Verify privacy guarantees

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python dp_training_basic.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Create training client with differential privacy
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={
            "rank": 16,
            "alpha": 32.0,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        dp_config={
            "enabled": True,
            # Noise multiplier: higher = more privacy, less utility
            "noise_multiplier": 1.0,
            # Maximum gradient norm for clipping
            "max_grad_norm": 1.0,
            # Target privacy budget
            "target_epsilon": 8.0,
            "target_delta": 1e-5,
            # Privacy accountant type
            "accountant_type": "rdp",  # Rényi Differential Privacy
        },
        batch_size=8,
    )

    print("Training client created with DP-SGD")
    print(f"  Target epsilon: 8.0")
    print(f"  Target delta: 1e-5")
    print()

    # Simulate training data (in practice, use real data)
    sample_batch = {
        "input_ids": [[1, 2, 3, 4, 5]] * 8,
        "attention_mask": [[1, 1, 1, 1, 1]] * 8,
        "labels": [[2, 3, 4, 5, 6]] * 8,
    }

    # Training loop with DP
    num_steps = 100
    print(f"Training for {num_steps} steps with DP-SGD...")
    print("-" * 50)

    for step in range(num_steps):
        # Forward-backward pass
        future = tc.forward_backward(batch=sample_batch)
        fb_result = future.result()

        # Optimizer step with DP noise
        future = tc.optim_step(apply_dp_noise=True)
        opt_result = future.result()

        # Check privacy budget periodically
        if (step + 1) % 20 == 0:
            metrics = tc.get_dp_metrics()
            print(f"Step {step + 1}:")
            print(f"  Epsilon consumed: {metrics['total_epsilon']:.4f}")
            print(f"  Delta: {metrics['delta']:.2e}")
            print(f"  Privacy budget remaining: {8.0 - metrics['total_epsilon']:.4f}")

            # Check if we've exceeded our budget
            if metrics['total_epsilon'] > 8.0:
                print("WARNING: Privacy budget exceeded!")
                break

    # Final privacy report
    print("\n" + "=" * 50)
    print("FINAL PRIVACY REPORT")
    print("=" * 50)
    final_metrics = tc.get_dp_metrics()
    print(f"Total epsilon: {final_metrics['total_epsilon']:.4f}")
    print(f"Delta: {final_metrics['delta']:.2e}")
    print(f"Training steps: {final_metrics['num_steps']}")
    print(f"Privacy guarantee: (ε={final_metrics['total_epsilon']:.2f}, δ={final_metrics['delta']:.0e})-DP")


if __name__ == "__main__":
    main()
