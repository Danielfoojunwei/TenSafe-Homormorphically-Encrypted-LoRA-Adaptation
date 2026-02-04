#!/usr/bin/env python3
"""
Differential Privacy Training with TenSafe

This example demonstrates how to train a LoRA adapter with differential
privacy (DP) guarantees using DP-SGD. Differential privacy provides
mathematical guarantees about what an adversary can learn about any
individual training example.

Key concepts:
- Epsilon (epsilon): Privacy budget - lower is more private
- Delta (delta): Probability of privacy breach
- Noise multiplier: Amount of noise added to gradients
- Max grad norm: Gradient clipping threshold

What this example demonstrates:
- Configuring DP-SGD parameters
- Training with privacy accounting
- Monitoring privacy budget consumption
- Understanding the privacy-utility tradeoff

Prerequisites:
- TenSafe server running
- API key configured

Expected Output:
    DP Configuration:
      Target epsilon: 8.0
      Target delta: 1e-5
      Noise multiplier: 1.0

    Training with DP-SGD...
    Step 10: loss=2.1, epsilon=0.5, budget_used=6.25%
    Step 20: loss=1.9, epsilon=1.1, budget_used=13.75%
    ...

    Final privacy guarantee: (epsilon=8.0, delta=1e-5)-DP
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    """Demonstrate differential privacy training with TenSafe."""

    from tg_tinker import ServiceClient
    from tg_tinker.schemas import (
        TrainingConfig, LoRAConfig, OptimizerConfig, DPConfig
    )

    # =========================================================================
    # Step 1: Understanding Differential Privacy
    # =========================================================================
    print("=" * 60)
    print("DIFFERENTIAL PRIVACY TRAINING")
    print("=" * 60)
    print("""
    Differential Privacy (DP) provides mathematical privacy guarantees:

    An (epsilon, delta)-DP algorithm ensures that for any individual training
    example, the probability of any output changes by at most exp(epsilon)
    whether or not that example is included, with probability 1-delta.

    Key parameters:
    - epsilon: Privacy budget (typical: 1-10, lower = more private)
    - delta: Should be << 1/dataset_size (e.g., 1e-5 for 100k samples)
    - noise_multiplier: Controls noise added to gradients
    - max_grad_norm: Clips per-sample gradients
    """)

    # =========================================================================
    # Step 2: Configure DP parameters
    # =========================================================================
    print("\nConfiguring differential privacy...")

    dp_config = DPConfig(
        enabled=True,

        # Privacy budget
        target_epsilon=8.0,    # Target epsilon for entire training
        target_delta=1e-5,     # Should be << 1/num_samples

        # DP-SGD parameters
        noise_multiplier=1.0,  # Noise scale (higher = more private, less utility)
        max_grad_norm=1.0,     # Per-sample gradient clipping threshold

        # Accounting method
        accounting="rdp",      # Renyi Differential Privacy (tighter bounds)
    )

    print(f"  Target epsilon: {dp_config.target_epsilon}")
    print(f"  Target delta: {dp_config.target_delta}")
    print(f"  Noise multiplier: {dp_config.noise_multiplier}")
    print(f"  Max gradient norm: {dp_config.max_grad_norm}")

    # =========================================================================
    # Step 3: Configure training with DP
    # =========================================================================
    print("\nConfiguring training parameters...")

    lora_config = LoRAConfig(
        rank=16,
        alpha=32,
        target_modules=["q_proj", "v_proj"],
    )

    training_config = TrainingConfig(
        model_ref="meta-llama/Llama-3-8B",
        lora_config=lora_config,
        optimizer=OptimizerConfig(name="adamw", lr=1e-4),
        dp_config=dp_config,
        batch_size=32,         # Larger batches are more efficient for DP
        max_steps=1000,
    )

    # =========================================================================
    # Step 4: Initialize and train
    # =========================================================================
    print("\nInitializing training client...")

    try:
        client = ServiceClient(
            base_url=os.environ.get("TG_TINKER_BASE_URL", "http://localhost:8000"),
            api_key=os.environ.get("TG_TINKER_API_KEY", "demo-api-key"),
        )

        tc = client.create_training_client(training_config)
        print(f"Training client created: {tc.training_client_id}")

        # Training loop would go here
        # ...

        # Get DP metrics
        dp_metrics = tc.get_dp_metrics()
        print(f"\nPrivacy consumed: epsilon={dp_metrics.epsilon_spent:.2f}")

        client.close()

    except Exception as e:
        print(f"Note: Server connection failed ({e})")
        print("Running in demonstration mode...")
        demonstrate_dp_training()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("PRIVACY-UTILITY TRADEOFF")
    print("=" * 60)
    print("""
    Tips for better privacy-utility tradeoff:

    1. Use larger batch sizes
       - More samples per batch = less noise per sample
       - Recommended: 256-2048 for DP training

    2. Tune noise multiplier
       - Higher noise = more private, less utility
       - Start with 1.0 and adjust based on results

    3. Use RDP accounting
       - Tighter bounds than naive composition
       - Already enabled by default

    4. Consider group privacy
       - If users have multiple samples, adjust epsilon accordingly

    5. Monitor privacy budget
       - Stop training before exceeding target epsilon
       - Consider early stopping if loss plateaus
    """)


def demonstrate_dp_training():
    """Demonstrate DP training simulation."""
    print("\n[Demo Mode] Simulating DP training...")

    # Simulated training with privacy accounting
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    losses = [2.1, 1.9, 1.8, 1.7, 1.65, 1.6, 1.55, 1.52, 1.5, 1.48]
    epsilons = [0.5, 1.1, 1.8, 2.5, 3.2, 4.0, 4.8, 5.7, 6.6, 7.5]

    print("\nTraining progress:")
    print("-" * 60)

    for step, loss, eps in zip(steps, losses, epsilons):
        budget_pct = (eps / 8.0) * 100
        print(f"  Step {step:3d}: loss={loss:.2f}, epsilon={eps:.2f}, budget_used={budget_pct:.1f}%")

    print("-" * 60)
    print("\n[Demo Mode] Final privacy guarantee: (epsilon=7.5, delta=1e-5)-DP")

    print("""
    The trained model satisfies (7.5, 1e-5)-differential privacy.
    This means that for any training example, an adversary who sees
    the trained model cannot determine whether that specific example
    was in the training set, with high confidence.
    """)


if __name__ == "__main__":
    main()
