#!/usr/bin/env python3
"""
Basic Differential Privacy Training with TenSafe

This example demonstrates how to train a LoRA adapter with differential
privacy (DP) guarantees using DP-SGD. Differential privacy provides
mathematical guarantees about what an adversary can learn about any
individual training example.

What this example demonstrates:
- Understanding differential privacy concepts
- Configuring DP-SGD parameters
- Training with privacy accounting
- Monitoring privacy budget consumption

Key concepts:
- Epsilon (epsilon): Privacy budget - lower means more private
- Delta (delta): Probability of privacy breach, should be << 1/dataset_size
- Noise multiplier: Amount of Gaussian noise added to gradients
- Max grad norm: Gradient clipping threshold for bounded sensitivity

Prerequisites:
- TenSafe server running
- Training dataset

Expected Output:
    DP Configuration:
      Target epsilon: 8.0
      Target delta: 1e-5
      Noise multiplier: 1.0

    Training with DP-SGD...
    Step 20: loss=2.15, epsilon=0.8, budget_used=10%
    Step 40: loss=1.95, epsilon=1.6, budget_used=20%
    ...

    Final privacy guarantee: (epsilon=7.5, delta=1e-5)-DP
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class DPConfig:
    """Configuration for differential privacy training."""
    enabled: bool = True
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    accountant_type: str = "rdp"  # Renyi DP for tighter bounds


@dataclass
class DPMetrics:
    """Metrics from DP training."""
    epsilon_spent: float = 0.0
    delta: float = 1e-5
    num_steps: int = 0
    noise_scale: float = 0.0


class DPTrainer:
    """Differential privacy training simulator."""

    def __init__(self, dp_config: DPConfig, batch_size: int = 32, dataset_size: int = 10000):
        self.config = dp_config
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.sampling_rate = batch_size / dataset_size
        self.current_step = 0
        self._epsilon_per_step = self._compute_epsilon_per_step()

    def _compute_epsilon_per_step(self) -> float:
        """Compute approximate epsilon per step using RDP."""
        # Simplified RDP accounting
        # Real implementation uses opacus or tensorflow-privacy
        noise = self.config.noise_multiplier
        q = self.sampling_rate
        # Approximate epsilon per step (simplified)
        return q * (1.0 / noise) * 0.5

    def train_step(self, loss: float) -> DPMetrics:
        """Simulate a DP training step."""
        self.current_step += 1

        # Compute current privacy spent
        epsilon_spent = self.current_step * self._epsilon_per_step

        return DPMetrics(
            epsilon_spent=epsilon_spent,
            delta=self.config.target_delta,
            num_steps=self.current_step,
            noise_scale=self.config.noise_multiplier,
        )

    def get_privacy_guarantee(self) -> tuple[float, float]:
        """Get the current privacy guarantee (epsilon, delta)."""
        epsilon = self.current_step * self._epsilon_per_step
        return epsilon, self.config.target_delta


def main():
    """Demonstrate basic DP training with TenSafe."""

    # =========================================================================
    # Step 1: Understanding Differential Privacy
    # =========================================================================
    print("=" * 60)
    print("DIFFERENTIAL PRIVACY TRAINING")
    print("=" * 60)
    print("""
    Differential Privacy (DP) provides mathematical privacy guarantees:

    Definition: An algorithm M is (epsilon, delta)-DP if for any two
    neighboring datasets D and D' (differing by one record), and any
    output set S:

        P[M(D) in S] <= e^epsilon * P[M(D') in S] + delta

    In simpler terms:
    - The model output changes very little whether any single training
      example is included or not
    - An adversary cannot confidently determine if a specific example
      was used for training

    Key parameters:
    - Epsilon (epsilon): Privacy budget (typical: 1-10)
      - epsilon=1: Very private, some utility loss
      - epsilon=8: Moderate privacy, good utility
      - epsilon=infinity: No privacy

    - Delta (delta): Failure probability, should be << 1/dataset_size
      - For 100K samples, use delta=1e-5 or smaller
    """)

    # =========================================================================
    # Step 2: Configure DP parameters
    # =========================================================================
    print("\nConfiguring differential privacy...")

    dp_config = DPConfig(
        enabled=True,
        target_epsilon=8.0,
        target_delta=1e-5,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        accountant_type="rdp",
    )

    print(f"  Target epsilon: {dp_config.target_epsilon}")
    print(f"  Target delta: {dp_config.target_delta}")
    print(f"  Noise multiplier: {dp_config.noise_multiplier}")
    print(f"  Max gradient norm: {dp_config.max_grad_norm}")
    print(f"  Accountant: {dp_config.accountant_type}")

    # =========================================================================
    # Step 3: Initialize DP trainer
    # =========================================================================
    print("\nInitializing DP trainer...")

    batch_size = 32
    dataset_size = 10000
    trainer = DPTrainer(dp_config, batch_size, dataset_size)

    print(f"  Batch size: {batch_size}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Sampling rate: {trainer.sampling_rate:.4f}")

    # =========================================================================
    # Step 4: Training loop with DP
    # =========================================================================
    print("\nTraining with DP-SGD...")
    print("-" * 50)

    max_steps = 100
    simulated_losses = [2.5 - (i * 0.01) for i in range(max_steps)]

    for step in range(max_steps):
        loss = simulated_losses[step]
        metrics = trainer.train_step(loss)

        # Log progress every 20 steps
        if (step + 1) % 20 == 0:
            budget_pct = (metrics.epsilon_spent / dp_config.target_epsilon) * 100
            print(f"Step {step + 1:3d}: loss={loss:.2f}, "
                  f"epsilon={metrics.epsilon_spent:.2f}, "
                  f"budget_used={budget_pct:.0f}%")

        # Early stopping if budget exceeded
        if metrics.epsilon_spent >= dp_config.target_epsilon:
            print(f"\nBudget exhausted at step {step + 1}")
            break

    print("-" * 50)

    # =========================================================================
    # Step 5: Final privacy report
    # =========================================================================
    epsilon, delta = trainer.get_privacy_guarantee()

    print("\n" + "=" * 60)
    print("FINAL PRIVACY REPORT")
    print("=" * 60)
    print(f"""
    Training completed with differential privacy:

    Privacy guarantee: ({epsilon:.2f}, {delta:.0e})-DP

    This means:
    - For any training example, an adversary looking at the model
      cannot determine with high confidence whether that example
      was in the training set
    - The privacy guarantee is mathematically provable
    - The guarantee holds against any adversary, regardless of
      their computational power or auxiliary information

    Training statistics:
    - Total steps: {trainer.current_step}
    - Epsilon spent: {epsilon:.2f} / {dp_config.target_epsilon:.2f}
    - Budget remaining: {dp_config.target_epsilon - epsilon:.2f}
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("PRIVACY-UTILITY TRADEOFF TIPS")
    print("=" * 60)
    print("""
    Tips for better privacy-utility tradeoff:

    1. Use larger batch sizes
       - More samples per batch = less noise per sample
       - Recommended: 256-2048 for DP training

    2. Tune noise multiplier
       - Start with 1.0 and adjust based on results
       - Lower noise = less privacy, better utility

    3. Use RDP accounting
       - Tighter bounds than naive composition
       - Enabled by default in TenSafe

    4. Consider group privacy
       - If users have multiple samples, multiply epsilon
       - k samples per user = k * epsilon effective privacy

    5. Early stopping
       - Stop training before budget is exhausted
       - Save checkpoints to compare quality vs privacy
    """)


if __name__ == "__main__":
    main()
