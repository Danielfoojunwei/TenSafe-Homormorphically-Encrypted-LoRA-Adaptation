#!/usr/bin/env python3
"""
Gradient Accumulation for Large Batch Training

This example demonstrates gradient accumulation, which allows training with
effectively larger batch sizes than what fits in GPU memory. This is crucial
for:
- Training large models on limited hardware
- Differential privacy (DP-SGD benefits from larger batches)
- Stable training with larger effective learning rates

How it works:
1. Process small "micro-batches" that fit in memory
2. Accumulate gradients over N micro-batches
3. Apply optimizer step with accumulated gradients
4. Effective batch = micro_batch_size * accumulation_steps

Expected Output:
    Configuration:
      Micro-batch size: 4
      Accumulation steps: 16
      Effective batch size: 64

    Training with gradient accumulation:
      [Micro 1/16] loss=2.45
      [Micro 2/16] loss=2.42
      ...
      [Micro 16/16] loss=2.30
      -> Optimizer step 1 (accumulated 16 micro-batches)

    Training complete!
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class GradAccumConfig:
    """Configuration for gradient accumulation training."""
    # Batch sizes
    micro_batch_size: int = 4           # Batch that fits in GPU memory
    gradient_accumulation_steps: int = 16  # Number of micro-batches to accumulate

    # Training
    learning_rate: float = 1e-4
    max_steps: int = 100
    warmup_steps: int = 10

    # Model
    model_name: str = "meta-llama/Llama-3-8B"

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.micro_batch_size * self.gradient_accumulation_steps


class GradientAccumulationTrainer:
    """Trainer with gradient accumulation support."""

    def __init__(self, config: GradAccumConfig):
        self.config = config
        self.accumulated_loss = 0.0
        self.accumulated_steps = 0
        self.optimizer_steps = 0

    def forward_backward(self, batch: Dict) -> float:
        """
        Run forward and backward pass for a micro-batch.

        The gradient is accumulated but optimizer is not stepped.
        """
        # Simulate forward pass
        loss = 2.5 - (self.optimizer_steps * 0.01)

        # Simulate backward pass (accumulates gradients)
        # In PyTorch: loss.backward() with no optimizer.zero_grad()

        self.accumulated_loss += loss
        self.accumulated_steps += 1

        return loss

    def should_step(self) -> bool:
        """Check if we should apply optimizer step."""
        return self.accumulated_steps >= self.config.gradient_accumulation_steps

    def optimizer_step(self):
        """Apply accumulated gradients and step optimizer."""
        # Average the accumulated loss
        avg_loss = self.accumulated_loss / self.accumulated_steps

        # Simulate optimizer step
        # In PyTorch: optimizer.step() then optimizer.zero_grad()

        self.optimizer_steps += 1
        self.accumulated_loss = 0.0
        self.accumulated_steps = 0

        return avg_loss

    def get_learning_rate(self, step: int) -> float:
        """Get learning rate with warmup schedule."""
        if step < self.config.warmup_steps:
            # Linear warmup
            return self.config.learning_rate * (step / self.config.warmup_steps)
        else:
            # Constant after warmup
            return self.config.learning_rate


def create_dummy_dataloader(batch_size: int, num_samples: int = 1000) -> Iterator[Dict]:
    """Create dummy dataloader for demonstration."""
    for i in range(0, num_samples, batch_size):
        yield {
            "input_ids": list(range(i, i + batch_size)),
            "labels": list(range(i + 1, i + batch_size + 1)),
        }


def main():
    """Demonstrate gradient accumulation training."""
    print("=" * 60)
    print("GRADIENT ACCUMULATION")
    print("=" * 60)
    print("""
    Gradient accumulation simulates larger batch sizes:

    Without accumulation:
      - micro_batch = 64, fits in GPU
      - effective_batch = 64

    With accumulation (4 steps):
      - micro_batch = 16, fits in smaller GPU
      - effective_batch = 16 * 4 = 64

    Benefits:
    1. Train with larger effective batches on limited memory
    2. Better for DP-SGD (larger batches = less noise per sample)
    3. More stable training dynamics
    """)

    # Configuration
    config = GradAccumConfig(
        micro_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        max_steps=5,  # Number of optimizer steps
    )

    print(f"\nConfiguration:")
    print(f"  Micro-batch size: {config.micro_batch_size}")
    print(f"  Accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.effective_batch_size}")

    # Initialize trainer
    trainer = GradientAccumulationTrainer(config)
    dataloader = create_dummy_dataloader(config.micro_batch_size)

    # Training loop
    print("\nTraining with gradient accumulation:")
    print("-" * 50)

    micro_batch_idx = 0
    for batch in dataloader:
        micro_batch_idx += 1

        # Forward and backward (accumulates gradients)
        loss = trainer.forward_backward(batch)
        lr = trainer.get_learning_rate(trainer.optimizer_steps)

        print(f"  [Micro {trainer.accumulated_steps:2d}/"
              f"{config.gradient_accumulation_steps}] loss={loss:.4f}")

        # Step optimizer when we've accumulated enough
        if trainer.should_step():
            avg_loss = trainer.optimizer_step()
            print(f"  -> Optimizer step {trainer.optimizer_steps} "
                  f"(avg_loss={avg_loss:.4f}, lr={lr:.6f})")
            print()

            if trainer.optimizer_steps >= config.max_steps:
                break

    print("-" * 50)
    print(f"\nTraining complete!")
    print(f"  Optimizer steps: {trainer.optimizer_steps}")
    print(f"  Micro-batches processed: {micro_batch_idx}")
    print(f"  Samples processed: {micro_batch_idx * config.micro_batch_size}")

    # Memory estimation
    print("\n" + "=" * 60)
    print("MEMORY SAVINGS")
    print("=" * 60)
    print("""
    Memory usage comparison (Llama-3-8B):

    Batch Size | Accumulation | Memory  | Effective Batch
    -----------|--------------|---------|----------------
    32         | 1            | 40 GB   | 32
    8          | 4            | 16 GB   | 32
    4          | 8            | 12 GB   | 32
    2          | 16           | 10 GB   | 32

    Note: Smaller micro-batches = more memory efficient
    but same training dynamics as large batches.
    """)


if __name__ == "__main__":
    main()
