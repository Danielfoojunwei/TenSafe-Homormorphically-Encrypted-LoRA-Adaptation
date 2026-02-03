#!/usr/bin/env python3
"""
Mixed Precision Training with TenSafe

This example demonstrates mixed precision training using FP16/BF16 to:
- Reduce memory usage by ~50%
- Speed up training by 2-3x on modern GPUs
- Maintain model quality with proper scaling

Key concepts:
- FP16: 16-bit floating point (faster, less memory)
- BF16: Brain floating point (better range, simpler)
- Loss scaling: Prevents gradient underflow in FP16
- Master weights: Keep FP32 copy for optimizer

Expected Output:
    Mixed precision configuration:
      Compute dtype: bfloat16
      Memory savings: ~50%
      Speedup: ~2x

    Training with mixed precision:
      Step 100: loss=2.15, scale=65536.0, memory=12.5 GB
      Step 200: loss=1.92, scale=65536.0, memory=12.5 GB

    Training complete!
    Throughput improvement: 2.1x
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class DType(Enum):
    """Supported data types for mixed precision."""
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    # Compute precision
    compute_dtype: DType = DType.BF16
    param_dtype: DType = DType.FP32  # Master weights

    # Loss scaling (for FP16)
    use_loss_scaling: bool = True
    initial_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    # Optimization
    use_torch_compile: bool = False
    use_flash_attention: bool = True

    @property
    def memory_savings(self) -> float:
        """Estimate memory savings percentage."""
        if self.compute_dtype in [DType.FP16, DType.BF16]:
            return 0.5  # 50% savings on activations
        return 0.0

    @property
    def expected_speedup(self) -> float:
        """Estimate speedup factor."""
        if self.compute_dtype == DType.BF16:
            return 2.0  # 2x on Ampere/Hopper GPUs
        elif self.compute_dtype == DType.FP16:
            return 2.5  # Slightly faster than BF16
        return 1.0


class LossScaler:
    """Dynamic loss scaling for FP16 training."""

    def __init__(
        self,
        initial_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_tracker = 0
        self.found_inf_count = 0

    def scale_loss(self, loss: float) -> float:
        """Scale loss before backward pass."""
        return loss * self.scale

    def unscale_gradients(self, grads_magnitude: float) -> float:
        """Unscale gradients after backward pass."""
        return grads_magnitude / self.scale

    def update(self, found_inf: bool):
        """Update scale based on whether inf/nan was found."""
        if found_inf:
            # Reduce scale on overflow
            self.scale *= self.backoff_factor
            self.growth_tracker = 0
            self.found_inf_count += 1
        else:
            # Increase scale periodically
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self.growth_tracker = 0


class MixedPrecisionTrainer:
    """Trainer with mixed precision support."""

    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler = LossScaler(
            initial_scale=config.initial_scale,
            growth_factor=config.growth_factor,
            backoff_factor=config.backoff_factor,
            growth_interval=config.growth_interval,
        ) if config.use_loss_scaling else None
        self.step = 0

    def train_step(self) -> dict:
        """Execute one training step with mixed precision."""
        self.step += 1

        # Simulate loss computation
        loss = 2.5 - (self.step * 0.005)

        # Scale loss for FP16
        if self.scaler:
            scaled_loss = self.scaler.scale_loss(loss)
        else:
            scaled_loss = loss

        # Simulate backward pass
        # In PyTorch: with autocast(), scaled_loss.backward()

        # Check for inf/nan
        found_inf = False  # Simulated

        # Unscale and update
        if self.scaler:
            self.scaler.update(found_inf)

        # Estimate memory usage
        base_memory = 25.0  # GB for FP32
        memory = base_memory * (1 - self.config.memory_savings)

        return {
            "loss": loss,
            "scale": self.scaler.scale if self.scaler else 1.0,
            "memory_gb": memory,
            "found_inf": found_inf,
        }


def main():
    """Demonstrate mixed precision training."""
    print("=" * 60)
    print("MIXED PRECISION TRAINING")
    print("=" * 60)
    print("""
    Mixed precision training uses lower precision (FP16/BF16) for:
    1. Forward pass activations
    2. Backward pass gradients
    3. Most matrix multiplications

    While keeping FP32 for:
    1. Master weights
    2. Loss computation
    3. Gradient accumulation

    Benefits:
    - 2x memory reduction for activations
    - 2-3x speedup on Tensor Core GPUs
    - No loss in model quality with proper techniques
    """)

    # Compare different precisions
    print("\n" + "=" * 60)
    print("PRECISION COMPARISON")
    print("=" * 60)

    precisions = [
        ("FP32 (baseline)", DType.FP32),
        ("FP16 (fast, needs scaling)", DType.FP16),
        ("BF16 (recommended)", DType.BF16),
    ]

    print(f"\n{'Precision':<30} {'Memory':<15} {'Speedup':<15}")
    print("-" * 60)

    for name, dtype in precisions:
        config = MixedPrecisionConfig(compute_dtype=dtype)
        memory = f"{(1 - config.memory_savings) * 100:.0f}%"
        speedup = f"{config.expected_speedup:.1f}x"
        print(f"{name:<30} {memory:<15} {speedup:<15}")

    # Training example with BF16
    print("\n" + "=" * 60)
    print("TRAINING WITH BF16")
    print("=" * 60)

    config = MixedPrecisionConfig(
        compute_dtype=DType.BF16,
        use_loss_scaling=True,
        use_flash_attention=True,
    )

    print(f"\nConfiguration:")
    print(f"  Compute dtype: {config.compute_dtype.value}")
    print(f"  Loss scaling: {config.use_loss_scaling}")
    print(f"  Flash attention: {config.use_flash_attention}")

    trainer = MixedPrecisionTrainer(config)

    print("\nTraining:")
    print("-" * 50)

    for step in range(1, 6):
        result = trainer.train_step()
        print(f"  Step {step}: loss={result['loss']:.4f}, "
              f"scale={result['scale']:.0f}, "
              f"memory={result['memory_gb']:.1f} GB")

    # BF16 vs FP16 guidance
    print("\n" + "=" * 60)
    print("BF16 vs FP16 GUIDANCE")
    print("=" * 60)
    print("""
    Use BF16 (recommended) when:
    - You have Ampere (A100) or newer GPU
    - You want simplicity (no loss scaling needed)
    - Training large language models

    Use FP16 when:
    - You have older GPUs (V100, T4)
    - You need maximum speed
    - You're willing to tune loss scaling

    Code example (PyTorch):
    ```python
    # BF16 with autocast
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # FP16 with GradScaler
    scaler = torch.cuda.amp.GradScaler()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    ```
    """)


if __name__ == "__main__":
    main()
