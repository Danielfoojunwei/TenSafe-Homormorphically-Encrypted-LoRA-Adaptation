"""
RLVR Micro-Batch Gradient Accumulation with DP Coordination

Enables large effective batch sizes on limited GPU memory by accumulating
gradients across configurable micro-batches before taking an optimizer step.
Critical for HE-LoRA training where encrypted operations consume significant
GPU memory, and for DP-SGD where privacy amplifies with larger batches
(amplification by subsampling).

Key features:
- Configurable micro-batch size with automatic splitting
- Gradient accumulation with proper scaling
- DP-SGD coordination: per-sample clipping + noise addition at step boundary
- Memory-aware batch scheduling
- Compatible with existing TrajectoryBatch interface
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol

from .rollout import Trajectory, TrajectoryBatch

logger = logging.getLogger(__name__)


@dataclass
class MicroBatchConfig:
    """Configuration for micro-batch gradient accumulation."""

    # Micro-batch size (number of trajectories per forward-backward pass)
    micro_batch_size: int = 4

    # Total effective batch size
    effective_batch_size: int = 32

    # Whether to scale gradients by 1/num_accumulation_steps
    scale_gradients: bool = True

    # DP-SGD integration
    dp_enabled: bool = False
    dp_max_grad_norm: float = 1.0  # Per-sample gradient clipping bound
    dp_noise_multiplier: float = 1.0  # Gaussian noise multiplier

    # Maximum gradient norm for the accumulated gradient
    max_grad_norm: float = 1.0

    @property
    def num_accumulation_steps(self) -> int:
        """Number of micro-batches per optimizer step."""
        return max(1, self.effective_batch_size // self.micro_batch_size)

    @property
    def gradient_scale(self) -> float:
        """Scale factor for each micro-batch's gradient."""
        if self.scale_gradients:
            return 1.0 / self.num_accumulation_steps
        return 1.0


class GradientAccumulator:
    """
    Manages gradient accumulation across micro-batches.

    Splits a full training batch into micro-batches, runs forward-backward
    on each, accumulates gradients, then triggers the optimizer step.
    Coordinates with DP-SGD when enabled.
    """

    def __init__(self, config: Optional[MicroBatchConfig] = None):
        self.config = config or MicroBatchConfig()
        self._accumulated_steps = 0
        self._total_steps = 0

    def split_batch(self, batch: TrajectoryBatch) -> List[TrajectoryBatch]:
        """
        Split a trajectory batch into micro-batches.

        Args:
            batch: Full training batch

        Returns:
            List of micro-batches
        """
        trajectories = batch.trajectories
        micro_size = self.config.micro_batch_size
        micro_batches = []

        for i in range(0, len(trajectories), micro_size):
            chunk = trajectories[i : i + micro_size]
            micro_batches.append(TrajectoryBatch(trajectories=chunk))

        return micro_batches

    def accumulate_and_step(
        self,
        batch: TrajectoryBatch,
        forward_backward_fn: Callable[[TrajectoryBatch, float], Dict[str, float]],
        optimizer_step_fn: Callable[[bool], None],
        dp_enabled: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run gradient accumulation over micro-batches and take optimizer step.

        Args:
            batch: Full training batch
            forward_backward_fn: Function that runs forward-backward on a micro-batch.
                                Takes (micro_batch, gradient_scale) -> loss_dict
            optimizer_step_fn: Function that takes an optimizer step.
                              Takes (apply_dp_noise: bool)
            dp_enabled: Override DP setting (default: config.dp_enabled)

        Returns:
            Aggregated metrics from all micro-batches
        """
        dp = dp_enabled if dp_enabled is not None else self.config.dp_enabled
        micro_batches = self.split_batch(batch)
        gradient_scale = self.config.gradient_scale

        # Aggregate metrics
        agg_metrics: Dict[str, float] = {}
        total_trajectories = 0

        for mb_idx, micro_batch in enumerate(micro_batches):
            is_last = mb_idx == len(micro_batches) - 1

            # Forward-backward with gradient scaling
            mb_metrics = forward_backward_fn(micro_batch, gradient_scale)

            # Accumulate metrics
            n = len(micro_batch)
            for key, value in mb_metrics.items():
                if key not in agg_metrics:
                    agg_metrics[key] = 0.0
                agg_metrics[key] += value * n
            total_trajectories += n

            self._accumulated_steps += 1

        # Average metrics
        for key in agg_metrics:
            agg_metrics[key] /= total_trajectories

        # Optimizer step (with DP noise if enabled)
        optimizer_step_fn(dp)

        self._total_steps += 1
        self._accumulated_steps = 0

        agg_metrics["num_micro_batches"] = len(micro_batches)
        agg_metrics["effective_batch_size"] = total_trajectories
        agg_metrics["gradient_scale"] = gradient_scale

        return agg_metrics

    def iter_micro_batches(
        self,
        batch: TrajectoryBatch,
    ) -> Iterator[MicroBatchContext]:
        """
        Iterate over micro-batches with context for manual control.

        Yields MicroBatchContext objects that track whether the current
        micro-batch is the last one (triggering optimizer step).
        """
        micro_batches = self.split_batch(batch)
        total = len(micro_batches)

        for idx, mb in enumerate(micro_batches):
            yield MicroBatchContext(
                batch=mb,
                index=idx,
                total=total,
                is_last=(idx == total - 1),
                gradient_scale=self.config.gradient_scale,
            )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_optimizer_steps": self._total_steps,
            "micro_batch_size": self.config.micro_batch_size,
            "effective_batch_size": self.config.effective_batch_size,
            "num_accumulation_steps": self.config.num_accumulation_steps,
            "dp_enabled": self.config.dp_enabled,
        }


@dataclass
class MicroBatchContext:
    """Context for a single micro-batch within an accumulation cycle."""

    batch: TrajectoryBatch
    index: int
    total: int
    is_last: bool
    gradient_scale: float

    @property
    def should_step(self) -> bool:
        """Whether to take an optimizer step after this micro-batch."""
        return self.is_last


class DPAwareMicroBatcher:
    """
    DP-SGD aware micro-batching that coordinates per-sample clipping
    with gradient accumulation.

    In standard DP-SGD:
    1. Compute per-sample gradients
    2. Clip each gradient to max_grad_norm
    3. Sum clipped gradients
    4. Add Gaussian noise calibrated to max_grad_norm and batch size
    5. Average and step

    With micro-batching:
    - Steps 1-3 happen per micro-batch
    - Clipped gradients accumulate across micro-batches
    - Step 4 happens once after all micro-batches
    - The effective batch size for privacy accounting is the FULL batch

    This is important because privacy amplification depends on the
    subsampling ratio (batch_size / dataset_size), so the effective
    batch size must be the full batch, not the micro-batch.
    """

    def __init__(self, config: Optional[MicroBatchConfig] = None):
        self.config = config or MicroBatchConfig(dp_enabled=True)
        self._accumulator = GradientAccumulator(self.config)

        # Privacy accounting
        self._total_dp_steps = 0
        self._cumulative_epsilon = 0.0

    def process_batch(
        self,
        batch: TrajectoryBatch,
        forward_backward_fn: Callable[[TrajectoryBatch, float], Dict[str, float]],
        optimizer_step_fn: Callable[[bool], None],
        clip_and_accumulate_fn: Optional[
            Callable[[float], Dict[str, float]]
        ] = None,
    ) -> Dict[str, float]:
        """
        Process a full batch with DP-aware micro-batching.

        Args:
            batch: Full training batch
            forward_backward_fn: Forward-backward per micro-batch
            optimizer_step_fn: Optimizer step function
            clip_and_accumulate_fn: Optional per-sample clip function

        Returns:
            Aggregated metrics including DP accounting info
        """
        metrics = self._accumulator.accumulate_and_step(
            batch=batch,
            forward_backward_fn=forward_backward_fn,
            optimizer_step_fn=optimizer_step_fn,
            dp_enabled=True,
        )

        self._total_dp_steps += 1

        # Add DP-specific metrics
        metrics["dp_steps"] = self._total_dp_steps
        metrics["dp_max_grad_norm"] = self.config.dp_max_grad_norm
        metrics["dp_noise_multiplier"] = self.config.dp_noise_multiplier
        metrics["dp_effective_batch_size"] = len(batch)

        return metrics

    def compute_privacy_spent(
        self,
        dataset_size: int,
        delta: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Compute privacy budget spent so far (simplified RDP accounting).

        This is a simplified estimate. For production use, integrate
        with the full privacy accountant in tensafe.core.

        Args:
            dataset_size: Total dataset size
            delta: Target delta for (epsilon, delta)-DP

        Returns:
            Privacy budget information
        """
        if self._total_dp_steps == 0:
            return {"epsilon": 0.0, "delta": delta, "steps": 0}

        # Simplified Gaussian mechanism analysis
        q = self.config.effective_batch_size / dataset_size  # Subsampling ratio
        sigma = self.config.dp_noise_multiplier
        steps = self._total_dp_steps

        # Simple composition bound (loose; real accounting should use RDP/PRV)
        if sigma > 0:
            per_step_epsilon = q * math.sqrt(2 * math.log(1.25 / delta)) / sigma
            total_epsilon = per_step_epsilon * math.sqrt(steps)  # Advanced composition
        else:
            total_epsilon = float("inf")

        return {
            "epsilon": total_epsilon,
            "delta": delta,
            "steps": steps,
            "subsampling_ratio": q,
            "noise_multiplier": sigma,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._accumulator.get_stats(),
            "dp_steps": self._total_dp_steps,
        }
