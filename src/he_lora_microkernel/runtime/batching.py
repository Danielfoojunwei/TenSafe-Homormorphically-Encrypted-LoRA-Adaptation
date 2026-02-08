"""
Batch Management for HE-LoRA Microkernel

This module handles dynamic batch management for HE-LoRA execution.
Key responsibilities:

  1. Batch size adjustment at runtime
  2. Schedule recompilation when batch changes
  3. Activation padding for partial batches
  4. Performance impact reporting

When batch_size changes, the packing layout and schedules must be
recompiled to maintain deterministic performance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..compiler import (
    CKKSParams,
    CostEstimate,
    ExecutionSchedule,
    LoRAConfig,
    compile_schedule,
    estimate_costs,
    get_profile,
)

# =============================================================================
# BATCH CONFIGURATION
# =============================================================================

@dataclass
class BatchConfig:
    """
    Batch configuration for HE-LoRA execution.

    Changes to batch_size require recompilation.
    """
    # Current batch size
    batch_size: int

    # Supported batch sizes (pre-compiled)
    supported_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])

    # Auto-recompile on size change
    auto_recompile: bool = True

    # Maximum batch size (hard limit)
    max_batch_size: int = 32

    # Minimum batch size
    min_batch_size: int = 1

    def validate(self) -> None:
        """Validate batch configuration."""
        if self.batch_size < self.min_batch_size:
            raise ValueError(
                f"batch_size {self.batch_size} < min {self.min_batch_size}"
            )
        if self.batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {self.batch_size} > max {self.max_batch_size}"
            )


# =============================================================================
# BATCH MANAGER
# =============================================================================

class BatchManager:
    """
    Manages batch size adjustments and schedule recompilation.

    This class tracks the current batch size, maintains pre-compiled
    schedules for common sizes, and triggers recompilation when needed.
    """

    # Maximum number of cached schedules to prevent unbounded memory growth
    MAX_CACHE_SIZE = 64

    def __init__(
        self,
        base_config: LoRAConfig,
        ckks_params: Optional[CKKSParams] = None,
        precompile_sizes: Optional[List[int]] = None,
    ):
        """
        Initialize batch manager.

        Args:
            base_config: Base LoRA configuration (batch_size will be overridden)
            ckks_params: CKKS parameters (defaults to profile from config)
            precompile_sizes: Batch sizes to precompile (default: [1, 4, 8, 16])
        """
        self._base_config = base_config
        self._ckks_params = ckks_params or get_profile(base_config.ckks_profile)

        # Current state
        self._current_batch_size = base_config.batch_size
        self._current_schedule: Optional[ExecutionSchedule] = None

        # Pre-compiled schedules cache (bounded)
        self._schedule_cache: Dict[int, ExecutionSchedule] = {}

        # Cost estimates for each batch size
        self._cost_estimates: Dict[int, CostEstimate] = {}

        # Precompile common sizes
        precompile_sizes = precompile_sizes or [1, 4, 8, 16]
        for size in precompile_sizes:
            self._precompile(size)

        # Set initial schedule
        if base_config.batch_size in self._schedule_cache:
            self._current_schedule = self._schedule_cache[base_config.batch_size]
        else:
            self._current_schedule = self._compile_for_size(base_config.batch_size)

    def _precompile(self, batch_size: int) -> None:
        """Precompile schedule for a batch size."""
        if batch_size not in self._schedule_cache:
            try:
                schedule = self._compile_for_size(batch_size)
                self._schedule_cache[batch_size] = schedule
            except ValueError:
                # May fail for sizes that don't fit
                pass

    def _compile_for_size(self, batch_size: int) -> ExecutionSchedule:
        """Compile schedule for specific batch size."""
        config = LoRAConfig(
            hidden_size=self._base_config.hidden_size,
            rank=self._base_config.rank,
            alpha=self._base_config.alpha,
            targets=self._base_config.targets,
            batch_size=batch_size,
            max_context_length=self._base_config.max_context_length,
            ckks_profile=self._base_config.ckks_profile,
        )

        schedule = compile_schedule(config, self._ckks_params)

        # Cache cost estimate
        self._cost_estimates[batch_size] = estimate_costs(
            config, schedule.layout, config.ckks_profile
        )

        return schedule

    @property
    def current_batch_size(self) -> int:
        """Get current batch size."""
        return self._current_batch_size

    @property
    def current_schedule(self) -> ExecutionSchedule:
        """Get current schedule."""
        return self._current_schedule

    def set_batch_size(self, batch_size: int) -> ExecutionSchedule:
        """
        Set batch size, recompiling if necessary.

        Args:
            batch_size: New batch size

        Returns:
            Schedule for new batch size
        """
        if batch_size == self._current_batch_size:
            return self._current_schedule

        # Check cache first
        if batch_size in self._schedule_cache:
            self._current_schedule = self._schedule_cache[batch_size]
        else:
            # Evict oldest entry if cache is full
            if len(self._schedule_cache) >= self.MAX_CACHE_SIZE:
                oldest_key = next(iter(self._schedule_cache))
                del self._schedule_cache[oldest_key]
                self._cost_estimates.pop(oldest_key, None)

            # Compile new schedule
            self._current_schedule = self._compile_for_size(batch_size)
            self._schedule_cache[batch_size] = self._current_schedule

        self._current_batch_size = batch_size
        return self._current_schedule

    def get_cost_estimate(self, batch_size: Optional[int] = None) -> CostEstimate:
        """
        Get cost estimate for a batch size.

        Args:
            batch_size: Batch size (default: current)

        Returns:
            Cost estimate
        """
        size = batch_size or self._current_batch_size

        if size not in self._cost_estimates:
            # Compile to get estimate
            self._compile_for_size(size)

        return self._cost_estimates[size]

    def compare_batch_sizes(
        self,
        sizes: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compare performance across batch sizes.

        Args:
            sizes: Batch sizes to compare

        Returns:
            Dict mapping size to performance metrics
        """
        comparison = {}

        for size in sizes:
            try:
                estimate = self.get_cost_estimate(size)

                # Get schedule for slot count
                if size in self._schedule_cache:
                    schedule = self._schedule_cache[size]
                else:
                    schedule = self._compile_for_size(size)

                comparison[size] = {
                    'rotations_per_token': estimate.num_rotations,
                    'total_time_us': estimate.total_us,
                    'tokens_per_second_estimate': (
                        1_000_000 / estimate.total_us if estimate.total_us > 0 else 0
                    ),
                    'aggregate_throughput': (
                        size * 1_000_000 / estimate.total_us if estimate.total_us > 0 else 0
                    ),
                    'slots_used': schedule.layout.total_slots_used,
                    'slot_utilization': (
                        schedule.layout.total_slots_used / schedule.layout.slot_count
                    ),
                    'valid': schedule.is_valid,
                }
            except ValueError as e:
                comparison[size] = {
                    'error': str(e),
                    'valid': False,
                }

        return comparison

    def get_optimal_batch_size(
        self,
        available_sizes: Optional[List[int]] = None,
        optimize_for: str = 'throughput',
    ) -> int:
        """
        Find optimal batch size for given metric.

        Args:
            available_sizes: Sizes to consider (default: all cached)
            optimize_for: 'throughput', 'latency', or 'efficiency'

        Returns:
            Optimal batch size
        """
        sizes = available_sizes or list(self._schedule_cache.keys())

        if not sizes:
            return self._current_batch_size

        comparison = self.compare_batch_sizes(sizes)

        valid_sizes = [s for s in sizes if comparison.get(s, {}).get('valid', False)]
        if not valid_sizes:
            return self._current_batch_size

        if optimize_for == 'throughput':
            # Maximize aggregate tokens/second
            return max(
                valid_sizes,
                key=lambda s: comparison[s].get('aggregate_throughput', 0)
            )
        elif optimize_for == 'latency':
            # Minimize per-token time
            return min(
                valid_sizes,
                key=lambda s: comparison[s].get('total_time_us', float('inf'))
            )
        elif optimize_for == 'efficiency':
            # Maximize slot utilization
            return max(
                valid_sizes,
                key=lambda s: comparison[s].get('slot_utilization', 0)
            )
        else:
            return self._current_batch_size


# =============================================================================
# ACTIVATION PADDING
# =============================================================================

def pad_activations(
    activations: np.ndarray,
    target_batch_size: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Pad activations to target batch size.

    Args:
        activations: Input activations (actual_batch, hidden_size)
        target_batch_size: Target batch size
        pad_value: Value to use for padding

    Returns:
        (padded_activations, actual_batch_size)
    """
    actual_batch = activations.shape[0]
    hidden_size = activations.shape[1]

    if actual_batch == target_batch_size:
        return activations, actual_batch

    if actual_batch > target_batch_size:
        raise ValueError(
            f"Actual batch {actual_batch} > target {target_batch_size}"
        )

    # Pad with zeros (or specified value)
    padded = np.full(
        (target_batch_size, hidden_size),
        pad_value,
        dtype=activations.dtype
    )
    padded[:actual_batch] = activations

    return padded, actual_batch


def unpad_activations(
    activations: np.ndarray,
    actual_batch_size: int,
) -> np.ndarray:
    """
    Remove padding from activations.

    Args:
        activations: Padded activations
        actual_batch_size: Original batch size

    Returns:
        Unpadded activations
    """
    return activations[:actual_batch_size]


# =============================================================================
# DYNAMIC BATCH EXECUTOR
# =============================================================================

class DynamicBatchExecutor:
    """
    Executor with dynamic batch size adjustment.

    This wraps an HELoRAExecutor and handles:
      - Automatic batch size adjustment
      - Activation padding for partial batches
      - Schedule recompilation on size change
      - Weight persistence across executor re-creation
    """

    def __init__(
        self,
        base_config: LoRAConfig,
        backend_type: str = 'SIMULATION',
        ckks_params: Optional[CKKSParams] = None,
    ):
        """
        Initialize dynamic batch executor.

        Args:
            base_config: Base LoRA configuration
            backend_type: Backend type string
            ckks_params: CKKS parameters
        """
        self._batch_manager = BatchManager(
            base_config,
            ckks_params,
            precompile_sizes=[1, 4, 8, 16, 32],
        )

        self._backend_type = backend_type
        self._base_config = base_config
        self._ckks_params = ckks_params or get_profile(base_config.ckks_profile)

        # Current executor (lazily created)
        self._executor = None
        self._executor_batch_size = None

        # Stored weights for re-loading after executor re-creation
        self._stored_weights: Optional[tuple] = None  # (A, B, alpha)

    def _get_executor(self, batch_size: int):
        """Get or create executor for batch size.

        When the batch size changes and a new executor is created,
        previously loaded weights are automatically re-loaded.
        """
        from ..backend.gpu_ckks_backend import BackendType
        from .executor import HELoRAExecutor

        if self._executor is None or self._executor_batch_size != batch_size:
            schedule = self._batch_manager.set_batch_size(batch_size)

            backend_type = BackendType[self._backend_type]
            self._executor = HELoRAExecutor(schedule, backend_type)
            self._executor_batch_size = batch_size

            # Re-load weights into the new executor
            if self._stored_weights is not None:
                A, B, alpha = self._stored_weights
                self._executor.load_weights(A, B, alpha)

        return self._executor

    def execute(
        self,
        activations: np.ndarray,
        position: Optional[int] = None,
    ) -> np.ndarray:
        """
        Execute HE-LoRA with automatic batch handling.

        Args:
            activations: Activations (any batch size)
            position: Token position

        Returns:
            LoRA deltas
        """
        actual_batch = activations.shape[0]

        # Find appropriate batch size
        target_batch = self._find_target_batch(actual_batch)

        # Get executor
        executor = self._get_executor(target_batch)

        # Pad if needed
        if actual_batch != target_batch:
            padded, _ = pad_activations(activations, target_batch)
        else:
            padded = activations

        # Execute
        delta = executor.execute_token(padded, position)

        # Unpad
        if actual_batch != target_batch:
            delta = unpad_activations(delta, actual_batch)

        return delta

    def _find_target_batch(self, actual_batch: int) -> int:
        """Find smallest target batch size >= actual."""
        available = sorted(self._batch_manager._schedule_cache.keys())

        for size in available:
            if size >= actual_batch:
                return size

        # Compile new size
        return actual_batch

    def load_weights(self, A: np.ndarray, B: np.ndarray, alpha: float) -> None:
        """Load weights into current executor and store for future re-creation."""
        self._stored_weights = (A.copy(), B.copy(), alpha)
        if self._executor is not None:
            self._executor.load_weights(A, B, alpha)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report across batch sizes."""
        sizes = [1, 4, 8, 16, 32]
        return {
            'batch_comparison': self._batch_manager.compare_batch_sizes(sizes),
            'optimal_for_throughput': self._batch_manager.get_optimal_batch_size(
                sizes, 'throughput'
            ),
            'optimal_for_latency': self._batch_manager.get_optimal_batch_size(
                sizes, 'latency'
            ),
            'current_batch_size': self._batch_manager.current_batch_size,
        }
