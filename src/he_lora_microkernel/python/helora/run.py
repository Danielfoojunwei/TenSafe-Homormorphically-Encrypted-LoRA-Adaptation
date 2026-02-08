"""
Execution Interface for HE-LoRA Microkernel

This module provides a high-level execution interface for running
HE-LoRA inference on every generated token.

Usage:
    from helora import run_helora

    # Compile and create executor
    executor = run_helora.create_executor(config, A, B, alpha)

    # Execute for each token
    for token in generation:
        delta = executor(activations)
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from he_lora_microkernel.backend.gpu_ckks_backend import BackendType
from he_lora_microkernel.compiler import ExecutionSchedule
from he_lora_microkernel.runtime import (
    HELoRAExecutor,
    InvariantChecker,
    PerformanceReporter,
    TelemetryCollector,
)

from .compile import CompilationResult, compile_lora
from .config import HELoRAConfig


class HELoRARunner:
    """
    High-level runner for HE-LoRA inference.

    This provides a simple interface for running HE-LoRA on every token.
    """

    def __init__(
        self,
        config: HELoRAConfig,
        A: np.ndarray,
        B: np.ndarray,
        alpha: Optional[float] = None,
        backend: str = "SIMULATION",
        device_id: int = 0,
        enable_telemetry: bool = True,
    ):
        """
        Initialize HE-LoRA runner.

        Args:
            config: HE-LoRA configuration
            A: Up-projection matrix (hidden_size, rank)
            B: Down-projection matrix (rank, hidden_size)
            alpha: LoRA scaling factor
            backend: Backend type ("SIMULATION", "HEONGPU", etc.)
            device_id: GPU device ID
            enable_telemetry: Enable performance telemetry
        """
        self._config = config
        self._alpha = alpha if alpha is not None else config.lora_alpha

        # Compile
        result = compile_lora(config, A, B, alpha)
        if not result.is_valid:
            raise ValueError(f"Compilation failed: {result.violations}")

        self._compilation = result

        # Create executor
        backend_type = BackendType[backend]
        self._executor = HELoRAExecutor(
            result.schedule,
            backend_type,
            device_id,
            config.get_cost_budget() if config.enforce_budget else None,
        )

        # Load weights
        self._executor.load_weights(A, B, self._alpha)

        # Telemetry
        self._telemetry_enabled = enable_telemetry
        if enable_telemetry:
            self._telemetry = TelemetryCollector()
            self._invariant_checker = InvariantChecker(
                max_rotations_per_token=config.rotation_budget,
                max_keyswitches_per_token=config.keyswitch_budget,
                max_rescales_per_token=config.rescale_budget,
            )
        else:
            self._telemetry = None
            self._invariant_checker = None

    def __call__(
        self,
        activations: np.ndarray,
        position: Optional[int] = None,
    ) -> np.ndarray:
        """
        Execute HE-LoRA for a token.

        Args:
            activations: Batch activations (batch_size, hidden_size)
            position: Optional token position

        Returns:
            LoRA delta
        """
        return self.execute(activations, position)

    def execute(
        self,
        activations: np.ndarray,
        position: Optional[int] = None,
    ) -> np.ndarray:
        """
        Execute HE-LoRA for a token.

        This applies HE-LoRA correction to the activations.
        NO SKIPPING - every call goes through full HE-LoRA.

        Args:
            activations: Batch activations (batch_size, hidden_size)
            position: Optional token position

        Returns:
            LoRA delta to add to base model output
        """
        # Execute
        delta = self._executor.execute_token(activations, position)

        # Record telemetry
        if self._telemetry_enabled:
            stats = self._executor.get_statistics()
            counters = stats['backend_counters']

            self._telemetry.record_token_metrics(
                rotations=counters['rotations'],
                keyswitches=counters['keyswitches'],
                rescales=counters['rescales'],
                he_time_ms=counters['compute_time_ms'],
                total_time_ms=counters['total_time_ms'],
            )

            # Check invariants
            if self._invariant_checker:
                self._invariant_checker.check_token(
                    rotations=counters['rotations'],
                    keyswitches=counters['keyswitches'],
                    rescales=counters['rescales'],
                    he_time_ms=counters['compute_time_ms'],
                    total_time_ms=counters['total_time_ms'],
                )

        return delta

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self._executor.get_statistics()

        if self._telemetry_enabled:
            reporter = PerformanceReporter(self._telemetry)
            stats['performance_report'] = reporter.generate_report()

            if self._invariant_checker:
                stats['invariant_check'] = self._invariant_checker.get_ci_result()

        return stats

    def get_ci_report(self) -> Dict[str, Any]:
        """Get CI validation report."""
        if not self._telemetry_enabled:
            return {'error': 'Telemetry not enabled'}

        reporter = PerformanceReporter(self._telemetry)
        return reporter.generate_ci_report(
            rotation_budget=self._config.rotation_budget,
            keyswitch_budget=self._config.keyswitch_budget,
            rescale_budget=self._config.rescale_budget,
        )

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._executor.reset_statistics()
        if self._telemetry:
            self._telemetry.reset()
        if self._invariant_checker:
            self._invariant_checker.reset()

    @property
    def compilation_result(self) -> CompilationResult:
        """Get compilation result."""
        return self._compilation

    @property
    def schedule(self) -> ExecutionSchedule:
        """Get compiled schedule."""
        return self._compilation.schedule


def create_executor(
    config: HELoRAConfig,
    A: np.ndarray,
    B: np.ndarray,
    alpha: Optional[float] = None,
    backend: str = "SIMULATION",
) -> HELoRARunner:
    """
    Create HE-LoRA executor from configuration.

    Args:
        config: HE-LoRA configuration
        A: Up-projection matrix
        B: Down-projection matrix
        alpha: LoRA scaling factor
        backend: Backend type

    Returns:
        HELoRARunner ready for execution
    """
    return HELoRARunner(config, A, B, alpha, backend)


def run_inference(
    config: HELoRAConfig,
    A: np.ndarray,
    B: np.ndarray,
    activations_sequence: List[np.ndarray],
    alpha: Optional[float] = None,
    backend: str = "SIMULATION",
) -> List[np.ndarray]:
    """
    Run HE-LoRA inference on a sequence of tokens.

    Args:
        config: HE-LoRA configuration
        A: Up-projection matrix
        B: Down-projection matrix
        activations_sequence: List of activation arrays
        alpha: LoRA scaling factor
        backend: Backend type

    Returns:
        List of LoRA deltas
    """
    runner = HELoRARunner(config, A, B, alpha, backend)

    deltas = []
    for i, activations in enumerate(activations_sequence):
        delta = runner.execute(activations, position=i)
        deltas.append(delta)

    return deltas


def benchmark_configuration(
    config: HELoRAConfig,
    A: np.ndarray,
    B: np.ndarray,
    num_tokens: int = 100,
    alpha: Optional[float] = None,
    backend: str = "SIMULATION",
) -> Dict[str, Any]:
    """
    Benchmark HE-LoRA configuration.

    Args:
        config: HE-LoRA configuration
        A: Up-projection matrix
        B: Down-projection matrix
        num_tokens: Number of tokens to process
        alpha: LoRA scaling factor
        backend: Backend type

    Returns:
        Benchmark results
    """
    runner = HELoRARunner(config, A, B, alpha, backend)

    # Generate random activations
    rng = np.random.default_rng(42)

    for i in range(num_tokens):
        activations = rng.standard_normal(
            (config.batch_size, config.hidden_size)
        ).astype(np.float64)
        runner.execute(activations, position=i)

    # Get statistics
    stats = runner.get_statistics()
    ci_report = runner.get_ci_report()

    return {
        'config': config.to_dict(),
        'num_tokens': num_tokens,
        'statistics': stats,
        'ci_report': ci_report,
        'compilation': runner.compilation_result.summary(),
    }
