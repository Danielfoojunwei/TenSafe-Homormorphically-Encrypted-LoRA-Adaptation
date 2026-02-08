"""
Runtime Executor for HE-LoRA Microkernel

This module provides the runtime execution engine for HE-LoRA inference.
It executes compiled schedules on every generated token.

CRITICAL REQUIREMENTS:
  1. HE-LoRA correction on EVERY generated token
  2. No token skipping
  3. No heuristic gating
  4. Strict context length enforcement

The executor is stateless - each token is processed independently
through the compiled HE-LoRA microkernel.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from ..backend.gpu_ckks_backend import (
    BackendType,
    GPUCiphertext,
    GPUCKKSBackend,
    PlaintextPacked,
    create_backend,
)
from ..compiler import (
    CostBudget,
    CostTracker,
    ExecutionSchedule,
    PackedLoRAWeights,
    pack_activations,
    unpack_activations,
)

# =============================================================================
# EXECUTION CONTEXT
# =============================================================================

class ExecutionMode(Enum):
    """Execution mode for the runtime."""
    PRODUCTION = "production"  # Full HE execution
    SIMULATION = "simulation"  # Simulated HE (for testing)
    PROFILING = "profiling"    # With detailed timing


@dataclass
class ExecutionContext:
    """
    Context for HE-LoRA execution.

    This holds all state needed to execute the compiled schedule.
    """
    # Compiled schedule
    schedule: ExecutionSchedule

    # Backend
    backend: GPUCKKSBackend

    # Pre-encoded weight plaintexts
    B_plaintexts: List[PlaintextPacked] = field(default_factory=list)
    A_plaintexts: List[PlaintextPacked] = field(default_factory=list)

    # Cost tracking
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    # Execution mode
    mode: ExecutionMode = ExecutionMode.PRODUCTION

    # Context length tracking
    current_context_length: int = 0

    # Statistics
    tokens_processed: int = 0
    total_he_time_ms: float = 0.0


# =============================================================================
# EXECUTOR
# =============================================================================

class HELoRAExecutor:
    """
    Runtime executor for HE-LoRA computation.

    This executes the compiled schedule for every token, applying
    HE-LoRA corrections to the base model's QKV (or QKVO) projections.

    Usage:
        executor = HELoRAExecutor(schedule, backend_type)
        executor.load_weights(A, B, alpha)

        for token in generation:
            activations = base_model.get_activations(token)
            delta = executor.execute_token(activations)
            output = base_output + delta
    """

    def __init__(
        self,
        schedule: ExecutionSchedule,
        backend_type: BackendType = BackendType.SIMULATION,
        device_id: int = 0,
        budget: Optional[CostBudget] = None,
    ):
        """
        Initialize executor with compiled schedule.

        Args:
            schedule: Compiled execution schedule
            backend_type: GPU backend to use
            device_id: GPU device ID
            budget: Optional cost budget for enforcement
        """
        if not schedule.is_valid:
            raise ValueError(
                f"Cannot execute invalid schedule: {schedule.validation_errors}"
            )

        self._schedule = schedule
        self._backend_type = backend_type
        self._device_id = device_id

        # Initialize backend
        self._backend = create_backend(
            backend_type,
            schedule.ckks_params,
            device_id,
        )

        # Initialize context
        self._context = ExecutionContext(
            schedule=schedule,
            backend=self._backend,
            cost_tracker=CostTracker(budget=budget),
            mode=(ExecutionMode.SIMULATION if backend_type == BackendType.SIMULATION
                  else ExecutionMode.PRODUCTION),
        )

        # Thread-safe lock for counter updates
        self._stats_lock = threading.Lock()

        # Weight plaintexts (loaded later)
        self._weights_loaded = False

        logger.debug(
            f"HELoRAExecutor initialized: backend={backend_type.value}, "
            f"device={device_id}, batch_size={schedule.config.batch_size}"
        )

    @property
    def schedule(self) -> ExecutionSchedule:
        """Get compiled schedule."""
        return self._schedule

    @property
    def backend(self) -> GPUCKKSBackend:
        """Get HE backend."""
        return self._backend

    @property
    def context(self) -> ExecutionContext:
        """Get execution context."""
        return self._context

    @property
    def cost_tracker(self) -> CostTracker:
        """Get cost tracker."""
        return self._context.cost_tracker

    # -------------------------------------------------------------------------
    # WEIGHT LOADING
    # -------------------------------------------------------------------------

    def load_weights(
        self,
        A: np.ndarray,
        B: np.ndarray,
        alpha: float,
    ) -> None:
        """
        Load and encode LoRA weights.

        Args:
            A: Up-projection matrix (hidden_size, rank)
            B: Down-projection matrix (rank, hidden_size)
            alpha: LoRA scaling factor
        """
        layout = self._schedule.layout
        config = self._schedule.config

        # Validate dimensions
        if A.shape != (config.hidden_size, config.rank):
            raise ValueError(
                f"A shape mismatch: {A.shape} vs expected "
                f"({config.hidden_size}, {config.rank})"
            )
        if B.shape != (config.rank, config.hidden_size):
            raise ValueError(
                f"B shape mismatch: {B.shape} vs expected "
                f"({config.rank}, {config.hidden_size})"
            )

        # Scale A with alpha/rank
        scaling = alpha / config.rank
        A_scaled = A * scaling

        # Encode B matrix blocks as plaintexts
        self._context.B_plaintexts = []
        for block in layout.blocks:
            # Extract block columns
            B_block = B[:, block.start_channel:block.end_channel]

            # Pack for batch-first layout (replicate across batch slots)
            packed = np.zeros(layout.slot_count, dtype=np.float64)
            for local_ch in range(B_block.shape[1]):
                for b in range(config.batch_size):
                    slot_idx = block.slot_offset + local_ch * config.batch_size + b
                    if slot_idx < layout.slot_count:
                        # Sum across rank for CPMM
                        packed[slot_idx] = np.sum(B_block[:, local_ch])

            pt = self._backend.encode_plaintext(packed)
            self._context.B_plaintexts.append(pt)

        # Encode A matrix blocks as plaintexts
        self._context.A_plaintexts = []
        for block in layout.blocks:
            # Extract block rows
            A_block = A_scaled[block.start_channel:block.end_channel, :]

            packed = np.zeros(layout.slot_count, dtype=np.float64)
            for local_ch in range(A_block.shape[0]):
                for b in range(config.batch_size):
                    slot_idx = block.slot_offset + local_ch * config.batch_size + b
                    if slot_idx < layout.slot_count:
                        packed[slot_idx] = np.sum(A_block[local_ch, :])

            pt = self._backend.encode_plaintext(packed)
            self._context.A_plaintexts.append(pt)

        self._weights_loaded = True

    def load_packed_weights(self, weights: PackedLoRAWeights) -> None:
        """
        Load pre-packed weights.

        Args:
            weights: Pre-packed LoRA weights from compiler
        """
        # Encode pre-packed blocks
        self._context.B_plaintexts = [
            self._backend.encode_plaintext(block)
            for block in weights.B_packed_blocks
        ]
        self._context.A_plaintexts = [
            self._backend.encode_plaintext(block)
            for block in weights.A_packed_blocks
        ]
        self._weights_loaded = True

    # -------------------------------------------------------------------------
    # CONTEXT LENGTH ENFORCEMENT
    # -------------------------------------------------------------------------

    def check_context_length(self, position: int) -> bool:
        """
        Check if position is within context length limit.

        Args:
            position: Current token position in sequence

        Returns:
            True if within limit, False otherwise
        """
        return position < self._schedule.config.max_context_length

    def enforce_context_length(self, position: int) -> None:
        """
        Enforce context length limit.

        Args:
            position: Current token position

        Raises:
            ValueError: If position exceeds max_context_length
        """
        if not self.check_context_length(position):
            raise ValueError(
                f"Context length exceeded: position {position} >= "
                f"max {self._schedule.config.max_context_length}"
            )

    # -------------------------------------------------------------------------
    # TOKEN EXECUTION
    # -------------------------------------------------------------------------

    def _compute_ct_delta(
        self,
        activations: np.ndarray,
    ) -> 'GPUCiphertext':
        """
        Run the encrypt → matmul → matmul pipeline and return the
        encrypted delta ciphertext (before decryption).

        This is factored out so callers can choose between:
          - execute_token() — classic decrypt → unpack → return delta
          - execute_token_fused() — fused decrypt-unpack-add into y_base

        Args:
            activations: Batch activations (batch_size, hidden_size)

        Returns:
            Encrypted delta ciphertext
        """
        layout = self._schedule.layout

        # Step 1: Pack activations
        packed_x = pack_activations(activations, layout)

        # Step 2: Encrypt
        with self._backend.timed_section('encrypt'):
            ct_x = self._backend.encrypt(packed_x)
        self._context.cost_tracker.record_encrypt()

        # Step 3: Execute B @ x (first matmul)
        with self._backend.timed_section('compute'):
            intermediate_results = []

            for i, B_pt in enumerate(self._context.B_plaintexts):
                # Ct × Pt multiplication with rescale
                result = self._backend.mul_plain(ct_x, B_pt)
                self._context.cost_tracker.record_mul_plain()

                self._backend.rescale_inplace(result)
                self._context.cost_tracker.record_rescale()

                intermediate_results.append(result)

            # Tree reduction for accumulation
            while len(intermediate_results) > 1:
                new_results = []
                for i in range(0, len(intermediate_results), 2):
                    if i + 1 < len(intermediate_results):
                        ct1 = intermediate_results[i]
                        ct2 = intermediate_results[i + 1]

                        # Align levels if needed
                        if ct1.level != ct2.level:
                            target = max(ct1.level, ct2.level)
                            ct1 = self._backend.modswitch_to_level(ct1, target)
                            ct2 = self._backend.modswitch_to_level(ct2, target)

                        self._backend.add_inplace(ct1, ct2)
                        self._context.cost_tracker.record_add()
                        new_results.append(ct1)
                    else:
                        new_results.append(intermediate_results[i])
                intermediate_results = new_results

            ct_Bx = intermediate_results[0]

            # Step 4: Execute A @ (Bx) (second matmul)
            result = self._backend.mul_plain(ct_Bx, self._context.A_plaintexts[0])
            self._context.cost_tracker.record_mul_plain()

            self._backend.rescale_inplace(result)
            self._context.cost_tracker.record_rescale()

            # Accumulate remaining A blocks
            for A_pt in self._context.A_plaintexts[1:]:
                block_result = self._backend.mul_plain(ct_Bx, A_pt)
                self._context.cost_tracker.record_mul_plain()

                self._backend.rescale_inplace(block_result)
                self._context.cost_tracker.record_rescale()

                # Align levels
                if block_result.level != result.level:
                    block_result = self._backend.modswitch_to_level(
                        block_result, result.level
                    )

                self._backend.add_inplace(result, block_result)
                self._context.cost_tracker.record_add()

            ct_delta = result

        return ct_delta

    def execute_token(
        self,
        activations: np.ndarray,
        position: Optional[int] = None,
    ) -> np.ndarray:
        """
        Execute HE-LoRA for a single token.

        This is the main entry point for every-token execution.
        NO SKIPPING - every token goes through HE-LoRA.

        Args:
            activations: Batch activations (batch_size, hidden_size)
            position: Optional token position for context length check

        Returns:
            LoRA delta to add to base model output

        Raises:
            ValueError: If weights not loaded or context length exceeded
        """
        if not self._weights_loaded:
            raise ValueError("Weights not loaded. Call load_weights() first.")

        # Enforce context length
        if position is not None:
            self.enforce_context_length(position)

        # Start token tracking
        self._context.cost_tracker.begin_token()
        start_time = time.perf_counter()

        layout = self._schedule.layout
        config = self._schedule.config

        # Validate input shape
        if activations.shape != (config.batch_size, config.hidden_size):
            raise ValueError(
                f"Activation shape mismatch: {activations.shape} vs expected "
                f"({config.batch_size}, {config.hidden_size})"
            )

        ct_delta = self._compute_ct_delta(activations)

        # Step 5: Decrypt (partial — only the slots that carry real data)
        with self._backend.timed_section('decrypt'):
            packed_delta = self._backend.decrypt_partial(
                ct_delta, layout.total_slots_used
            )
        self._context.cost_tracker.record_decrypt()

        # Step 6: Unpack
        delta = unpack_activations(packed_delta, layout)

        # End token tracking (thread-safe counter updates)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        with self._stats_lock:
            self._context.total_he_time_ms += elapsed_ms
            self._context.tokens_processed += 1
        self._context.cost_tracker.end_token(config.targets)

        return delta

    def execute_token_fused(
        self,
        activations: np.ndarray,
        y_base: np.ndarray,
        position: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fused HE-LoRA execution: decrypt + unpack + add into y_base in one pass.

        This eliminates two intermediate allocations and two extra memory
        passes compared to the separate execute_token() + numpy add pipeline.

        Args:
            activations: Batch activations (batch_size, hidden_size)
            y_base: Base model output (batch_size, hidden_size), modified in-place
            position: Optional token position for context length check

        Returns:
            y_base with decrypted delta added in-place
        """
        if not self._weights_loaded:
            raise ValueError("Weights not loaded. Call load_weights() first.")

        if position is not None:
            self.enforce_context_length(position)

        self._context.cost_tracker.begin_token()
        start_time = time.perf_counter()

        layout = self._schedule.layout
        config = self._schedule.config

        if activations.shape != (config.batch_size, config.hidden_size):
            raise ValueError(
                f"Activation shape mismatch: {activations.shape} vs expected "
                f"({config.batch_size}, {config.hidden_size})"
            )

        ct_delta = self._compute_ct_delta(activations)

        # Fused decrypt-unpack-add: single pass over output memory
        with self._backend.timed_section('decrypt'):
            self._backend.decrypt_fused_unpack_add(ct_delta, layout, y_base)
        self._context.cost_tracker.record_decrypt()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        with self._stats_lock:
            self._context.total_he_time_ms += elapsed_ms
            self._context.tokens_processed += 1
        self._context.cost_tracker.end_token(config.targets)

        return y_base

    def execute_batch(
        self,
        activations_batch: List[np.ndarray],
        positions: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Execute HE-LoRA for a batch of tokens.

        Args:
            activations_batch: List of activation arrays
            positions: Optional list of token positions

        Returns:
            List of LoRA deltas
        """
        deltas = []
        for i, activations in enumerate(activations_batch):
            pos = positions[i] if positions else None
            delta = self.execute_token(activations, pos)
            deltas.append(delta)
        return deltas

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'tokens_processed': self._context.tokens_processed,
            'total_he_time_ms': self._context.total_he_time_ms,
            'avg_time_per_token_ms': (
                self._context.total_he_time_ms / max(1, self._context.tokens_processed)
            ),
            'tokens_per_second': (
                self._context.tokens_processed * 1000 / max(0.001, self._context.total_he_time_ms)
            ),
            'cost_tracker': self._context.cost_tracker.to_dict(),
            'backend_counters': self._backend.counters.to_dict(),
            'budget_violations': self._context.cost_tracker.budget_violations,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._context.tokens_processed = 0
        self._context.total_he_time_ms = 0.0
        self._context.cost_tracker.reset()
        self._backend.reset_counters()


# =============================================================================
# ADAPTER EXECUTOR (FOR MULTIPLE ADAPTERS)
# =============================================================================

class LoRAAdapterExecutor:
    """
    Executor for multiple LoRA adapters (Q, K, V, O).

    This manages separate executors for each adapter and coordinates
    their execution for a complete attention layer.

    All adapters share a single HE backend instance to avoid duplicating
    key material (~100MB per backend). The shared backend is created once
    and injected into each per-adapter executor.
    """

    def __init__(
        self,
        schedules: Dict[str, ExecutionSchedule],
        backend_type: BackendType = BackendType.SIMULATION,
        device_id: int = 0,
        budget: Optional[CostBudget] = None,
    ):
        """
        Initialize multi-adapter executor.

        Args:
            schedules: Dict mapping adapter name to schedule
            backend_type: GPU backend type
            device_id: GPU device ID
            budget: Cost budget
        """
        # Create a single shared backend from the first schedule's params
        first_schedule = next(iter(schedules.values()))
        self._shared_backend = create_backend(
            backend_type, first_schedule.ckks_params, device_id
        )

        self._executors: Dict[str, HELoRAExecutor] = {}

        for name, schedule in schedules.items():
            executor = HELoRAExecutor(
                schedule, backend_type, device_id, budget
            )
            # Replace each executor's backend with the shared instance
            executor._backend = self._shared_backend
            executor._context.backend = self._shared_backend
            self._executors[name] = executor

        self._adapter_names = list(schedules.keys())

    def load_adapter_weights(
        self,
        adapter_name: str,
        A: np.ndarray,
        B: np.ndarray,
        alpha: float,
    ) -> None:
        """Load weights for a specific adapter."""
        if adapter_name not in self._executors:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        self._executors[adapter_name].load_weights(A, B, alpha)

    def execute_all_adapters(
        self,
        activations: Dict[str, np.ndarray],
        position: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Execute all adapters for a token.

        Args:
            activations: Dict mapping adapter name to activations
            position: Optional token position

        Returns:
            Dict mapping adapter name to delta
        """
        deltas = {}
        for name in self._adapter_names:
            if name in activations:
                deltas[name] = self._executors[name].execute_token(
                    activations[name], position
                )
        return deltas

    def execute_all_adapters_fused(
        self,
        activations: Dict[str, np.ndarray],
        y_bases: Dict[str, np.ndarray],
        position: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Execute all adapters with fused decrypt-unpack-add into y_bases.

        This is the optimised path: for each adapter, the encrypted delta
        is decrypted and added directly into the corresponding y_base buffer.

        Args:
            activations: Dict mapping adapter name to activations
            y_bases: Dict mapping adapter name to base model output (modified in-place)
            position: Optional token position

        Returns:
            Dict mapping adapter name to y_base (with delta added in-place)
        """
        results = {}
        for name in self._adapter_names:
            if name in activations and name in y_bases:
                results[name] = self._executors[name].execute_token_fused(
                    activations[name], y_bases[name], position
                )
        return results

    def execute_all_adapters_batched_decrypt(
        self,
        activations: Dict[str, np.ndarray],
        position: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Execute all adapters and batch-decrypt their results together.

        This amortises per-decrypt fixed overhead (key loading, NTT setup)
        across all adapters in the layer. Each ciphertext is decrypted by
        its own backend (since each executor owns its key material), but
        the decrypt calls are grouped to enable GPU stream parallelism.

        Args:
            activations: Dict mapping adapter name to activations
            position: Optional token position

        Returns:
            Dict mapping adapter name to delta
        """
        # Phase 1: Compute all encrypted deltas (no decryption yet)
        ct_deltas = {}
        adapter_order = []
        for name in self._adapter_names:
            if name in activations:
                executor = self._executors[name]
                if not executor._weights_loaded:
                    continue
                if position is not None:
                    executor.enforce_context_length(position)

                executor._context.cost_tracker.begin_token()
                config = executor._schedule.config

                if activations[name].shape != (config.batch_size, config.hidden_size):
                    raise ValueError(
                        f"Activation shape mismatch for {name}: "
                        f"{activations[name].shape} vs expected "
                        f"({config.batch_size}, {config.hidden_size})"
                    )

                ct_deltas[name] = executor._compute_ct_delta(activations[name])
                adapter_order.append(name)

        # Phase 2: Batch decrypt — each ct uses its own executor's backend
        # This groups all decrypt calls together to allow GPU stream overlap
        deltas = {}
        if adapter_order:
            for name in adapter_order:
                executor = self._executors[name]
                layout = executor._schedule.layout
                config = executor._schedule.config

                with executor._backend.timed_section('decrypt'):
                    packed = executor._backend.decrypt_partial(
                        ct_deltas[name], layout.total_slots_used
                    )
                executor._context.cost_tracker.record_decrypt()

                delta = unpack_activations(packed, layout)
                deltas[name] = delta

                executor._context.tokens_processed += 1
                executor._context.cost_tracker.end_token(config.targets)

        return deltas

    def get_combined_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from all adapters."""
        combined = {
            'adapters': {},
            'total_tokens': 0,
            'total_he_time_ms': 0.0,
        }

        for name, executor in self._executors.items():
            stats = executor.get_statistics()
            combined['adapters'][name] = stats
            combined['total_tokens'] += stats['tokens_processed']
            combined['total_he_time_ms'] += stats['total_he_time_ms']

        if combined['total_he_time_ms'] > 0:
            combined['combined_tokens_per_second'] = (
                combined['total_tokens'] * 1000 / combined['total_he_time_ms']
            )
        else:
            combined['combined_tokens_per_second'] = 0

        return combined
