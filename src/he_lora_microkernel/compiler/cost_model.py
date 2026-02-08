"""
Cost Model for HE-LoRA Microkernel

This module provides cost estimation and budget enforcement for
rotation-minimal HE-LoRA computation.

Key responsibilities:
  1. Estimate costs before compilation
  2. Enforce rotation/keyswitch/rescale budgets
  3. Track actual costs during execution
  4. Provide CI-enforceable invariants

The cost model is critical for MOAI-style optimization - rotation
minimization only matters if we can measure and enforce it.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ckks_params import CKKSProfile
from .lora_ir import LoRAConfig, LoRATargets
from .packer import PackingLayout

# =============================================================================
# COST BUDGETS
# =============================================================================

@dataclass
class RotationBudget:
    """
    Rotation budget for CI enforcement.

    These are hard limits that cause CI to fail if exceeded.
    """
    # Per-token rotation limit
    max_rotations_per_token: int

    # Per-layer rotation limit (all adapters)
    max_rotations_per_layer: int

    # Total rotations for QKV vs QKVO
    max_rotations_qkv: int   # For targets=QKV
    max_rotations_qkvo: int  # For targets=QKVO

    def validate(
        self,
        actual_per_token: int,
        actual_per_layer: int,
        targets: LoRATargets,
    ) -> Tuple[bool, List[str]]:
        """
        Validate actual costs against budget.

        Returns:
            (passed, list of violations)
        """
        violations = []

        if actual_per_token > self.max_rotations_per_token:
            violations.append(
                f"Rotation budget exceeded: {actual_per_token} > "
                f"{self.max_rotations_per_token} per token"
            )

        if actual_per_layer > self.max_rotations_per_layer:
            violations.append(
                f"Layer rotation budget exceeded: {actual_per_layer} > "
                f"{self.max_rotations_per_layer} per layer"
            )

        target_max = (
            self.max_rotations_qkv if targets == LoRATargets.QKV
            else self.max_rotations_qkvo
        )
        if actual_per_layer > target_max:
            violations.append(
                f"Target-specific rotation budget exceeded: {actual_per_layer} > "
                f"{target_max} for {targets.value}"
            )

        return len(violations) == 0, violations


@dataclass
class KeyswitchBudget:
    """Key switching budget."""
    max_keyswitches_per_token: int
    max_keyswitches_per_layer: int

    def validate(
        self,
        actual_per_token: int,
        actual_per_layer: int,
    ) -> Tuple[bool, List[str]]:
        violations = []
        if actual_per_token > self.max_keyswitches_per_token:
            violations.append(
                f"Keyswitch budget exceeded: {actual_per_token} > "
                f"{self.max_keyswitches_per_token} per token"
            )
        if actual_per_layer > self.max_keyswitches_per_layer:
            violations.append(
                f"Layer keyswitch budget exceeded: {actual_per_layer} > "
                f"{self.max_keyswitches_per_layer} per layer"
            )
        return len(violations) == 0, violations


@dataclass
class RescaleBudget:
    """Rescale budget."""
    max_rescales_per_token: int

    def validate(self, actual: int) -> Tuple[bool, List[str]]:
        violations = []
        if actual > self.max_rescales_per_token:
            violations.append(
                f"Rescale budget exceeded: {actual} > "
                f"{self.max_rescales_per_token} per token"
            )
        return len(violations) == 0, violations


@dataclass
class CostBudget:
    """
    Complete cost budget for CI enforcement.

    Default values are initial targets that should be achievable
    with MOAI-style optimization.
    """
    rotation: RotationBudget = field(default_factory=lambda: RotationBudget(
        max_rotations_per_token=16,    # R_max per token
        max_rotations_per_layer=64,    # Per layer
        max_rotations_qkv=48,          # QKV target
        max_rotations_qkvo=64,         # QKVO target
    ))

    keyswitch: KeyswitchBudget = field(default_factory=lambda: KeyswitchBudget(
        max_keyswitches_per_token=16,  # K_max per token
        max_keyswitches_per_layer=64,  # Per layer
    ))

    rescale: RescaleBudget = field(default_factory=lambda: RescaleBudget(
        max_rescales_per_token=8,      # S_max per token
    ))

    def validate_all(
        self,
        rotations_per_token: int,
        rotations_per_layer: int,
        keyswitches_per_token: int,
        keyswitches_per_layer: int,
        rescales_per_token: int,
        targets: LoRATargets,
    ) -> Tuple[bool, List[str]]:
        """Validate all budgets at once."""
        all_violations = []

        passed, violations = self.rotation.validate(
            rotations_per_token, rotations_per_layer, targets
        )
        all_violations.extend(violations)

        passed, violations = self.keyswitch.validate(
            keyswitches_per_token, keyswitches_per_layer
        )
        all_violations.extend(violations)

        passed, violations = self.rescale.validate(rescales_per_token)
        all_violations.extend(violations)

        return len(all_violations) == 0, all_violations

    @classmethod
    def strict(cls) -> 'CostBudget':
        """Strict budget for production."""
        return cls(
            rotation=RotationBudget(
                max_rotations_per_token=8,
                max_rotations_per_layer=32,
                max_rotations_qkv=24,
                max_rotations_qkvo=32,
            ),
            keyswitch=KeyswitchBudget(
                max_keyswitches_per_token=8,
                max_keyswitches_per_layer=32,
            ),
            rescale=RescaleBudget(
                max_rescales_per_token=4,
            ),
        )

    @classmethod
    def relaxed(cls) -> 'CostBudget':
        """Relaxed budget for development."""
        return cls(
            rotation=RotationBudget(
                max_rotations_per_token=64,
                max_rotations_per_layer=256,
                max_rotations_qkv=192,
                max_rotations_qkvo=256,
            ),
            keyswitch=KeyswitchBudget(
                max_keyswitches_per_token=64,
                max_keyswitches_per_layer=256,
            ),
            rescale=RescaleBudget(
                max_rescales_per_token=16,
            ),
        )


# =============================================================================
# COST ESTIMATION
# =============================================================================

@dataclass
class CostEstimate:
    """Estimated costs for a LoRA computation."""

    # Per-operation costs (microseconds on reference hardware)
    rotation_cost_us: float = 500.0     # Most expensive
    keyswitch_cost_us: float = 500.0    # Same as rotation (required for rotation)
    rescale_cost_us: float = 50.0       # Cheaper
    mul_plain_cost_us: float = 100.0    # Ct×Pt
    add_cost_us: float = 20.0           # Cheap
    encrypt_cost_us: float = 200.0
    decrypt_cost_us: float = 200.0

    # Optimized decrypt variants (lower cost due to reduced work)
    decrypt_partial_cost_us: float = 120.0      # Partial decrypt: skip unused slots
    decrypt_fused_add_cost_us: float = 100.0    # Fused decrypt+unpack+add: single memory pass
    decrypt_batch_overhead_us: float = 50.0     # Per-ciphertext cost in batch mode (amortised)

    # Operation counts
    num_rotations: int = 0
    num_keyswitches: int = 0
    num_rescales: int = 0
    num_mul_plain: int = 0
    num_add: int = 0
    num_encrypt: int = 1
    num_decrypt: int = 1

    @property
    def total_compute_us(self) -> float:
        """Total compute time estimate in microseconds."""
        return (
            self.num_rotations * self.rotation_cost_us +
            self.num_keyswitches * self.keyswitch_cost_us +
            self.num_rescales * self.rescale_cost_us +
            self.num_mul_plain * self.mul_plain_cost_us +
            self.num_add * self.add_cost_us
        )

    @property
    def total_crypto_us(self) -> float:
        """Total encrypt/decrypt time (using optimised decrypt cost)."""
        return (
            self.num_encrypt * self.encrypt_cost_us +
            self.num_decrypt * self.decrypt_fused_add_cost_us
        )

    @property
    def total_us(self) -> float:
        """Total estimated time."""
        return self.total_compute_us + self.total_crypto_us

    @property
    def rotation_percentage(self) -> float:
        """Percentage of time spent on rotations."""
        if self.total_compute_us == 0:
            return 0.0
        return (self.num_rotations * self.rotation_cost_us) / self.total_compute_us * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_counts': {
                'rotations': self.num_rotations,
                'keyswitches': self.num_keyswitches,
                'rescales': self.num_rescales,
                'mul_plain': self.num_mul_plain,
                'add': self.num_add,
                'encrypt': self.num_encrypt,
                'decrypt': self.num_decrypt,
            },
            'time_estimates_us': {
                'compute': self.total_compute_us,
                'crypto': self.total_crypto_us,
                'total': self.total_us,
                'rotation_percentage': self.rotation_percentage,
            },
        }


def estimate_costs(
    config: LoRAConfig,
    layout: PackingLayout,
    ckks_profile: CKKSProfile = CKKSProfile.FAST,
) -> CostEstimate:
    """
    Estimate costs for a LoRA computation.

    Args:
        config: LoRA configuration
        layout: Packing layout
        ckks_profile: CKKS profile (affects operation costs)

    Returns:
        Cost estimate
    """
    # Adjust costs based on profile
    cost_multiplier = 1.0 if ckks_profile == CKKSProfile.FAST else 1.5

    estimate = CostEstimate()

    # Scale costs by profile
    estimate.rotation_cost_us *= cost_multiplier
    estimate.keyswitch_cost_us *= cost_multiplier
    estimate.rescale_cost_us *= cost_multiplier
    estimate.mul_plain_cost_us *= cost_multiplier

    # Count operations for LoRA: Δy = A(Bx)

    # First matmul (B × x): one mul_plain per block + rescale
    estimate.num_mul_plain = layout.num_blocks
    estimate.num_rescales = layout.num_blocks

    # Block accumulation: tree reduction
    if layout.num_blocks > 1:
        accumulation_rounds = int(math.ceil(math.log2(layout.num_blocks)))
        estimate.num_add = layout.num_blocks - 1

        # Rotations for cross-block alignment
        estimate.num_rotations = accumulation_rounds
        estimate.num_keyswitches = accumulation_rounds

    # Second matmul (A × intermediate)
    estimate.num_mul_plain += 1  # Simplified: single block for rank output
    estimate.num_rescales += 1

    return estimate


def estimate_layer_costs(
    config: LoRAConfig,
    layout: PackingLayout,
) -> CostEstimate:
    """
    Estimate costs for all adapters in a layer.

    Args:
        config: LoRA configuration
        layout: Packing layout

    Returns:
        Aggregated cost estimate
    """
    single_adapter = estimate_costs(config, layout, config.ckks_profile)

    # Multiply by number of adapters
    num_adapters = config.num_adapters

    return CostEstimate(
        rotation_cost_us=single_adapter.rotation_cost_us,
        keyswitch_cost_us=single_adapter.keyswitch_cost_us,
        rescale_cost_us=single_adapter.rescale_cost_us,
        mul_plain_cost_us=single_adapter.mul_plain_cost_us,
        add_cost_us=single_adapter.add_cost_us,
        encrypt_cost_us=single_adapter.encrypt_cost_us,
        decrypt_cost_us=single_adapter.decrypt_cost_us,
        num_rotations=single_adapter.num_rotations * num_adapters,
        num_keyswitches=single_adapter.num_keyswitches * num_adapters,
        num_rescales=single_adapter.num_rescales * num_adapters,
        num_mul_plain=single_adapter.num_mul_plain * num_adapters,
        num_add=single_adapter.num_add * num_adapters,
        num_encrypt=num_adapters,
        num_decrypt=num_adapters,
    )


# =============================================================================
# RUNTIME COST TRACKING
# =============================================================================

@dataclass
class CostTracker:
    """
    Runtime cost tracker for monitoring actual costs.

    This tracks actual operation counts during execution and
    compares against budgets.
    """
    # Accumulated counts
    total_rotations: int = 0
    total_keyswitches: int = 0
    total_rescales: int = 0
    total_mul_plain: int = 0
    total_add: int = 0
    total_encryptions: int = 0
    total_decryptions: int = 0

    # Per-token tracking
    current_token_rotations: int = 0
    current_token_keyswitches: int = 0
    current_token_rescales: int = 0

    # Token count
    tokens_processed: int = 0

    # Budget for validation
    budget: Optional[CostBudget] = None

    # Violation tracking
    budget_violations: List[str] = field(default_factory=list)

    def reset(self) -> None:
        """Reset all counters."""
        self.total_rotations = 0
        self.total_keyswitches = 0
        self.total_rescales = 0
        self.total_mul_plain = 0
        self.total_add = 0
        self.total_encryptions = 0
        self.total_decryptions = 0
        self.current_token_rotations = 0
        self.current_token_keyswitches = 0
        self.current_token_rescales = 0
        self.tokens_processed = 0
        self.budget_violations.clear()

    def begin_token(self) -> None:
        """Start tracking a new token."""
        self.current_token_rotations = 0
        self.current_token_keyswitches = 0
        self.current_token_rescales = 0

    def end_token(self, targets: LoRATargets = LoRATargets.QKV) -> None:
        """Finalize token tracking and validate budget."""
        self.tokens_processed += 1

        if self.budget:
            # Calculate per-layer (assuming single layer for this token)
            per_layer_rotations = self.current_token_rotations
            per_layer_keyswitches = self.current_token_keyswitches

            passed, violations = self.budget.validate_all(
                rotations_per_token=self.current_token_rotations,
                rotations_per_layer=per_layer_rotations,
                keyswitches_per_token=self.current_token_keyswitches,
                keyswitches_per_layer=per_layer_keyswitches,
                rescales_per_token=self.current_token_rescales,
                targets=targets,
            )

            self.budget_violations.extend(violations)

    def record_rotation(self, count: int = 1) -> None:
        """Record rotation operations."""
        self.total_rotations += count
        self.current_token_rotations += count

    def record_keyswitch(self, count: int = 1) -> None:
        """Record keyswitch operations."""
        self.total_keyswitches += count
        self.current_token_keyswitches += count

    def record_rescale(self, count: int = 1) -> None:
        """Record rescale operations."""
        self.total_rescales += count
        self.current_token_rescales += count

    def record_mul_plain(self, count: int = 1) -> None:
        """Record plaintext multiplication."""
        self.total_mul_plain += count

    def record_add(self, count: int = 1) -> None:
        """Record addition."""
        self.total_add += count

    def record_encrypt(self, count: int = 1) -> None:
        """Record encryption."""
        self.total_encryptions += count

    def record_decrypt(self, count: int = 1) -> None:
        """Record decryption."""
        self.total_decryptions += count

    @property
    def avg_rotations_per_token(self) -> float:
        """Average rotations per token."""
        if self.tokens_processed == 0:
            return 0.0
        return self.total_rotations / self.tokens_processed

    @property
    def has_violations(self) -> bool:
        """Check if any budget violations occurred."""
        return len(self.budget_violations) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'totals': {
                'rotations': self.total_rotations,
                'keyswitches': self.total_keyswitches,
                'rescales': self.total_rescales,
                'mul_plain': self.total_mul_plain,
                'add': self.total_add,
                'encryptions': self.total_encryptions,
                'decryptions': self.total_decryptions,
            },
            'tokens_processed': self.tokens_processed,
            'averages': {
                'rotations_per_token': self.avg_rotations_per_token,
            },
            'budget_violations': self.budget_violations,
            'has_violations': self.has_violations,
        }


# =============================================================================
# CI ENFORCEMENT
# =============================================================================

def check_budget_compliance(
    estimate: CostEstimate,
    budget: CostBudget,
    targets: LoRATargets,
) -> Tuple[bool, List[str]]:
    """
    Check if estimated costs comply with budget.

    This is used by CI to reject schedules that exceed budgets.

    Args:
        estimate: Cost estimate
        budget: Cost budget
        targets: LoRA targets

    Returns:
        (passed, violations)
    """
    return budget.validate_all(
        rotations_per_token=estimate.num_rotations,
        rotations_per_layer=estimate.num_rotations,  # Simplified
        keyswitches_per_token=estimate.num_keyswitches,
        keyswitches_per_layer=estimate.num_keyswitches,
        rescales_per_token=estimate.num_rescales,
        targets=targets,
    )


def enforce_rotation_invariant(
    actual_rotations: int,
    expected_rotations: int,
    tolerance: float = 0.0,
) -> Tuple[bool, str]:
    """
    Enforce rotation count invariant.

    CI should fail if actual rotations exceed expected by more than tolerance.

    Args:
        actual_rotations: Actual rotation count
        expected_rotations: Expected rotation count
        tolerance: Allowed deviation (fraction)

    Returns:
        (passed, message)
    """
    max_allowed = int(expected_rotations * (1 + tolerance))

    if actual_rotations > max_allowed:
        return False, (
            f"Rotation invariant violated: {actual_rotations} > "
            f"{max_allowed} (expected {expected_rotations} ± {tolerance*100}%)"
        )

    return True, f"Rotation invariant passed: {actual_rotations} <= {max_allowed}"
