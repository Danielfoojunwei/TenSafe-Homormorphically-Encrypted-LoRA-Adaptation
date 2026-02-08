"""
Rotation-Minimal Execution Scheduler for HE-LoRA Microkernel

This module generates optimized execution schedules for LoRA computations
that minimize rotations following MOAI principles:

  1. Column packing for rotation-free Ct×Pt
  2. Tree reduction for accumulation
  3. Fused operations to reduce memory traffic
  4. Explicit level tracking to prevent depth overflow

The scheduler outputs a deterministic execution plan that can be
directly executed by the runtime.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from .ckks_params import CKKSParams, verify_schedule_fits
from .lora_ir import (
    CostPrediction,
    IRBuilder,
    IROpType,
    LoRAConfig,
    LoRAIRModule,
    predict_costs,
)
from .packer import PackingLayout, compute_packing_layout

# =============================================================================
# SCHEDULE TYPES
# =============================================================================

class ScheduleStrategy(Enum):
    """Scheduling strategy for LoRA computation."""
    MOAI_CPMM = "moai_cpmm"           # MOAI Column-Packed Matrix Multiply
    DIAGONAL = "diagonal"             # Halevi-Shoup diagonal method
    BABY_GIANT = "baby_giant"         # Baby-step giant-step optimization


@dataclass
class RotationSchedule:
    """
    Explicit rotation schedule for a computation.

    This captures exactly which rotations will be performed,
    enabling rotation budget verification.
    """
    # List of rotation amounts needed
    rotation_steps: List[int]

    # Total rotation count
    total_rotations: int

    # Required Galois keys
    required_galois_keys: List[int]

    # Strategy used
    strategy: ScheduleStrategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rotation_steps': self.rotation_steps,
            'total_rotations': self.total_rotations,
            'required_galois_keys': self.required_galois_keys,
            'strategy': self.strategy.value,
        }


@dataclass
class LevelPlan:
    """
    Plan for managing ciphertext levels through computation.

    Tracks when rescales and modswitches are needed.
    """
    # Level at each computation step
    level_sequence: List[int]

    # Rescale points (step indices)
    rescale_points: List[int]

    # Modswitch points (step index, target level)
    modswitch_points: List[Tuple[int, int]]

    # Final level
    final_level: int

    # Maximum level reached
    max_level: int

    def validate(self, max_depth: int) -> bool:
        """Check if plan fits within depth budget."""
        return self.max_level <= max_depth


@dataclass
class ExecutionSchedule:
    """
    Complete execution schedule for a LoRA adapter.

    This is the final output of the scheduler, ready for runtime execution.
    """
    # Configuration
    config: LoRAConfig
    ckks_params: CKKSParams
    layout: PackingLayout

    # Compiled IR module
    ir_module: LoRAIRModule

    # Sub-schedules
    rotation_schedule: RotationSchedule
    level_plan: LevelPlan

    # Pre-computed costs
    predicted_costs: CostPrediction

    # Determinism
    schedule_hash: str = ""

    # Validity
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'ckks_params': self.ckks_params.to_dict(),
            'layout': self.layout.to_dict(),
            'ir_module': self.ir_module.to_dict(),
            'rotation_schedule': self.rotation_schedule.to_dict(),
            'level_plan': {
                'level_sequence': self.level_plan.level_sequence,
                'rescale_points': self.level_plan.rescale_points,
                'modswitch_points': self.level_plan.modswitch_points,
                'final_level': self.level_plan.final_level,
                'max_level': self.level_plan.max_level,
            },
            'predicted_costs': self.predicted_costs.to_dict(),
            'schedule_hash': self.schedule_hash,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors,
        }


# =============================================================================
# MOAI CPMM SCHEDULER
# =============================================================================

def compute_moai_rotation_schedule(
    layout: PackingLayout,
) -> RotationSchedule:
    """
    Compute rotation schedule using MOAI Column-Packed Matrix Multiply.

    MOAI CPMM achieves rotation-free Ct×Pt by column packing.
    Rotations are only needed for:
      1. Cross-block accumulation (if multiple blocks)
      2. Final result collection

    Args:
        layout: Packing layout

    Returns:
        Rotation schedule
    """
    rotation_steps = []

    # For MOAI CPMM, intra-block operations are rotation-free
    # Cross-block accumulation uses tree reduction
    if layout.num_blocks > 1:
        # Tree reduction: log2(num_blocks) rotation rounds
        stride = 1
        while stride < layout.num_blocks:
            # Rotation to align blocks for addition
            rotation_amount = stride * layout.block_size * layout.batch_size
            rotation_steps.append(rotation_amount)
            stride *= 2

    # Compute required Galois keys
    galois_keys = list(set(abs(r) for r in rotation_steps))
    galois_keys.sort()

    return RotationSchedule(
        rotation_steps=rotation_steps,
        total_rotations=len(rotation_steps),
        required_galois_keys=galois_keys,
        strategy=ScheduleStrategy.MOAI_CPMM,
    )


def compute_level_plan(
    layout: PackingLayout,
    ckks_params: CKKSParams,
) -> LevelPlan:
    """
    Compute level management plan for LoRA computation.

    LoRA: Δy = A(Bx)
    Operations:
      1. Encrypt x (level 0)
      2. B × x (mul_plain) → rescale (level 1)
      3. A × (Bx) (mul_plain) → rescale (level 2)

    Args:
        layout: Packing layout
        ckks_params: CKKS parameters

    Returns:
        Level plan
    """
    level_sequence = []
    rescale_points = []
    modswitch_points = []

    current_level = 0

    # Step 0: Encrypt
    level_sequence.append(current_level)

    # Step 1: First matmul (B × x) + rescale
    # All blocks undergo mul_plain + rescale, moving from L0 -> L1 simultaneously
    current_level += 1
    for block_idx in range(layout.num_blocks):
        level_sequence.append(current_level)
        rescale_points.append(len(level_sequence) - 1)

    # Accumulation across blocks (additions, no level change)
    if layout.num_blocks > 1:
        for _ in range(int(math.log2(layout.num_blocks))):
            level_sequence.append(current_level)

    # Step 2: Second matmul (A × intermediate) + rescale
    current_level += 1
    level_sequence.append(current_level)
    rescale_points.append(len(level_sequence) - 1)

    return LevelPlan(
        level_sequence=level_sequence,
        rescale_points=rescale_points,
        modswitch_points=modswitch_points,
        final_level=current_level,
        max_level=current_level,
    )


# =============================================================================
# IR GENERATION
# =============================================================================

def generate_lora_ir(
    config: LoRAConfig,
    ckks_params: CKKSParams,
    layout: PackingLayout,
) -> LoRAIRModule:
    """
    Generate IR for LoRA computation.

    This creates the complete IR module with all operations
    for computing Δy = A(Bx).

    Args:
        config: LoRA configuration
        ckks_params: CKKS parameters
        layout: Packing layout

    Returns:
        Compiled IR module
    """
    module = LoRAIRModule(
        config=config,
        ckks_params=ckks_params,
        packing_layout=layout,
    )
    builder = IRBuilder(module)

    # === Block 0: Setup (encode plaintexts) ===
    setup_block = module.add_block("setup")
    builder.set_block(setup_block)
    builder.comment("Pre-encode LoRA weight blocks")

    # Encode B matrix blocks
    B_blocks = []
    for i in range(layout.num_blocks):
        pt = builder.encode_plaintext(f"pt_B_block{i}", f"B_block_{i}_values")
        B_blocks.append(pt)

    # Encode A matrix blocks
    A_blocks = []
    for i in range(layout.num_blocks):
        pt = builder.encode_plaintext(f"pt_A_block{i}", f"A_block_{i}_values")
        A_blocks.append(pt)

    # === Block 1: Encrypt input ===
    encrypt_block = module.add_block("encrypt")
    encrypt_block.depends_on.add(setup_block.block_id)
    builder.set_block(encrypt_block)
    builder.comment("Encrypt batch activations")

    ct_x = builder.encrypt("packed_activations")

    # === Block 2: First matmul (B @ x) ===
    matmul1_block = module.add_block("matmul_Bx")
    matmul1_block.depends_on.add(encrypt_block.block_id)
    builder.set_block(matmul1_block)
    builder.comment("Compute B @ x (down projection)")

    # Multiply each block and accumulate
    intermediate_results = []
    for i, B_pt in enumerate(B_blocks):
        builder.comment(f"Block {i}: B[{i}] × x[{i}]")
        result = builder.mul_plain(ct_x, B_pt, rescale=True)
        intermediate_results.append(result)

    # Tree reduction for accumulation
    builder.comment("Accumulate block results (tree reduction)")
    while len(intermediate_results) > 1:
        new_results = []
        for i in range(0, len(intermediate_results), 2):
            if i + 1 < len(intermediate_results):
                # Align levels if needed
                ct1 = intermediate_results[i]
                ct2 = intermediate_results[i + 1]
                if (ct1.level or 0) != (ct2.level or 0):
                    target = max(ct1.level or 0, ct2.level or 0)
                    ct1 = builder.modswitch_to_level(ct1, target)
                    ct2 = builder.modswitch_to_level(ct2, target)
                combined = builder.add(ct1, ct2)
                new_results.append(combined)
            else:
                new_results.append(intermediate_results[i])
        intermediate_results = new_results

    ct_Bx = intermediate_results[0]
    builder.level_check(ct_Bx, 1)

    # === Block 3: Second matmul (A @ Bx) ===
    matmul2_block = module.add_block("matmul_ABx")
    matmul2_block.depends_on.add(matmul1_block.block_id)
    builder.set_block(matmul2_block)
    builder.comment("Compute A @ (Bx) (up projection)")

    # For MOAI CPMM, A is pre-packed to match the layout
    # Here we use a simplified single multiplication
    # In practice, this would be blocked similarly to B@x
    result = builder.mul_plain(ct_Bx, A_blocks[0], rescale=True)

    # Accumulate remaining A blocks
    for i in range(1, len(A_blocks)):
        block_result = builder.mul_plain(ct_Bx, A_blocks[i], rescale=True)
        result = builder.modswitch_to_level(result, block_result.level or 0)
        result = builder.add(result, block_result)

    ct_delta = result
    builder.level_check(ct_delta, 2)

    # === Block 4: Decrypt output ===
    decrypt_block = module.add_block("decrypt")
    decrypt_block.depends_on.add(matmul2_block.block_id)
    builder.set_block(decrypt_block)
    builder.comment("Decrypt LoRA delta")

    pt_delta = builder.decrypt(ct_delta)
    builder.sync()

    # Finalize module
    module.compute_costs()
    module.compute_hash()

    return module


# =============================================================================
# MAIN SCHEDULER
# =============================================================================

def compile_schedule(
    config: LoRAConfig,
    ckks_params: CKKSParams,
    strategy: ScheduleStrategy = ScheduleStrategy.MOAI_CPMM,
) -> ExecutionSchedule:
    """
    Compile a complete execution schedule for LoRA computation.

    This is the main entry point for the scheduler. It:
      1. Validates parameters
      2. Computes packing layout
      3. Generates rotation schedule
      4. Plans level management
      5. Generates IR
      6. Validates the complete schedule

    Args:
        config: LoRA configuration
        ckks_params: CKKS parameters
        strategy: Scheduling strategy

    Returns:
        Complete execution schedule

    Raises:
        ValueError: If schedule cannot be compiled (e.g., depth overflow)
    """
    validation_errors = []

    # Validate CKKS parameters
    ckks_params.validate()

    # Compute packing layout
    layout = compute_packing_layout(
        hidden_size=config.hidden_size,
        lora_rank=config.rank,
        batch_size=config.batch_size,
        params=ckks_params,
    )

    # Verify schedule fits
    compat = verify_schedule_fits(
        params=ckks_params,
        hidden_size=config.hidden_size,
        lora_rank=config.rank,
        batch_size=config.batch_size,
        block_size=layout.block_size,
    )

    if not compat.compatible:
        validation_errors.append(compat.error_message or "Schedule doesn't fit")

    # Compute rotation schedule
    rotation_schedule = compute_moai_rotation_schedule(layout)

    # Compute level plan
    level_plan = compute_level_plan(layout, ckks_params)

    if not level_plan.validate(ckks_params.max_depth):
        validation_errors.append(
            f"Level plan exceeds depth: {level_plan.max_level} > {ckks_params.max_depth}. "
            f"Bootstrapping NOT supported."
        )

    # Generate IR
    ir_module = generate_lora_ir(config, ckks_params, layout)

    # Predict costs
    predicted_costs = predict_costs(config, layout)

    # Build schedule
    schedule = ExecutionSchedule(
        config=config,
        ckks_params=ckks_params,
        layout=layout,
        ir_module=ir_module,
        rotation_schedule=rotation_schedule,
        level_plan=level_plan,
        predicted_costs=predicted_costs,
        is_valid=len(validation_errors) == 0,
        validation_errors=validation_errors,
    )

    # Compute deterministic hash
    schedule.schedule_hash = ir_module.module_hash

    return schedule


def validate_schedule(schedule: ExecutionSchedule) -> List[str]:
    """
    Validate a compiled schedule against all constraints.

    Args:
        schedule: Compiled schedule

    Returns:
        List of validation errors (empty if valid)
    """
    errors = list(schedule.validation_errors)

    # Check rotation budget
    if schedule.rotation_schedule.total_rotations > schedule.layout.num_blocks * 4:
        errors.append(
            f"Rotation budget exceeded: {schedule.rotation_schedule.total_rotations} "
            f"rotations for {schedule.layout.num_blocks} blocks"
        )

    # Check level consistency in IR
    for block in schedule.ir_module.blocks:
        for op in block.operations:
            if op.op_type == IROpType.LEVEL_CHECK:
                expected = op.attributes.get('expected_level', 0)
                operand = op.operands[0] if op.operands else None
                if operand and operand.level != expected:
                    errors.append(
                        f"Level mismatch in {block.name}: expected {expected}, "
                        f"got {operand.level}"
                    )

    # Check IR completeness
    if len(schedule.ir_module.blocks) < 4:
        errors.append("IR module incomplete: missing required blocks")

    return errors


# =============================================================================
# SCHEDULE COMPARISON
# =============================================================================

def compare_schedules(
    schedule1: ExecutionSchedule,
    schedule2: ExecutionSchedule,
) -> Dict[str, Any]:
    """
    Compare two schedules for cost analysis.

    Args:
        schedule1: First schedule
        schedule2: Second schedule

    Returns:
        Comparison results
    """
    return {
        'rotation_diff': (
            schedule1.rotation_schedule.total_rotations -
            schedule2.rotation_schedule.total_rotations
        ),
        'level_diff': (
            schedule1.level_plan.max_level -
            schedule2.level_plan.max_level
        ),
        'hash_match': schedule1.schedule_hash == schedule2.schedule_hash,
        'schedule1': {
            'rotations': schedule1.rotation_schedule.total_rotations,
            'max_level': schedule1.level_plan.max_level,
        },
        'schedule2': {
            'rotations': schedule2.rotation_schedule.total_rotations,
            'max_level': schedule2.level_plan.max_level,
        },
    }
