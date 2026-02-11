"""
HE-LoRA Microkernel Compiler

This package provides the compilation infrastructure for transforming
LoRA adapters into rotation-minimal HE execution schedules.

Key components:
  - ckks_params: CKKS parameter profiles (FAST/SAFE)
  - packer: SIMD packing strategies for batch-first computation
  - lora_ir: Intermediate representation for LoRA computation
  - scheduler: Rotation-minimal schedule generation
  - cost_model: Cost estimation and budget enforcement
  - emit_artifacts: Artifact serialization and emission

Usage:
    from he_lora_microkernel.compiler import (
        LoRAConfig, LoRATargets,
        CKKSProfile, get_profile,
        compile_schedule,
        CostBudget,
    )

    # Configure LoRA
    config = LoRAConfig(
        hidden_size=4096,
        rank=16,
        alpha=32.0,
        targets=LoRATargets.QKV,
        batch_size=8,
        max_context_length=2048,
        ckks_profile=CKKSProfile.FAST,
    )

    # Get CKKS parameters
    ckks_params = get_profile(CKKSProfile.FAST)

    # Compile schedule
    schedule = compile_schedule(config, ckks_params)

    # Validate against budget
    budget = CostBudget()
    passed, violations = budget.validate_all(...)
"""

# CKKS Parameters
from .ckks_params import (
    CKKSParams,
    CKKSProfile,
    ScheduleCompatibility,
    get_fast_profile,
    get_profile,
    get_safe_profile,
    select_optimal_profile,
    verify_schedule_fits,
)

# Cost Model
from .cost_model import (
    CostBudget,
    CostEstimate,
    CostTracker,
    KeyswitchBudget,
    RescaleBudget,
    RotationBudget,
    check_budget_compliance,
    enforce_rotation_invariant,
    estimate_costs,
    estimate_layer_costs,
)

# Artifact Emission
from .emit_artifacts import (
    ArtifactEncoder,
    ArtifactMetadata,
    CompiledArtifacts,
    compute_artifact_checksum,
    create_artifact_bundle,
    decode_packed_weights,
    emit_cost_report,
    emit_rotation_keys_spec,
    emit_schedule_json,
    encode_packed_weights,
    load_artifacts,
    save_artifacts,
    verify_determinism,
)

# LoRA IR
from .lora_ir import (
    CostPrediction,
    IRBasicBlock,
    IRBuilder,
    IROp,
    IROperand,
    IROpType,
    LoRAConfig,
    LoRAIRModule,
    LoRATargets,
    predict_costs,
)

# Packing
from .packer import (
    BlockSpec,
    PackedLoRAWeights,
    PackedTensor,
    PackingLayout,
    PackingStrategy,
    compute_optimal_block_size,
    compute_packing_layout,
    create_slot_map,
    estimate_rotation_cost,
    pack_activations,
    pack_lora_weights,
    unpack_activations,
    verify_packing_roundtrip,
)

# Scheduler
from .scheduler import (
    ExecutionSchedule,
    LevelPlan,
    RotationSchedule,
    ScheduleStrategy,
    compare_schedules,
    compile_schedule,
    validate_schedule,
)

__all__ = [
    # CKKS
    'CKKSProfile',
    'CKKSParams',
    'get_profile',
    'get_fast_profile',
    'get_safe_profile',
    'select_optimal_profile',
    'verify_schedule_fits',
    'ScheduleCompatibility',
    # Packing
    'PackingStrategy',
    'PackingLayout',
    'BlockSpec',
    'PackedTensor',
    'PackedLoRAWeights',
    'compute_packing_layout',
    'compute_optimal_block_size',
    'pack_activations',
    'unpack_activations',
    'pack_lora_weights',
    'create_slot_map',
    'estimate_rotation_cost',
    'verify_packing_roundtrip',
    # LoRA IR
    'LoRATargets',
    'LoRAConfig',
    'IROpType',
    'IROperand',
    'IROp',
    'IRBasicBlock',
    'LoRAIRModule',
    'IRBuilder',
    'CostPrediction',
    'predict_costs',
    # Scheduler
    'ScheduleStrategy',
    'RotationSchedule',
    'LevelPlan',
    'ExecutionSchedule',
    'compile_schedule',
    'validate_schedule',
    'compare_schedules',
    # Cost Model
    'RotationBudget',
    'KeyswitchBudget',
    'RescaleBudget',
    'CostBudget',
    'CostEstimate',
    'CostTracker',
    'estimate_costs',
    'estimate_layer_costs',
    'check_budget_compliance',
    'enforce_rotation_invariant',
    # Artifacts
    'ArtifactMetadata',
    'CompiledArtifacts',
    'ArtifactEncoder',
    'encode_packed_weights',
    'decode_packed_weights',
    'emit_schedule_json',
    'emit_rotation_keys_spec',
    'emit_cost_report',
    'create_artifact_bundle',
    'save_artifacts',
    'load_artifacts',
    'verify_determinism',
    'compute_artifact_checksum',
]
