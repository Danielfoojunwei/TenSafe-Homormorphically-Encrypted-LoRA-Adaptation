"""
Compilation Interface for HE-LoRA Microkernel

This module provides a high-level compilation interface for
transforming LoRA adapters into executable HE-LoRA microkernels.

Usage:
    from helora import compile_lora

    schedule = compile_lora(
        config=config,
        A=lora_A,
        B=lora_B,
        alpha=lora_alpha,
    )
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from he_lora_microkernel.compiler import (
    LoRAConfig,
    CKKSParams,
    get_profile,
    compile_schedule,
    validate_schedule,
    ExecutionSchedule,
    PackingLayout,
    PackedLoRAWeights,
    pack_lora_weights,
    estimate_costs,
    CostEstimate,
    CostBudget,
    check_budget_compliance,
    create_artifact_bundle,
    save_artifacts,
    CompiledArtifacts,
)

from .config import HELoRAConfig


class CompilationResult:
    """
    Result of HE-LoRA compilation.

    Contains the compiled schedule, packed weights, and cost estimates.
    """

    def __init__(
        self,
        schedule: ExecutionSchedule,
        weights: Optional[PackedLoRAWeights] = None,
        cost_estimate: Optional[CostEstimate] = None,
        budget_check: Optional[Tuple[bool, list]] = None,
    ):
        self.schedule = schedule
        self.weights = weights
        self.cost_estimate = cost_estimate
        self.budget_check = budget_check

    @property
    def is_valid(self) -> bool:
        """Check if compilation produced valid result."""
        return self.schedule.is_valid

    @property
    def budget_passed(self) -> bool:
        """Check if budget constraints passed."""
        if self.budget_check is None:
            return True
        return self.budget_check[0]

    @property
    def violations(self) -> list:
        """Get list of validation/budget violations."""
        violations = list(self.schedule.validation_errors)
        if self.budget_check and not self.budget_check[0]:
            violations.extend(self.budget_check[1])
        return violations

    def summary(self) -> Dict[str, Any]:
        """Get compilation summary."""
        return {
            'valid': self.is_valid,
            'budget_passed': self.budget_passed,
            'violations': self.violations,
            'schedule_hash': self.schedule.schedule_hash,
            'predicted_rotations': (
                self.schedule.predicted_costs.rotations_per_token
                if self.schedule.predicted_costs else None
            ),
            'num_blocks': self.schedule.layout.num_blocks,
            'total_slots_used': self.schedule.layout.total_slots_used,
        }


def compile_lora(
    config: Union[HELoRAConfig, LoRAConfig],
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
    enforce_budget: bool = True,
    budget: Optional[CostBudget] = None,
) -> CompilationResult:
    """
    Compile LoRA adapter into HE-LoRA microkernel.

    Args:
        config: HE-LoRA configuration
        A: Up-projection matrix (optional, for weight packing)
        B: Down-projection matrix (optional, for weight packing)
        alpha: LoRA scaling factor (uses config default if not provided)
        enforce_budget: Whether to check budget constraints
        budget: Custom budget (uses config default if not provided)

    Returns:
        CompilationResult with schedule and optional packed weights
    """
    # Convert to low-level config if needed
    if isinstance(config, HELoRAConfig):
        lora_config = config.to_lora_config()
        ckks_params = config.get_ckks_params()

        if budget is None and enforce_budget:
            budget = config.get_cost_budget()
    else:
        lora_config = config
        ckks_params = get_profile(config.ckks_profile)

    # Compile schedule
    schedule = compile_schedule(lora_config, ckks_params)

    # Validate schedule
    validation_errors = validate_schedule(schedule)
    if validation_errors:
        schedule.validation_errors.extend(validation_errors)

    # Pack weights if provided
    packed_weights = None
    if A is not None and B is not None:
        lora_alpha = alpha if alpha is not None else lora_config.alpha
        packed_weights = pack_lora_weights(
            A, B, lora_alpha, schedule.layout
        )

    # Estimate costs
    cost_estimate = estimate_costs(
        lora_config,
        schedule.layout,
        lora_config.ckks_profile,
    )

    # Check budget
    budget_check = None
    if enforce_budget and budget is not None:
        budget_check = check_budget_compliance(
            cost_estimate,
            budget,
            lora_config.targets,
        )

    return CompilationResult(
        schedule=schedule,
        weights=packed_weights,
        cost_estimate=cost_estimate,
        budget_check=budget_check,
    )


def compile_and_save(
    config: Union[HELoRAConfig, LoRAConfig],
    output_dir: Union[str, Path],
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
) -> Path:
    """
    Compile LoRA adapter and save artifacts to disk.

    Args:
        config: HE-LoRA configuration
        output_dir: Directory to save artifacts
        A: Up-projection matrix (optional)
        B: Down-projection matrix (optional)
        alpha: LoRA scaling factor

    Returns:
        Path to artifact directory
    """
    # Compile
    result = compile_lora(config, A, B, alpha)

    if not result.is_valid:
        raise ValueError(
            f"Compilation failed: {result.violations}"
        )

    # Create artifact bundle
    bundle = create_artifact_bundle(
        schedule=result.schedule,
        weights=result.weights,
        cost_estimate=result.cost_estimate,
    )

    # Save
    artifact_path = save_artifacts(bundle, output_dir)

    return artifact_path


def validate_compilation(
    config: Union[HELoRAConfig, LoRAConfig],
    budget: Optional[CostBudget] = None,
) -> Dict[str, Any]:
    """
    Validate that configuration can be compiled within budget.

    This is useful for CI to verify that configurations are valid
    before running full compilation.

    Args:
        config: HE-LoRA configuration
        budget: Cost budget (uses default if not provided)

    Returns:
        Validation result dict
    """
    result = compile_lora(config, enforce_budget=True, budget=budget)

    return {
        'valid': result.is_valid,
        'budget_passed': result.budget_passed,
        'violations': result.violations,
        'summary': result.summary(),
    }


def recompile_for_batch_size(
    original_config: Union[HELoRAConfig, LoRAConfig],
    new_batch_size: int,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
) -> CompilationResult:
    """
    Recompile schedule for new batch size.

    Args:
        original_config: Original configuration
        new_batch_size: New batch size
        A: Up-projection matrix
        B: Down-projection matrix
        alpha: LoRA scaling factor

    Returns:
        CompilationResult for new batch size
    """
    if isinstance(original_config, HELoRAConfig):
        new_config = HELoRAConfig(
            hidden_size=original_config.hidden_size,
            lora_rank=original_config.lora_rank,
            lora_alpha=original_config.lora_alpha,
            lora_targets=original_config.lora_targets,
            batch_size=new_batch_size,
            max_context_length=original_config.max_context_length,
            performance_profile=original_config.performance_profile,
            ckks_profile=original_config.ckks_profile,
        )
    else:
        from he_lora_microkernel.compiler import LoRAConfig
        new_config = LoRAConfig(
            hidden_size=original_config.hidden_size,
            rank=original_config.rank,
            alpha=original_config.alpha,
            targets=original_config.targets,
            batch_size=new_batch_size,
            max_context_length=original_config.max_context_length,
            ckks_profile=original_config.ckks_profile,
        )

    return compile_lora(new_config, A, B, alpha)


def compare_configurations(
    configs: Dict[str, Union[HELoRAConfig, LoRAConfig]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compare compilation results across configurations.

    Args:
        configs: Dict mapping name to config

    Returns:
        Comparison results
    """
    results = {}

    for name, config in configs.items():
        result = compile_lora(config)
        results[name] = {
            'valid': result.is_valid,
            'budget_passed': result.budget_passed,
            'summary': result.summary(),
            'cost_estimate': result.cost_estimate.to_dict() if result.cost_estimate else None,
        }

    return results
