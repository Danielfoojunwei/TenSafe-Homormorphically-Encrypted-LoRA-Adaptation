"""
Configuration for HE-LoRA Microkernel

This module provides a high-level configuration interface for
the HE-LoRA microkernel, abstracting away the details of CKKS
parameters and packing strategies.

Usage:
    from helora import HELoRAConfig

    config = HELoRAConfig(
        hidden_size=4096,
        lora_rank=16,
        batch_size=8,
    )
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

# Import from compiler
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from he_lora_microkernel.compiler import (
    LoRAConfig,
    LoRATargets,
    CKKSProfile,
    CKKSParams,
    get_profile,
    select_optimal_profile,
    CostBudget,
)


class PerformanceProfile(Enum):
    """High-level performance profile selection."""
    FAST = "fast"              # Minimize latency
    BALANCED = "balanced"      # Balance latency and precision
    PRECISE = "precise"        # Maximize precision


@dataclass
class HELoRAConfig:
    """
    High-level configuration for HE-LoRA microkernel.

    This provides a simple interface for common configurations.
    For advanced use cases, use LoRAConfig directly.
    """
    # Model dimensions
    hidden_size: int
    lora_rank: int = 16

    # LoRA parameters
    lora_alpha: float = None  # Defaults to 2 * rank
    lora_targets: str = "qkv"  # "qkv" or "qkvo"

    # Batch configuration
    batch_size: int = 8
    max_context_length: int = 2048

    # Performance profile
    performance_profile: PerformanceProfile = PerformanceProfile.BALANCED

    # Advanced (usually auto-selected)
    ckks_profile: Optional[CKKSProfile] = None

    # Budget enforcement
    enforce_budget: bool = True
    rotation_budget: int = 16
    keyswitch_budget: int = 16
    rescale_budget: int = 8

    def __post_init__(self):
        """Validate and set defaults."""
        # Default alpha
        if self.lora_alpha is None:
            self.lora_alpha = 2.0 * self.lora_rank

        # Map performance profile to CKKS profile
        if self.ckks_profile is None:
            if self.performance_profile == PerformanceProfile.FAST:
                self.ckks_profile = CKKSProfile.FAST
            elif self.performance_profile == PerformanceProfile.PRECISE:
                self.ckks_profile = CKKSProfile.SAFE
            else:
                # Balanced: auto-select based on parameters
                self.ckks_profile = CKKSProfile.FAST

    def to_lora_config(self) -> LoRAConfig:
        """Convert to low-level LoRAConfig."""
        targets = (
            LoRATargets.QKV if self.lora_targets.lower() == "qkv"
            else LoRATargets.QKVO
        )

        return LoRAConfig(
            hidden_size=self.hidden_size,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            targets=targets,
            batch_size=self.batch_size,
            max_context_length=self.max_context_length,
            ckks_profile=self.ckks_profile,
        )

    def get_ckks_params(self) -> CKKSParams:
        """Get CKKS parameters for this configuration."""
        return get_profile(self.ckks_profile)

    def get_cost_budget(self) -> CostBudget:
        """Get cost budget for this configuration."""
        from he_lora_microkernel.compiler import (
            RotationBudget, KeyswitchBudget, RescaleBudget, CostBudget
        )

        return CostBudget(
            rotation=RotationBudget(
                max_rotations_per_token=self.rotation_budget,
                max_rotations_per_layer=self.rotation_budget * 4,
                max_rotations_qkv=self.rotation_budget * 3,
                max_rotations_qkvo=self.rotation_budget * 4,
            ),
            keyswitch=KeyswitchBudget(
                max_keyswitches_per_token=self.keyswitch_budget,
                max_keyswitches_per_layer=self.keyswitch_budget * 4,
            ),
            rescale=RescaleBudget(
                max_rescales_per_token=self.rescale_budget,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'hidden_size': self.hidden_size,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_targets': self.lora_targets,
            'batch_size': self.batch_size,
            'max_context_length': self.max_context_length,
            'performance_profile': self.performance_profile.value,
            'ckks_profile': self.ckks_profile.value,
            'enforce_budget': self.enforce_budget,
            'rotation_budget': self.rotation_budget,
            'keyswitch_budget': self.keyswitch_budget,
            'rescale_budget': self.rescale_budget,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HELoRAConfig':
        """Deserialize from dictionary."""
        return cls(
            hidden_size=d['hidden_size'],
            lora_rank=d.get('lora_rank', 16),
            lora_alpha=d.get('lora_alpha'),
            lora_targets=d.get('lora_targets', 'qkv'),
            batch_size=d.get('batch_size', 8),
            max_context_length=d.get('max_context_length', 2048),
            performance_profile=PerformanceProfile(
                d.get('performance_profile', 'balanced')
            ),
            ckks_profile=CKKSProfile(d['ckks_profile']) if 'ckks_profile' in d else None,
            enforce_budget=d.get('enforce_budget', True),
            rotation_budget=d.get('rotation_budget', 16),
            keyswitch_budget=d.get('keyswitch_budget', 16),
            rescale_budget=d.get('rescale_budget', 8),
        )


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def llama_7b_config(
    batch_size: int = 8,
    lora_rank: int = 16,
) -> HELoRAConfig:
    """Configuration for Llama-2 7B."""
    return HELoRAConfig(
        hidden_size=4096,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_context_length=4096,
    )


def llama_13b_config(
    batch_size: int = 4,
    lora_rank: int = 16,
) -> HELoRAConfig:
    """Configuration for Llama-2 13B."""
    return HELoRAConfig(
        hidden_size=5120,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_context_length=4096,
    )


def llama_70b_config(
    batch_size: int = 1,
    lora_rank: int = 8,
) -> HELoRAConfig:
    """Configuration for Llama-2 70B."""
    return HELoRAConfig(
        hidden_size=8192,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_context_length=4096,
        performance_profile=PerformanceProfile.FAST,
    )


def mistral_7b_config(
    batch_size: int = 8,
    lora_rank: int = 16,
) -> HELoRAConfig:
    """Configuration for Mistral 7B."""
    return HELoRAConfig(
        hidden_size=4096,
        lora_rank=lora_rank,
        batch_size=batch_size,
        max_context_length=8192,
    )
