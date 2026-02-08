"""
TenSafe LoRA Best Practices Module

This module implements research-backed best practices for LoRA fine-tuning,
based on findings from:
- "LoRA Without Regret" (Schulman et al., 2025)
- DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., ICML 2024)
- LoRA+: Efficient Low Rank Adaptation (Hayou et al., ICML 2024)
- rsLoRA: Rank-Stabilized LoRA (Kalajdzievski, 2023)
- QLoRA, AdaLoRA, VeRA, and other variants

Key improvements over standard LoRA:
1. Apply to ALL layers (especially MLPs) - not just attention
2. Use α = 2 * rank for optimal scaling
3. rsLoRA scaling (α/√r) for high-rank stability
4. LoRA+ separate learning rates for A and B matrices
5. DoRA weight decomposition for better learning dynamics
6. Adaptive rank selection based on dataset size
7. FFA-LoRA for privacy-preserving federated scenarios
"""

from .adaptive_rank import (
    AdaptiveRankSelector,
    RankSelectionStrategy,
    estimate_optimal_rank,
)
from .config import (
    LoRABestPracticesConfig,
    LoRAScalingMethod,
    LoRAVariant,
    TargetModulePreset,
)
from .merging import (
    LoRAMerger,
    MergeMethod,
    merge_lora_adapters,
)
from .optimizer import (
    LoRAPlusOptimizer,
    create_lora_plus_optimizer,
    get_lora_param_groups,
)
from .presets import (
    PRESET_CONFIGS,
    PresetType,
    get_preset_config,
)
from .privacy import (
    FFALoRAConfig,
    PrivacyPreservingLoRA,
)
from .scaling import (
    RSLoRAScaling,
    StandardLoRAScaling,
    compute_lora_scaling,
)

__all__ = [
    # Config
    "LoRABestPracticesConfig",
    "LoRAScalingMethod",
    "LoRAVariant",
    "TargetModulePreset",
    # Presets
    "get_preset_config",
    "PresetType",
    "PRESET_CONFIGS",
    # Optimizer
    "LoRAPlusOptimizer",
    "create_lora_plus_optimizer",
    "get_lora_param_groups",
    # Adaptive Rank
    "AdaptiveRankSelector",
    "estimate_optimal_rank",
    "RankSelectionStrategy",
    # Merging
    "LoRAMerger",
    "MergeMethod",
    "merge_lora_adapters",
    # Privacy
    "FFALoRAConfig",
    "PrivacyPreservingLoRA",
    # Scaling
    "compute_lora_scaling",
    "RSLoRAScaling",
    "StandardLoRAScaling",
]

__version__ = "1.0.0"
