"""
LoRA Best Practices Configuration

Implements research-backed defaults and configurations for optimal LoRA fine-tuning.

Key research findings implemented:
1. Target ALL layers (MLP + Attention) - "LoRA Without Regret"
2. Alpha = 2 * rank - optimal scaling heuristic
3. Learning rate 10x higher than full fine-tuning
4. rsLoRA scaling (α/√r) for high-rank stability
5. DoRA for weight-decomposed adaptation
6. LoRA+ for asymmetric learning rates
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import math


class LoRAScalingMethod(Enum):
    """LoRA scaling methods based on research."""

    STANDARD = "standard"
    """Standard scaling: α/r. Works well for low ranks (r <= 32)."""

    RSLORA = "rslora"
    """Rank-stabilized scaling: α/√r. Better for high ranks (r > 32)."""

    UNIT = "unit"
    """Unit scaling: 1.0. For when you want raw LoRA output."""


class LoRAVariant(Enum):
    """LoRA implementation variants."""

    STANDARD = "standard"
    """Standard LoRA: W' = W + BA with scaling."""

    DORA = "dora"
    """DoRA: Weight-Decomposed LoRA. Decomposes into magnitude and direction."""

    VERA = "vera"
    """VeRA: Vector-based Random Matrix Adaptation. 10x fewer parameters."""

    ADALORA = "adalora"
    """AdaLoRA: Adaptive rank allocation per layer."""


class TargetModulePreset(Enum):
    """Preset target module configurations based on research."""

    ATTENTION_ONLY = "attention_only"
    """Only attention layers (q, k, v, o). NOT RECOMMENDED - underperforms."""

    MLP_ONLY = "mlp_only"
    """Only MLP layers. Surprisingly effective per "LoRA Without Regret"."""

    ALL_LINEAR = "all_linear"
    """All linear layers (attention + MLP). RECOMMENDED for best performance."""

    QKV_ONLY = "qkv_only"
    """Only Q, K, V projections. Original LoRA paper default."""

    CUSTOM = "custom"
    """Custom target modules specified by user."""


# Target module mappings for common architectures
TARGET_MODULE_PRESETS = {
    # LLaMA-style architectures (LLaMA, LLaMA 2, LLaMA 3, Mistral, etc.)
    "llama": {
        TargetModulePreset.ATTENTION_ONLY: [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
        TargetModulePreset.MLP_ONLY: [
            "gate_proj", "up_proj", "down_proj"
        ],
        TargetModulePreset.ALL_LINEAR: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        TargetModulePreset.QKV_ONLY: [
            "q_proj", "k_proj", "v_proj"
        ],
    },
    # GPT-style architectures (GPT-2, GPT-J, GPT-NeoX)
    "gpt": {
        TargetModulePreset.ATTENTION_ONLY: [
            "c_attn", "c_proj"
        ],
        TargetModulePreset.MLP_ONLY: [
            "c_fc", "c_proj"
        ],
        TargetModulePreset.ALL_LINEAR: [
            "c_attn", "c_proj", "c_fc"
        ],
        TargetModulePreset.QKV_ONLY: [
            "c_attn"
        ],
    },
    # Falcon architecture
    "falcon": {
        TargetModulePreset.ATTENTION_ONLY: [
            "query_key_value", "dense"
        ],
        TargetModulePreset.MLP_ONLY: [
            "dense_h_to_4h", "dense_4h_to_h"
        ],
        TargetModulePreset.ALL_LINEAR: [
            "query_key_value", "dense",
            "dense_h_to_4h", "dense_4h_to_h"
        ],
        TargetModulePreset.QKV_ONLY: [
            "query_key_value"
        ],
    },
    # Qwen architecture
    "qwen": {
        TargetModulePreset.ATTENTION_ONLY: [
            "c_attn", "c_proj"
        ],
        TargetModulePreset.MLP_ONLY: [
            "w1", "w2", "c_proj"
        ],
        TargetModulePreset.ALL_LINEAR: [
            "c_attn", "c_proj", "w1", "w2"
        ],
        TargetModulePreset.QKV_ONLY: [
            "c_attn"
        ],
    },
}


@dataclass
class LoRABestPracticesConfig:
    """
    LoRA configuration implementing research best practices.

    This configuration class provides sensible defaults based on extensive
    research findings, particularly from "LoRA Without Regret" (Schulman et al.).

    Key defaults:
    - rank: 32 (good balance for most tasks)
    - alpha: 64 (2 * rank, research-backed optimal)
    - target_modules: ALL_LINEAR (critical for matching full fine-tuning)
    - scaling_method: RSLORA (stable at all ranks)
    - learning_rate: 2e-4 (10x typical full fine-tuning rate)

    Attributes:
        rank: LoRA rank (dimension of low-rank matrices). Higher = more capacity.
        alpha: Scaling factor. Research shows alpha = 2 * rank is optimal.
        dropout: LoRA dropout. Set to 0 for faster training unless overfitting.
        target_preset: Which layers to apply LoRA to. ALL_LINEAR recommended.
        target_modules: Custom target modules (used when preset is CUSTOM).
        scaling_method: How to compute scaling factor (STANDARD or RSLORA).
        variant: LoRA implementation variant (STANDARD, DORA, etc.).
        use_dora: Enable DoRA weight decomposition.
        use_rslora: Enable rank-stabilized scaling.
        bias: Bias handling ('none', 'all', 'lora_only').

    Training parameters (LoRA+ and optimization):
        learning_rate: Base learning rate (10x full fine-tuning optimal).
        lora_plus_ratio: Learning rate ratio for B matrix (λ in LoRA+).
        weight_decay: Weight decay for regularization.
        warmup_ratio: Warmup fraction of total steps.

    Privacy parameters (for HE/federated scenarios):
        freeze_a_matrix: Freeze A matrix for FFA-LoRA (privacy-preserving).
        use_he_compatible_init: Use HE-compatible initialization.
    """

    # Core LoRA parameters
    rank: int = 32
    alpha: Optional[float] = None  # Defaults to 2 * rank
    dropout: float = 0.0  # Research suggests 0 is fine for short training

    # Target module configuration
    target_preset: TargetModulePreset = TargetModulePreset.ALL_LINEAR
    target_modules: Optional[List[str]] = None
    architecture: str = "llama"  # Model architecture for preset lookup

    # Scaling and variants
    scaling_method: LoRAScalingMethod = LoRAScalingMethod.RSLORA
    variant: LoRAVariant = LoRAVariant.STANDARD
    use_dora: bool = False
    use_rslora: bool = True

    # Bias handling
    bias: str = "none"

    # Task type
    task_type: str = "CAUSAL_LM"

    # Training parameters (LoRA+)
    learning_rate: float = 2e-4  # 10x full fine-tuning
    lora_plus_ratio: float = 16.0  # λ in LoRA+: lr_B = λ * lr_A
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Training duration
    num_epochs: int = 1  # Usually 1-3 is sufficient
    max_steps: Optional[int] = None

    # Privacy-preserving options (for HE/federated)
    freeze_a_matrix: bool = False  # FFA-LoRA: freeze A, only train B
    use_he_compatible_init: bool = False

    # Quantization (QLoRA)
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Modules to NOT apply LoRA to
    modules_to_save: Optional[List[str]] = None

    def __post_init__(self):
        """Validate and set defaults based on research best practices."""
        # Default alpha to 2 * rank (research-backed optimal)
        if self.alpha is None:
            self.alpha = 2.0 * self.rank

        # Enable rsLoRA automatically for high ranks
        if self.rank > 32 and not self.use_rslora:
            self.use_rslora = True
            self.scaling_method = LoRAScalingMethod.RSLORA

        # Set target modules from preset if not custom
        if self.target_modules is None:
            self.target_modules = self._get_target_modules_from_preset()

        # Enable DoRA if variant is DORA
        if self.variant == LoRAVariant.DORA:
            self.use_dora = True

        # Validate
        self._validate()

    def _get_target_modules_from_preset(self) -> List[str]:
        """Get target modules based on preset and architecture."""
        if self.target_preset == TargetModulePreset.CUSTOM:
            return []

        arch_presets = TARGET_MODULE_PRESETS.get(
            self.architecture.lower(),
            TARGET_MODULE_PRESETS["llama"]  # Default to LLaMA
        )

        return arch_presets.get(
            self.target_preset,
            arch_presets[TargetModulePreset.ALL_LINEAR]
        )

    def _validate(self):
        """Validate configuration parameters."""
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive")
        if self.lora_plus_ratio <= 0:
            raise ValueError(f"lora_plus_ratio must be positive")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")

    @property
    def scaling_factor(self) -> float:
        """Compute the scaling factor based on method."""
        if self.scaling_method == LoRAScalingMethod.RSLORA or self.use_rslora:
            # rsLoRA: α / √r
            return self.alpha / math.sqrt(self.rank)
        elif self.scaling_method == LoRAScalingMethod.UNIT:
            return 1.0
        else:
            # Standard: α / r
            return self.alpha / self.rank

    @property
    def effective_rank(self) -> int:
        """Get effective rank considering any adaptations."""
        return self.rank

    def get_learning_rates(self) -> Dict[str, float]:
        """Get learning rates for LoRA+ optimization."""
        return {
            "lora_A": self.learning_rate,
            "lora_B": self.learning_rate * self.lora_plus_ratio,
            "default": self.learning_rate,
        }

    def to_peft_config(self) -> Dict[str, Any]:
        """Convert to Hugging Face PEFT LoraConfig parameters."""
        config = {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "use_rslora": self.use_rslora,
            "use_dora": self.use_dora,
        }

        if self.modules_to_save:
            config["modules_to_save"] = self.modules_to_save

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_preset": self.target_preset.value,
            "target_modules": self.target_modules,
            "architecture": self.architecture,
            "scaling_method": self.scaling_method.value,
            "variant": self.variant.value,
            "use_dora": self.use_dora,
            "use_rslora": self.use_rslora,
            "bias": self.bias,
            "task_type": self.task_type,
            "learning_rate": self.learning_rate,
            "lora_plus_ratio": self.lora_plus_ratio,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "freeze_a_matrix": self.freeze_a_matrix,
            "use_he_compatible_init": self.use_he_compatible_init,
            "use_4bit": self.use_4bit,
            "use_8bit": self.use_8bit,
            "scaling_factor": self.scaling_factor,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoRABestPracticesConfig":
        """Deserialize from dictionary."""
        # Handle enums
        if "target_preset" in d and isinstance(d["target_preset"], str):
            d["target_preset"] = TargetModulePreset(d["target_preset"])
        if "scaling_method" in d and isinstance(d["scaling_method"], str):
            d["scaling_method"] = LoRAScalingMethod(d["scaling_method"])
        if "variant" in d and isinstance(d["variant"], str):
            d["variant"] = LoRAVariant(d["variant"])

        # Remove computed properties
        d.pop("scaling_factor", None)

        return cls(**d)

    def estimate_trainable_params(self, hidden_size: int, num_layers: int) -> int:
        """Estimate number of trainable parameters."""
        # Each LoRA adapter has: A (r x in_features) + B (out_features x r)
        # For attention: Q, K, V, O each have hidden_size x hidden_size
        # For MLP: gate, up have hidden_size x intermediate, down has intermediate x hidden_size

        params_per_adapter = 2 * self.rank * hidden_size

        # Count adapters based on target modules
        num_adapters = len(self.target_modules) * num_layers

        return params_per_adapter * num_adapters

    def __repr__(self) -> str:
        return (
            f"LoRABestPracticesConfig("
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling_factor:.3f}, "
            f"targets={len(self.target_modules)} modules, "
            f"variant={self.variant.value})"
        )


def create_optimal_config(
    dataset_size: int,
    task_complexity: str = "medium",
    memory_constraint_gb: Optional[float] = None,
    architecture: str = "llama",
) -> LoRABestPracticesConfig:
    """
    Create an optimal LoRA configuration based on dataset and constraints.

    Args:
        dataset_size: Number of training examples
        task_complexity: "simple", "medium", or "complex"
        memory_constraint_gb: Available GPU memory in GB (None for no constraint)
        architecture: Model architecture ("llama", "gpt", "falcon", "qwen")

    Returns:
        Optimized LoRABestPracticesConfig
    """
    # Rank selection based on dataset size (from "LoRA Without Regret")
    # rank-32 handles ~50k examples, scale proportionally
    if dataset_size < 10000:
        base_rank = 16
    elif dataset_size < 50000:
        base_rank = 32
    elif dataset_size < 200000:
        base_rank = 64
    else:
        base_rank = 128

    # Adjust for task complexity
    complexity_multiplier = {
        "simple": 0.5,
        "medium": 1.0,
        "complex": 2.0,
    }.get(task_complexity, 1.0)

    rank = int(base_rank * complexity_multiplier)
    rank = max(8, min(rank, 512))  # Clamp to reasonable range

    # Memory constraint handling
    use_4bit = False
    if memory_constraint_gb is not None:
        if memory_constraint_gb < 16:
            use_4bit = True
            rank = min(rank, 64)  # Limit rank with QLoRA
        elif memory_constraint_gb < 24:
            rank = min(rank, 128)

    # Learning rate adjustment for rank
    # Higher ranks may benefit from slightly lower LR
    if rank > 64:
        learning_rate = 1e-4
    else:
        learning_rate = 2e-4

    return LoRABestPracticesConfig(
        rank=rank,
        architecture=architecture,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_4bit=use_4bit,
        learning_rate=learning_rate,
        use_rslora=rank > 32,
    )
