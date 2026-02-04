"""
LoRA Configuration Presets

Pre-configured settings for common use cases based on research best practices.
Each preset is optimized for specific scenarios:
- Model size
- Task type
- Memory constraints
- Training data size
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

from .config import (
    LoRABestPracticesConfig,
    LoRAScalingMethod,
    LoRAVariant,
    TargetModulePreset,
)


class PresetType(Enum):
    """Available configuration presets."""

    # By task type
    INSTRUCTION_TUNING = "instruction_tuning"
    """For instruction-following fine-tuning (Alpaca, OASST, etc.)"""

    CHAT = "chat"
    """For conversational fine-tuning"""

    CODE = "code"
    """For code generation/completion tasks"""

    MATH = "math"
    """For mathematical reasoning tasks"""

    CLASSIFICATION = "classification"
    """For text classification tasks"""

    # By memory constraint
    MEMORY_EFFICIENT = "memory_efficient"
    """For limited VRAM (< 16GB) - uses QLoRA"""

    STANDARD = "standard"
    """Balanced configuration for 24GB+ VRAM"""

    HIGH_CAPACITY = "high_capacity"
    """Maximum capacity for 48GB+ VRAM"""

    # By optimization goal
    FAST_TRAINING = "fast_training"
    """Optimized for training speed"""

    HIGH_QUALITY = "high_quality"
    """Optimized for output quality"""

    # Privacy-preserving
    PRIVACY_PRESERVING = "privacy_preserving"
    """For federated/HE scenarios with FFA-LoRA"""

    # Model-specific
    LLAMA_7B = "llama_7b"
    LLAMA_13B = "llama_13b"
    LLAMA_70B = "llama_70b"
    MISTRAL_7B = "mistral_7b"
    QWEN_7B = "qwen_7b"


@dataclass
class PresetDescription:
    """Detailed description of a preset."""
    name: str
    description: str
    best_for: str
    limitations: str
    recommended_dataset_size: str
    estimated_vram_gb: float


# Preset configurations
PRESET_CONFIGS: Dict[PresetType, LoRABestPracticesConfig] = {
    # =========================================================================
    # TASK-BASED PRESETS
    # =========================================================================

    PresetType.INSTRUCTION_TUNING: LoRABestPracticesConfig(
        rank=32,
        alpha=64,  # 2 * rank
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=2e-4,
        lora_plus_ratio=16.0,
        num_epochs=1,
        warmup_ratio=0.03,
    ),

    PresetType.CHAT: LoRABestPracticesConfig(
        rank=64,
        alpha=128,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=1e-4,
        lora_plus_ratio=16.0,
        num_epochs=2,
        warmup_ratio=0.05,
    ),

    PresetType.CODE: LoRABestPracticesConfig(
        rank=64,
        alpha=128,
        dropout=0.05,  # Slight regularization for code
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=1e-4,
        lora_plus_ratio=8.0,
        num_epochs=3,
        warmup_ratio=0.1,
    ),

    PresetType.MATH: LoRABestPracticesConfig(
        rank=128,  # Higher rank for complex reasoning
        alpha=256,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=5e-5,
        lora_plus_ratio=16.0,
        num_epochs=3,
        warmup_ratio=0.1,
    ),

    PresetType.CLASSIFICATION: LoRABestPracticesConfig(
        rank=16,  # Lower rank sufficient for classification
        alpha=32,
        dropout=0.1,
        target_preset=TargetModulePreset.ATTENTION_ONLY,
        use_rslora=False,
        learning_rate=2e-4,
        lora_plus_ratio=8.0,
        num_epochs=3,
        task_type="SEQ_CLS",
    ),

    # =========================================================================
    # MEMORY-BASED PRESETS
    # =========================================================================

    PresetType.MEMORY_EFFICIENT: LoRABestPracticesConfig(
        rank=16,
        alpha=32,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=False,
        use_4bit=True,  # QLoRA
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        learning_rate=2e-4,
        lora_plus_ratio=16.0,
    ),

    PresetType.STANDARD: LoRABestPracticesConfig(
        rank=32,
        alpha=64,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=2e-4,
        lora_plus_ratio=16.0,
    ),

    PresetType.HIGH_CAPACITY: LoRABestPracticesConfig(
        rank=128,
        alpha=256,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        use_dora=True,  # DoRA for better capacity
        learning_rate=1e-4,
        lora_plus_ratio=16.0,
    ),

    # =========================================================================
    # OPTIMIZATION GOAL PRESETS
    # =========================================================================

    PresetType.FAST_TRAINING: LoRABestPracticesConfig(
        rank=16,
        alpha=32,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=False,
        learning_rate=5e-4,  # Higher LR for faster convergence
        lora_plus_ratio=8.0,
        num_epochs=1,
        warmup_ratio=0.01,
    ),

    PresetType.HIGH_QUALITY: LoRABestPracticesConfig(
        rank=128,
        alpha=256,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        use_dora=True,
        learning_rate=5e-5,
        lora_plus_ratio=16.0,
        num_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
    ),

    # =========================================================================
    # PRIVACY-PRESERVING PRESET
    # =========================================================================

    PresetType.PRIVACY_PRESERVING: LoRABestPracticesConfig(
        rank=32,
        alpha=64,
        dropout=0.0,
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        freeze_a_matrix=True,  # FFA-LoRA
        use_he_compatible_init=True,
        learning_rate=2e-4,
        lora_plus_ratio=1.0,  # No LoRA+ when A is frozen
    ),

    # =========================================================================
    # MODEL-SPECIFIC PRESETS
    # =========================================================================

    PresetType.LLAMA_7B: LoRABestPracticesConfig(
        rank=32,
        alpha=64,
        dropout=0.0,
        architecture="llama",
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=2e-4,
        lora_plus_ratio=16.0,
    ),

    PresetType.LLAMA_13B: LoRABestPracticesConfig(
        rank=32,
        alpha=64,
        dropout=0.0,
        architecture="llama",
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=1e-4,  # Slightly lower for larger model
        lora_plus_ratio=16.0,
    ),

    PresetType.LLAMA_70B: LoRABestPracticesConfig(
        rank=16,  # Lower rank due to memory
        alpha=32,
        dropout=0.0,
        architecture="llama",
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        use_4bit=True,  # Required for 70B
        learning_rate=5e-5,
        lora_plus_ratio=16.0,
    ),

    PresetType.MISTRAL_7B: LoRABestPracticesConfig(
        rank=32,
        alpha=64,
        dropout=0.0,
        architecture="llama",  # Mistral uses LLaMA architecture
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=2e-4,
        lora_plus_ratio=16.0,
    ),

    PresetType.QWEN_7B: LoRABestPracticesConfig(
        rank=32,
        alpha=64,
        dropout=0.0,
        architecture="qwen",
        target_preset=TargetModulePreset.ALL_LINEAR,
        use_rslora=True,
        learning_rate=2e-4,
        lora_plus_ratio=16.0,
    ),
}

# Preset descriptions
PRESET_DESCRIPTIONS: Dict[PresetType, PresetDescription] = {
    PresetType.INSTRUCTION_TUNING: PresetDescription(
        name="Instruction Tuning",
        description="Optimized for instruction-following tasks like Alpaca or OASST",
        best_for="General instruction following, chat-like interactions",
        limitations="May need more epochs for complex reasoning",
        recommended_dataset_size="10K - 100K examples",
        estimated_vram_gb=24.0,
    ),
    PresetType.MEMORY_EFFICIENT: PresetDescription(
        name="Memory Efficient (QLoRA)",
        description="4-bit quantized base model with LoRA adapters",
        best_for="Limited VRAM environments, large models",
        limitations="Slightly lower quality than full precision",
        recommended_dataset_size="Any",
        estimated_vram_gb=12.0,
    ),
    PresetType.HIGH_QUALITY: PresetDescription(
        name="High Quality",
        description="Maximum capacity configuration with DoRA",
        best_for="When quality is more important than training time",
        limitations="Requires more VRAM and training time",
        recommended_dataset_size="50K+ examples",
        estimated_vram_gb=48.0,
    ),
    PresetType.PRIVACY_PRESERVING: PresetDescription(
        name="Privacy Preserving (FFA-LoRA)",
        description="FFA-LoRA configuration for federated/HE scenarios",
        best_for="Privacy-sensitive applications, federated learning",
        limitations="Cannot use LoRA+ optimization",
        recommended_dataset_size="Any",
        estimated_vram_gb=24.0,
    ),
}


def get_preset_config(
    preset: PresetType,
    **overrides,
) -> LoRABestPracticesConfig:
    """
    Get a preset configuration with optional overrides.

    Args:
        preset: The preset type to use
        **overrides: Any config parameters to override

    Returns:
        LoRABestPracticesConfig with preset values and overrides applied

    Example:
        >>> config = get_preset_config(
        ...     PresetType.INSTRUCTION_TUNING,
        ...     rank=64,  # Override rank
        ...     learning_rate=1e-4,  # Override LR
        ... )
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}")

    # Get base config
    base_config = PRESET_CONFIGS[preset]

    # Convert to dict and apply overrides
    config_dict = base_config.to_dict()
    config_dict.update(overrides)

    # Remove computed properties before reconstruction
    config_dict.pop("scaling_factor", None)

    return LoRABestPracticesConfig.from_dict(config_dict)


def list_presets() -> Dict[str, str]:
    """List all available presets with descriptions."""
    return {
        preset.value: PRESET_DESCRIPTIONS.get(
            preset,
            PresetDescription(
                name=preset.value,
                description="No description available",
                best_for="",
                limitations="",
                recommended_dataset_size="",
                estimated_vram_gb=0.0,
            )
        ).description
        for preset in PresetType
    }


def recommend_preset(
    task_type: str = "general",
    vram_gb: float = 24.0,
    dataset_size: int = 50000,
    quality_priority: bool = False,
    speed_priority: bool = False,
    privacy_required: bool = False,
) -> PresetType:
    """
    Recommend a preset based on requirements.

    Args:
        task_type: Type of task ("instruction", "chat", "code", "math", "classification")
        vram_gb: Available GPU VRAM in GB
        dataset_size: Number of training examples
        quality_priority: Prioritize quality over speed
        speed_priority: Prioritize speed over quality
        privacy_required: Privacy-preserving training required

    Returns:
        Recommended PresetType
    """
    # Privacy takes precedence
    if privacy_required:
        return PresetType.PRIVACY_PRESERVING

    # Memory constraints
    if vram_gb < 16:
        return PresetType.MEMORY_EFFICIENT

    # Task-specific recommendations
    task_presets = {
        "instruction": PresetType.INSTRUCTION_TUNING,
        "chat": PresetType.CHAT,
        "code": PresetType.CODE,
        "math": PresetType.MATH,
        "classification": PresetType.CLASSIFICATION,
    }

    if task_type.lower() in task_presets:
        return task_presets[task_type.lower()]

    # Optimization goal
    if speed_priority:
        return PresetType.FAST_TRAINING
    if quality_priority:
        return PresetType.HIGH_QUALITY

    # Default
    return PresetType.STANDARD
