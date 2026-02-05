"""
Hyperparameter utilities for LoRA fine-tuning.

Provides functions for calculating optimal learning rates, counting
LoRA parameters, and converting between full fine-tuning and LoRA
hyperparameters.

These utilities help ensure consistent and effective training
configurations across different model sizes and LoRA ranks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default LoRA target modules for common architectures
DEFAULT_LORA_TARGETS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["c_attn", "c_proj", "w1", "w2"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt": ["c_attn", "c_proj", "c_fc"],
}

# Hidden sizes for common models (for quick lookup without loading config)
HIDDEN_SIZES = {
    "meta-llama/Llama-3.1-8B": 4096,
    "meta-llama/Llama-3.1-70B": 8192,
    "meta-llama/Llama-3.2-1B": 2048,
    "meta-llama/Llama-3.2-3B": 3072,
    "meta-llama/Llama-3.3-70B": 8192,
    "Qwen/Qwen3-8B": 4096,
    "Qwen/Qwen3-14B": 5120,
    "Qwen/Qwen3-72B": 8192,
    "deepseek-ai/DeepSeek-V3": 7168,
}


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    target_modules: List[str] = None
    bias: str = "none"  # "none", "all", "lora_only"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = DEFAULT_LORA_TARGETS["llama"]

    @property
    def scaling(self) -> float:
        """LoRA scaling factor (alpha / rank)."""
        return self.alpha / self.rank

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }


def get_lora_lr_over_full_finetune_lr() -> float:
    """
    Get the scaling factor to convert full fine-tuning learning rate
    to an equivalent LoRA learning rate.

    LoRA typically requires higher learning rates than full fine-tuning
    due to the low-rank decomposition.

    Returns:
        Scaling factor (typically 10.0)
    """
    return 10.0


def get_hidden_size(model_name: str) -> int:
    """
    Get the hidden dimension size for a model.

    Uses hardcoded values for common models, falling back to
    loading the config from HuggingFace if available.

    Args:
        model_name: Model name (e.g., "meta-llama/Llama-3.1-8B")

    Returns:
        Hidden dimension size
    """
    # Check hardcoded values first
    base_name = model_name.split(":")[0]
    if base_name in HIDDEN_SIZES:
        return HIDDEN_SIZES[base_name]

    # Try to load from HuggingFace config
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(base_name, trust_remote_code=True)
        return config.hidden_size
    except Exception as e:
        logger.warning(f"Could not load config for {model_name}: {e}")
        # Default to a reasonable size
        return 4096


def get_lora_param_count(
    model_name: str,
    rank: int = 32,
    target_modules: Optional[List[str]] = None,
    num_layers: Optional[int] = None,
    detailed: bool = False,
) -> int | Dict[str, int]:
    """
    Calculate the total number of parameters in a LoRA adapter.

    Args:
        model_name: Model name for config lookup
        rank: LoRA rank
        target_modules: Which modules to apply LoRA to
        num_layers: Number of transformer layers (if known)
        detailed: If True, return breakdown by module type

    Returns:
        Total parameter count, or dict with breakdown if detailed=True
    """
    hidden_size = get_hidden_size(model_name)

    if target_modules is None:
        target_modules = DEFAULT_LORA_TARGETS.get("llama", [])

    # Estimate layer count from model name if not provided
    if num_layers is None:
        # Extract from common naming patterns
        if "70B" in model_name or "72B" in model_name:
            num_layers = 80
        elif "14B" in model_name:
            num_layers = 40
        elif "8B" in model_name:
            num_layers = 32
        elif "3B" in model_name:
            num_layers = 28
        elif "1B" in model_name:
            num_layers = 16
        else:
            num_layers = 32  # Default

    # Calculate parameters per module
    # LoRA adds two matrices: A (d x r) and B (r x d) for each target
    params_per_target = 2 * hidden_size * rank  # A and B matrices

    if detailed:
        breakdown: Dict[str, int] = {}
        for module in target_modules:
            # Each module appears once per layer
            module_params = params_per_target * num_layers
            breakdown[module] = module_params
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    # Total: params_per_target * num_modules * num_layers
    total = params_per_target * len(target_modules) * num_layers
    return total


def get_lora_lr(
    model_name: str,
    base_lr: float = 2e-4,
    rank: int = 32,
    adjust_for_rank: bool = True,
) -> float:
    """
    Calculate optimal LoRA learning rate for a model.

    Uses model-specific adjustments and optionally scales by rank.

    Args:
        model_name: Model name
        base_lr: Base learning rate for LoRA
        rank: LoRA rank (for rank-based adjustment)
        adjust_for_rank: Whether to adjust LR based on rank

    Returns:
        Recommended learning rate
    """
    lr = base_lr

    # Model-specific adjustments
    if "70B" in model_name or "72B" in model_name:
        lr *= 0.5  # Larger models need smaller LR
    elif "1B" in model_name:
        lr *= 2.0  # Smaller models can use larger LR

    # Rank adjustment (higher rank -> lower LR)
    if adjust_for_rank:
        # sqrt scaling: LR ~ 1/sqrt(rank)
        rank_factor = math.sqrt(32.0 / rank)
        lr *= rank_factor

    return lr


def get_lora_lr_multiplier(model_name: str) -> float:
    """
    Get a model-specific learning rate multiplier.

    Provides a comparative multiplier for estimating optimal
    learning rates across different model families.

    Args:
        model_name: Model name

    Returns:
        Learning rate multiplier
    """
    # Normalize model name
    name_lower = model_name.lower()

    if "deepseek" in name_lower:
        return 0.8  # DeepSeek models tend to need lower LR
    elif "qwen" in name_lower:
        return 1.0
    elif "llama" in name_lower:
        return 1.0
    elif "mistral" in name_lower:
        return 1.2  # Mistral can use slightly higher LR
    else:
        return 1.0


@lru_cache(maxsize=16)
def get_full_finetune_param_count(model_name: str) -> int:
    """
    Get the total parameter count for full fine-tuning.

    Cached for efficiency.

    Args:
        model_name: Model name

    Returns:
        Total parameter count
    """
    # Common model sizes
    param_counts = {
        "1B": 1_000_000_000,
        "3B": 3_000_000_000,
        "7B": 7_000_000_000,
        "8B": 8_000_000_000,
        "13B": 13_000_000_000,
        "14B": 14_000_000_000,
        "70B": 70_000_000_000,
        "72B": 72_000_000_000,
    }

    for size_str, count in param_counts.items():
        if size_str in model_name:
            return count

    # Try to load from HuggingFace
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Estimate from config
        hidden = config.hidden_size
        layers = getattr(config, "num_hidden_layers", 32)
        vocab = getattr(config, "vocab_size", 32000)
        intermediate = getattr(config, "intermediate_size", hidden * 4)

        # Rough estimate: embedding + attention + FFN per layer
        embedding_params = vocab * hidden
        attention_params = 4 * hidden * hidden  # Q, K, V, O
        ffn_params = 2 * hidden * intermediate  # up + down

        total = embedding_params + layers * (attention_params + ffn_params)
        return total
    except Exception:
        return 8_000_000_000  # Default to 8B


def get_full_finetune_lr_multiplier(model_name: str) -> float:
    """
    Get inverse square-root scaling factor based on parameter count.

    Larger models typically need proportionally smaller learning rates.

    Args:
        model_name: Model name

    Returns:
        Learning rate multiplier
    """
    param_count = get_full_finetune_param_count(model_name)

    # Inverse square-root scaling relative to 8B baseline
    baseline = 8_000_000_000
    return math.sqrt(baseline / param_count)


def calculate_warmup_steps(
    total_steps: int,
    warmup_ratio: float = 0.03,
    min_warmup: int = 100,
    max_warmup: int = 1000,
) -> int:
    """
    Calculate number of warmup steps.

    Args:
        total_steps: Total training steps
        warmup_ratio: Fraction of steps for warmup
        min_warmup: Minimum warmup steps
        max_warmup: Maximum warmup steps

    Returns:
        Number of warmup steps
    """
    warmup = int(total_steps * warmup_ratio)
    return max(min_warmup, min(warmup, max_warmup))


def get_recommended_batch_size(
    model_name: str,
    context_length: int = 4096,
    gradient_accumulation: int = 1,
) -> Tuple[int, int]:
    """
    Get recommended batch size and gradient accumulation steps.

    Args:
        model_name: Model name
        context_length: Maximum context length
        gradient_accumulation: Desired gradient accumulation

    Returns:
        Tuple of (micro_batch_size, gradient_accumulation_steps)
    """
    # Base recommendations by model size
    if "70B" in model_name or "72B" in model_name:
        micro_batch = 1
        accum = max(32, gradient_accumulation)
    elif "14B" in model_name:
        micro_batch = 2
        accum = max(16, gradient_accumulation)
    elif "8B" in model_name or "7B" in model_name:
        micro_batch = 4
        accum = max(8, gradient_accumulation)
    else:
        micro_batch = 8
        accum = max(4, gradient_accumulation)

    # Adjust for context length
    if context_length > 8192:
        micro_batch = max(1, micro_batch // 2)
        accum *= 2

    return micro_batch, accum
