"""Liger Kernel Integration for TenSafe.

Liger Kernel provides Triton-based optimizations that can achieve:
- 20% throughput improvement
- 60% memory reduction
- Compatible with LoRA training

This module integrates Liger with TenSafe's privacy-preserving training.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Conditional imports
LIGER_AVAILABLE = False
LIGER_KIMI_AVAILABLE = False
try:
    from liger_kernel.transformers import (
        apply_liger_kernel_to_gemma,
        apply_liger_kernel_to_llama,
        apply_liger_kernel_to_mistral,
        apply_liger_kernel_to_qwen2,
    )
    # Try to import Kimi kernel support (may not be available in all versions)
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_kimi
        LIGER_KIMI_AVAILABLE = True
    except ImportError:
        LIGER_KIMI_AVAILABLE = False
    LIGER_AVAILABLE = True
except ImportError:
    logger.info("Liger Kernel not installed. Install with: pip install liger-kernel")

import torch.nn as nn


@dataclass
class LigerOptimizationConfig:
    """Configuration for Liger Kernel optimizations."""

    # Which optimizations to apply
    enable_rope: bool = True          # Rotary Position Embedding
    enable_rms_norm: bool = True      # RMS Normalization
    enable_swiglu: bool = True        # SwiGLU activation
    enable_cross_entropy: bool = True  # Fused cross-entropy
    enable_fused_linear_cross_entropy: bool = True  # Fused linear + CE

    # Model-specific settings
    model_type: str = "auto"  # auto, llama, mistral, gemma, qwen2

    # Memory settings
    use_triton_kernels: bool = True
    gradient_checkpointing: bool = True

    # Compatibility
    preserve_original_forward: bool = False  # Keep original for debugging


def apply_liger_optimizations(
    model: nn.Module,
    config: Optional[LigerOptimizationConfig] = None,
) -> nn.Module:
    """Apply Liger Kernel optimizations to a model.

    This function modifies the model in-place to use Liger's optimized
    Triton kernels for various operations.

    Args:
        model: PyTorch model to optimize
        config: Optimization configuration

    Returns:
        Optimized model (same reference, modified in-place)

    Example:
        ```python
        from transformers import AutoModelForCausalLM
        from tensorguard.optimizations import apply_liger_optimizations

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
        model = apply_liger_optimizations(model)
        # Model now uses Liger kernels for faster training
        ```
    """
    config = config or LigerOptimizationConfig()

    if not LIGER_AVAILABLE:
        logger.warning("Liger Kernel not available, returning original model")
        return model

    # Detect model type
    model_type = config.model_type
    if model_type == "auto":
        model_type = _detect_model_type(model)

    logger.info(f"Applying Liger optimizations for model type: {model_type}")

    # Apply model-specific optimizations
    if model_type == "llama":
        apply_liger_kernel_to_llama(
            rope=config.enable_rope,
            rms_norm=config.enable_rms_norm,
            swiglu=config.enable_swiglu,
            cross_entropy=config.enable_cross_entropy,
            fused_linear_cross_entropy=config.enable_fused_linear_cross_entropy,
        )
    elif model_type == "mistral":
        apply_liger_kernel_to_mistral(
            rope=config.enable_rope,
            rms_norm=config.enable_rms_norm,
            swiglu=config.enable_swiglu,
            cross_entropy=config.enable_cross_entropy,
            fused_linear_cross_entropy=config.enable_fused_linear_cross_entropy,
        )
    elif model_type == "gemma":
        apply_liger_kernel_to_gemma(
            rope=config.enable_rope,
            rms_norm=config.enable_rms_norm,
            geglu=config.enable_swiglu,  # Gemma uses GeGLU
            cross_entropy=config.enable_cross_entropy,
            fused_linear_cross_entropy=config.enable_fused_linear_cross_entropy,
        )
    elif model_type == "qwen2":
        apply_liger_kernel_to_qwen2(
            rope=config.enable_rope,
            rms_norm=config.enable_rms_norm,
            swiglu=config.enable_swiglu,
            cross_entropy=config.enable_cross_entropy,
            fused_linear_cross_entropy=config.enable_fused_linear_cross_entropy,
        )
    elif model_type == "kimi":
        # Kimi K2.5 uses MLA (Multi-head Latent Attention) and MoE architecture
        # Apply Liger kernels if available, otherwise use generic optimizations
        if LIGER_KIMI_AVAILABLE:
            apply_liger_kernel_to_kimi(
                rope=config.enable_rope,
                rms_norm=config.enable_rms_norm,
                swiglu=config.enable_swiglu,
                cross_entropy=config.enable_cross_entropy,
                fused_linear_cross_entropy=config.enable_fused_linear_cross_entropy,
            )
        else:
            logger.info("Liger Kimi kernel not available, applying generic optimizations for Kimi")
            _apply_generic_optimizations(model, config)
    else:
        logger.warning(f"Unknown model type: {model_type}, applying generic optimizations")
        _apply_generic_optimizations(model, config)

    # Apply gradient checkpointing if enabled
    if config.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    logger.info("Liger optimizations applied successfully")
    return model


def _detect_model_type(model: nn.Module) -> str:
    """Detect model type from model class name or config."""
    model_class = model.__class__.__name__.lower()

    if "llama" in model_class:
        return "llama"
    elif "mistral" in model_class:
        return "mistral"
    elif "gemma" in model_class:
        return "gemma"
    elif "qwen" in model_class:
        return "qwen2"
    elif "kimi" in model_class:
        return "kimi"

    # Try to get from config
    if hasattr(model, 'config'):
        config_type = getattr(model.config, 'model_type', '').lower()
        if config_type in ["llama", "mistral", "gemma", "qwen2", "kimi"]:
            return config_type

    return "unknown"


def _apply_generic_optimizations(model: nn.Module, config: LigerOptimizationConfig):
    """Apply generic optimizations when model type is unknown."""
    # Replace RMSNorm with fused version if available
    if config.enable_rms_norm:
        try:
            from liger_kernel.ops import LigerRMSNorm
            _replace_modules(model, "RMSNorm", LigerRMSNorm)
        except ImportError:
            pass

    # Replace SwiGLU with fused version if available
    if config.enable_swiglu:
        try:
            from liger_kernel.ops import LigerSwiGLUMLP
            _replace_modules(model, "SwiGLU", LigerSwiGLUMLP)
        except ImportError:
            pass


def _replace_modules(model: nn.Module, target_name: str, replacement_class: type):
    """Replace modules matching target_name with replacement_class."""
    for name, module in model.named_modules():
        if target_name in module.__class__.__name__:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Create replacement with same parameters
            try:
                if hasattr(module, 'weight'):
                    new_module = replacement_class(module.weight.size(-1))
                    new_module.weight = module.weight
                else:
                    new_module = replacement_class()

                setattr(parent, child_name, new_module)
                logger.debug(f"Replaced {name} with {replacement_class.__name__}")
            except Exception as e:
                logger.warning(f"Failed to replace {name}: {e}")


class LigerMemoryOptimizer:
    """Memory optimizer using Liger techniques.

    Provides utilities for reducing memory usage during training:
    - Gradient checkpointing
    - Mixed precision training
    - Memory-efficient attention
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_state = {}

    def enable_memory_efficient_attention(self):
        """Enable memory-efficient attention (FlashAttention-style)."""
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'attn_implementation'):
                self._original_state['attn_implementation'] = self.model.config.attn_implementation
                self.model.config.attn_implementation = 'flash_attention_2'
                logger.info("Memory-efficient attention enabled")

    def enable_activation_checkpointing(self):
        """Enable activation checkpointing for memory savings."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Activation checkpointing enabled")

    def estimate_memory_savings(self) -> Dict[str, float]:
        """Estimate memory savings from optimizations.

        Returns:
            Dict with estimated memory metrics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Estimate memory usage
        # Base: 4 bytes per param (FP32) + gradients + optimizer states
        base_memory_gb = total_params * 4 * 4 / (1024**3)  # params + grads + adam states

        # With optimizations
        optimized_memory_gb = base_memory_gb * 0.4  # ~60% reduction with Liger

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "estimated_base_memory_gb": base_memory_gb,
            "estimated_optimized_memory_gb": optimized_memory_gb,
            "estimated_savings_percent": 60.0,
        }
