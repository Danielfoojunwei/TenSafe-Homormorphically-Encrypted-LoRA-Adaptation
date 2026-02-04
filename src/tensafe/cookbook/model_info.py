"""
Model metadata and recommended configurations.

Provides model attribute lookups and recommended renderer/tokenizer
selections for different model families.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional


@dataclass
class ModelAttributes:
    """
    Attributes and metadata for a model.

    Stores information about model organization, version, size,
    and capabilities.
    """

    organization: str  # e.g., "meta-llama", "Qwen"
    version: str  # e.g., "3.1", "3"
    size: str  # e.g., "8B", "70B"
    is_instruct: bool = False  # Chat/instruct tuned
    is_vision: bool = False  # Vision-language model
    is_thinking: bool = False  # Has thinking/reasoning mode
    context_length: int = 4096  # Default context length
    vocab_size: int = 32000  # Vocabulary size

    @property
    def full_name(self) -> str:
        """Construct the full model name."""
        suffix = "-Instruct" if self.is_instruct else ""
        return f"{self.organization}/{self.version}-{self.size}{suffix}"


@lru_cache(maxsize=1)
def _get_llama_info() -> Dict[str, ModelAttributes]:
    """Get Llama model information."""
    return {
        "meta-llama/Llama-3.1-8B": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.1",
            size="8B",
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.1-8B-Instruct": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.1",
            size="8B",
            is_instruct=True,
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.1-70B": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.1",
            size="70B",
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.1-70B-Instruct": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.1",
            size="70B",
            is_instruct=True,
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.2-1B": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.2",
            size="1B",
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.2-1B-Instruct": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.2",
            size="1B",
            is_instruct=True,
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.2-3B": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.2",
            size="3B",
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.2-3B-Instruct": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.2",
            size="3B",
            is_instruct=True,
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.2-11B-Vision": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.2",
            size="11B",
            is_vision=True,
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.2-11B-Vision-Instruct": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.2",
            size="11B",
            is_instruct=True,
            is_vision=True,
            context_length=131072,
            vocab_size=128256,
        ),
        "meta-llama/Llama-3.3-70B-Instruct": ModelAttributes(
            organization="meta-llama",
            version="Llama-3.3",
            size="70B",
            is_instruct=True,
            context_length=131072,
            vocab_size=128256,
        ),
    }


@lru_cache(maxsize=1)
def _get_qwen_info() -> Dict[str, ModelAttributes]:
    """Get Qwen model information."""
    return {
        "Qwen/Qwen3-8B": ModelAttributes(
            organization="Qwen",
            version="Qwen3",
            size="8B",
            context_length=32768,
            vocab_size=151936,
        ),
        "Qwen/Qwen3-8B-Instruct": ModelAttributes(
            organization="Qwen",
            version="Qwen3",
            size="8B",
            is_instruct=True,
            context_length=32768,
            vocab_size=151936,
        ),
        "Qwen/Qwen3-14B": ModelAttributes(
            organization="Qwen",
            version="Qwen3",
            size="14B",
            context_length=32768,
            vocab_size=151936,
        ),
        "Qwen/Qwen3-14B-Instruct": ModelAttributes(
            organization="Qwen",
            version="Qwen3",
            size="14B",
            is_instruct=True,
            context_length=32768,
            vocab_size=151936,
        ),
        "Qwen/Qwen3-72B": ModelAttributes(
            organization="Qwen",
            version="Qwen3",
            size="72B",
            context_length=32768,
            vocab_size=151936,
        ),
        "Qwen/Qwen3-72B-Instruct": ModelAttributes(
            organization="Qwen",
            version="Qwen3",
            size="72B",
            is_instruct=True,
            context_length=32768,
            vocab_size=151936,
        ),
    }


@lru_cache(maxsize=1)
def _get_deepseek_info() -> Dict[str, ModelAttributes]:
    """Get DeepSeek model information."""
    return {
        "deepseek-ai/DeepSeek-V3": ModelAttributes(
            organization="deepseek-ai",
            version="DeepSeek-V3",
            size="671B-MoE",
            is_instruct=True,
            is_thinking=True,
            context_length=131072,
            vocab_size=129280,
        ),
        "deepseek-ai/DeepSeek-V3.1": ModelAttributes(
            organization="deepseek-ai",
            version="DeepSeek-V3.1",
            size="671B-MoE",
            is_instruct=True,
            is_thinking=True,
            context_length=131072,
            vocab_size=129280,
        ),
    }


@lru_cache(maxsize=1)
def _get_moonshot_info() -> Dict[str, ModelAttributes]:
    """Get Moonshot/Kimi model information."""
    return {
        "moonshotai/Kimi-K2-Thinking": ModelAttributes(
            organization="moonshotai",
            version="Kimi-K2",
            size="1T-MoE",
            is_instruct=True,
            is_thinking=True,
            context_length=131072,
            vocab_size=131072,
        ),
    }


def get_model_attributes(model_name: str) -> Optional[ModelAttributes]:
    """
    Get attributes for a model by name.

    Args:
        model_name: Model name (e.g., "meta-llama/Llama-3.1-8B")

    Returns:
        ModelAttributes if found, None otherwise
    """
    # Strip any version/variant suffixes
    base_name = model_name.split(":")[0]

    # Check all model registries
    registries = [
        _get_llama_info(),
        _get_qwen_info(),
        _get_deepseek_info(),
        _get_moonshot_info(),
    ]

    for registry in registries:
        if base_name in registry:
            return registry[base_name]

    return None


def get_recommended_renderer_name(model_name: str) -> str:
    """
    Get the recommended renderer for a model.

    Args:
        model_name: Model name

    Returns:
        Renderer name (e.g., "llama3", "qwen3")
    """
    names = get_recommended_renderer_names(model_name)
    return names[0] if names else "role-colon"


def get_recommended_renderer_names(model_name: str) -> List[str]:
    """
    Get recommended renderer names for a model.

    Returns a list of compatible renderers in order of preference.

    Args:
        model_name: Model name

    Returns:
        List of renderer names
    """
    name_lower = model_name.lower()

    # Match by model family
    if "llama-3" in name_lower or "llama3" in name_lower:
        attrs = get_model_attributes(model_name)
        if attrs and attrs.is_vision:
            return ["llama3-vision", "llama3"]
        return ["llama3", "llama3-instruct"]

    if "qwen" in name_lower:
        return ["qwen3", "qwen"]

    if "deepseek" in name_lower:
        return ["deepseek-v3", "deepseek"]

    if "kimi" in name_lower:
        return ["kimi-k2", "kimi"]

    if "mistral" in name_lower:
        # Mistral uses similar format to Llama
        return ["llama3", "role-colon"]

    if "gpt" in name_lower:
        return ["role-colon", "generic"]

    # Default fallback
    return ["role-colon", "generic"]


def is_chat_model(model_name: str) -> bool:
    """
    Check if a model is a chat/instruct model.

    Args:
        model_name: Model name

    Returns:
        True if chat/instruct model
    """
    attrs = get_model_attributes(model_name)
    if attrs:
        return attrs.is_instruct

    # Check naming patterns
    name_lower = model_name.lower()
    return any(
        pattern in name_lower
        for pattern in ["instruct", "chat", "-it", "-hf"]
    )


def is_vision_model(model_name: str) -> bool:
    """
    Check if a model is a vision-language model.

    Args:
        model_name: Model name

    Returns:
        True if vision-language model
    """
    attrs = get_model_attributes(model_name)
    if attrs:
        return attrs.is_vision

    name_lower = model_name.lower()
    return "vision" in name_lower or "vlm" in name_lower


def is_thinking_model(model_name: str) -> bool:
    """
    Check if a model has thinking/reasoning capabilities.

    Args:
        model_name: Model name

    Returns:
        True if thinking model
    """
    attrs = get_model_attributes(model_name)
    if attrs:
        return attrs.is_thinking

    name_lower = model_name.lower()
    return "thinking" in name_lower or "reasoning" in name_lower


def get_context_length(model_name: str) -> int:
    """
    Get the context length for a model.

    Args:
        model_name: Model name

    Returns:
        Context length in tokens
    """
    attrs = get_model_attributes(model_name)
    if attrs:
        return attrs.context_length

    # Default context lengths by model family
    if "llama-3" in model_name.lower():
        return 131072
    if "qwen" in model_name.lower():
        return 32768
    if "deepseek" in model_name.lower():
        return 131072

    return 4096  # Conservative default
