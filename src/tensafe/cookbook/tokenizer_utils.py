"""
Tokenizer utilities with model-specific handling.

Provides cached tokenizer loading with special handling for
different model families to avoid slow imports at module initialization.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    Tokenizer: TypeAlias = Any

# Special tokenizer configurations for specific models
TOKENIZER_CONFIGS = {
    "moonshotai/Kimi-K2-Thinking": {
        "trust_remote_code": True,
        "revision": "612681931a8c906ddb349f8ad0f582cb552189cd",
    },
    "deepseek-ai/DeepSeek-V3": {
        "trust_remote_code": True,
    },
}

# Tokenizer redirects (use a different tokenizer than the model)
TOKENIZER_REDIRECTS = {
    # Llama 3 models can use the shared instruct tokenizer
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B": "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
}


@lru_cache(maxsize=16)
def get_tokenizer(model_name: str, **kwargs) -> Tokenizer:
    """
    Get a tokenizer for the specified model.

    Uses caching to avoid repeated loading. Handles model-specific
    configurations and redirects automatically.

    Args:
        model_name: Model name (e.g., "meta-llama/Llama-3.1-8B")
        **kwargs: Additional arguments passed to AutoTokenizer

    Returns:
        Loaded tokenizer

    Example:
        >>> tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
        >>> tokens = tokenizer.encode("Hello, world!")
    """
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    # Strip any version/variant suffixes
    base_name = model_name.split(":")[0]

    # Check for redirects
    if base_name in TOKENIZER_REDIRECTS:
        base_name = TOKENIZER_REDIRECTS[base_name]

    # Build kwargs
    load_kwargs: dict[str, Any] = {"use_fast": True}

    # Apply model-specific configs
    if base_name in TOKENIZER_CONFIGS:
        load_kwargs.update(TOKENIZER_CONFIGS[base_name])

    # Merge with user-provided kwargs
    load_kwargs.update(kwargs)

    return AutoTokenizer.from_pretrained(base_name, **load_kwargs)


def get_tokenizer_for_model(model_name: str) -> Tokenizer:
    """
    Alias for get_tokenizer with model name handling.

    Args:
        model_name: Model name

    Returns:
        Loaded tokenizer
    """
    return get_tokenizer(model_name)


def count_tokens(text: str, tokenizer: Tokenizer) -> int:
    """
    Count the number of tokens in a text.

    Args:
        text: Text to tokenize
        tokenizer: Tokenizer to use

    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    tokenizer: Tokenizer,
    truncate_side: str = "right",
) -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        tokenizer: Tokenizer to use
        truncate_side: "left" or "right" side to truncate

    Returns:
        Truncated text
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return text

    if truncate_side == "left":
        tokens = tokens[-max_tokens:]
    else:
        tokens = tokens[:max_tokens]

    return tokenizer.decode(tokens)


def get_special_tokens(tokenizer: Tokenizer) -> dict[str, int | None]:
    """
    Get special token IDs for a tokenizer.

    Args:
        tokenizer: Tokenizer to inspect

    Returns:
        Dictionary mapping token names to IDs
    """
    return {
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "sep_token_id": getattr(tokenizer, "sep_token_id", None),
        "cls_token_id": getattr(tokenizer, "cls_token_id", None),
        "mask_token_id": getattr(tokenizer, "mask_token_id", None),
    }


def pad_sequences(
    sequences: list[list[int]],
    pad_id: int,
    max_length: int | None = None,
    pad_side: str = "right",
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Pad a batch of token sequences to the same length.

    Args:
        sequences: List of token ID lists
        pad_id: Token ID to use for padding
        max_length: Maximum length (None = use longest sequence)
        pad_side: "left" or "right" side to pad

    Returns:
        Tuple of (padded_sequences, attention_masks)
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    masks = []

    for seq in sequences:
        seq_len = len(seq)
        if seq_len >= max_length:
            padded.append(seq[:max_length])
            masks.append([1] * max_length)
        else:
            pad_len = max_length - seq_len
            if pad_side == "left":
                padded.append([pad_id] * pad_len + seq)
                masks.append([0] * pad_len + [1] * seq_len)
            else:
                padded.append(seq + [pad_id] * pad_len)
                masks.append([1] * seq_len + [0] * pad_len)

    return padded, masks


def batch_encode(
    texts: list[str],
    tokenizer: Tokenizer,
    max_length: int | None = None,
    padding: bool = True,
    truncation: bool = True,
) -> dict[str, list[list[int]]]:
    """
    Batch encode multiple texts.

    Args:
        texts: List of texts to encode
        tokenizer: Tokenizer to use
        max_length: Maximum length per sequence
        padding: Whether to pad sequences
        truncation: Whether to truncate sequences

    Returns:
        Dictionary with input_ids and attention_mask
    """
    # Encode all texts
    encoded = [
        tokenizer.encode(text, add_special_tokens=True)
        for text in texts
    ]

    # Truncate if needed
    if truncation and max_length:
        encoded = [seq[:max_length] for seq in encoded]

    # Pad if needed
    if padding:
        pad_id = tokenizer.pad_token_id or 0
        input_ids, attention_mask = pad_sequences(
            encoded, pad_id, max_length
        )
    else:
        input_ids = encoded
        attention_mask = [[1] * len(seq) for seq in encoded]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
