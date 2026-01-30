"""
Canonical serialization for deterministic hashing.

This module provides functions to serialize data structures into
canonical byte representations suitable for cryptographic signing
and hash verification.

The canonical format ensures:
1. Deterministic ordering of dictionary keys
2. Consistent numeric representations
3. No whitespace variations
"""

import json
from typing import Any, Dict, List, Union

try:
    import msgpack
    _MSGPACK_AVAILABLE = True
except ImportError:
    _MSGPACK_AVAILABLE = False


def _sort_dict_recursive(obj: Any) -> Any:
    """Recursively sort dictionary keys for deterministic ordering."""
    if isinstance(obj, dict):
        return {k: _sort_dict_recursive(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [_sort_dict_recursive(item) for item in obj]
    else:
        return obj


def canonical_json(data: Union[Dict, List, Any]) -> str:
    """
    Convert data to canonical JSON string.

    Args:
        data: Data to serialize (dict, list, or primitive)

    Returns:
        Canonical JSON string with sorted keys and no extra whitespace
    """
    sorted_data = _sort_dict_recursive(data)
    return json.dumps(
        sorted_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_bytes(data: Union[Dict, List, Any]) -> bytes:
    """
    Convert data to canonical byte representation.

    Uses msgpack if available for compact binary format,
    otherwise falls back to canonical JSON encoded as UTF-8.

    Args:
        data: Data to serialize (dict, list, or primitive)

    Returns:
        Canonical byte representation

    Note:
        The output is deterministic - the same input will always
        produce the same output bytes, suitable for hashing.
    """
    sorted_data = _sort_dict_recursive(data)

    if _MSGPACK_AVAILABLE:
        # Use msgpack with strict settings for determinism
        return msgpack.packb(
            sorted_data,
            use_bin_type=True,
            strict_types=False,
        )
    else:
        # Fallback to canonical JSON
        return canonical_json(sorted_data).encode("utf-8")


def verify_canonical_hash(data: Union[Dict, List, Any], expected_hash: str) -> bool:
    """
    Verify that data matches an expected SHA-256 hash.

    Args:
        data: Data to verify
        expected_hash: Expected SHA-256 hex digest

    Returns:
        True if hash matches, False otherwise
    """
    import hashlib

    actual = hashlib.sha256(canonical_bytes(data)).hexdigest()
    return actual == expected_hash.lower()
