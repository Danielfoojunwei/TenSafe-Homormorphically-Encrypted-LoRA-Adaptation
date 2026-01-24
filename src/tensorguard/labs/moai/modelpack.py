"""
MOAI Model Packaging
Defines the schema for exporting FHE-servable submodules.

SECURITY NOTE: This module uses safe serialization (msgpack) instead of pickle.
Pickle is not safe for untrusted data as it can execute arbitrary code.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import hashlib
import warnings

from .moai_config import MoaiConfig
from ..utils.serialization import safe_dumps, safe_loads, safe_dump, safe_load


@dataclass
class ModelPackMetadata:
    """Metadata for an exported MOAI model package."""
    model_id: str
    version: str
    base_model: str  # e.g. "pi0", "openvla"
    target_modules: List[str]  # e.g. ["policy_head", "visual_router"]
    created_at: str
    git_commit_hash: str
    config: Dict[str, Any]


@dataclass
class ModelPack:
    """
    Container for a FHE-optimized model export.
    Contains weights, config, and verification hashes.

    SECURITY: Uses msgpack for serialization instead of pickle.
    """
    meta: ModelPackMetadata
    weights: Dict[str, bytes]  # Serialized weights
    tokenizer_config: Optional[Dict[str, Any]] = None

    def calculate_hash(self) -> str:
        """Calculate integrity hash of the package."""
        data = {
            "meta": asdict(self.meta),
            "weights_keys": sorted(self.weights.keys()),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

    def serialize(self) -> bytes:
        """
        Serialize for storage/transfer using safe serialization.

        Uses msgpack instead of pickle to prevent arbitrary code execution.
        """
        return safe_dumps({
            "meta": asdict(self.meta),
            "weights": self.weights,
            "tokenizer_config": self.tokenizer_config
        })

    @classmethod
    def deserialize(cls, data: bytes) -> 'ModelPack':
        """
        Deserialize from bytes using safe deserialization.

        Safe for untrusted data - cannot execute arbitrary code.
        """
        obj = safe_loads(data)
        meta = ModelPackMetadata(**obj["meta"])
        return cls(
            meta=meta,
            weights=obj["weights"],
            tokenizer_config=obj.get("tokenizer_config")
        )

    @classmethod
    def load(cls, path: str) -> 'ModelPack':
        """Load ModelPack from file using safe deserialization."""
        with open(path, 'rb') as f:
            data = f.read()
        return cls.deserialize(data)

    def save(self, path: str) -> None:
        """Save ModelPack to file using safe serialization."""
        with open(path, 'wb') as f:
            f.write(self.serialize())
