"""
TenSafe N2HE Integration Module.

Provides homomorphic encryption capabilities for privacy-preserving
LoRA adapter computation and secure inference.

Architecture:
    1. Encrypted LoRA Adapter Service - Compute LoRA deltas under HE
    2. Private Inference Mode - Encrypted prompts/activations path
    3. Encrypted Compute Artifacts - HE manifests, key flows, ciphertext formats

This module integrates with the N2HE library (Neural Network Homomorphic Encryption)
for optimized FHE operations on neural network computations.
"""

from .core import (
    N2HEScheme,
    N2HEContext,
    HESchemeParams,
    LWECiphertext,
    RLWECiphertext,
)
from .keys import (
    HEKeyManager,
    HEKeyBundle,
    PublicKey,
    EvaluationKey,
    SecretKey,
)
from .adapter import (
    EncryptedLoRARuntime,
    EncryptedLoRAAdapter,
    AdapterEncryptionConfig,
)
from .inference import (
    PrivateInferenceMode,
    EncryptedBatch,
    EncryptedOutput,
)
from .serialization import (
    CiphertextFormat,
    serialize_ciphertext,
    deserialize_ciphertext,
)

__all__ = [
    # Core
    "N2HEScheme",
    "N2HEContext",
    "HESchemeParams",
    "LWECiphertext",
    "RLWECiphertext",
    # Keys
    "HEKeyManager",
    "HEKeyBundle",
    "PublicKey",
    "EvaluationKey",
    "SecretKey",
    # Adapter
    "EncryptedLoRARuntime",
    "EncryptedLoRAAdapter",
    "AdapterEncryptionConfig",
    # Inference
    "PrivateInferenceMode",
    "EncryptedBatch",
    "EncryptedOutput",
    # Serialization
    "CiphertextFormat",
    "serialize_ciphertext",
    "deserialize_ciphertext",
]
