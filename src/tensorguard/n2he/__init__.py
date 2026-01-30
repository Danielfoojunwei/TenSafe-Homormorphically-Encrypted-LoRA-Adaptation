"""
TenSafe N2HE Integration Module.

Provides homomorphic encryption capabilities for privacy-preserving
LoRA adapter computation and secure inference.

WARNING: The default ToyN2HEScheme is NOT CRYPTOGRAPHICALLY SECURE.
It is intended only for development and testing. For production use,
install the N2HE C++ library.

To enable toy mode for development, set: TENSAFE_TOY_HE=1

Architecture:
    1. Encrypted LoRA Adapter Service - Compute LoRA deltas under HE
    2. Private Inference Mode - Encrypted prompts/activations path
    3. Encrypted Compute Artifacts - HE manifests, key flows, ciphertext formats

This module integrates with the N2HE library (Neural Network Homomorphic Encryption)
for optimized FHE operations on neural network computations.
"""

from .adapter import (
    AdapterEncryptionConfig,
    EncryptedLoRAAdapter,
    EncryptedLoRARuntime,
    create_encrypted_runtime,
)
from .core import (
    HESchemeParams,
    LWECiphertext,
    N2HEContext,
    N2HEScheme,
    RLWECiphertext,
    ToyModeNotEnabledError,
    ToyN2HEScheme,
    create_context,
)
from .inference import (
    EncryptedBatch,
    EncryptedOutput,
    PrivateInferenceMode,
    create_private_inference_mode,
)
from .keys import (
    EvaluationKey,
    HEKeyBundle,
    HEKeyManager,
    PublicKey,
    SecretKey,
)
from .serialization import (
    CiphertextBundle,
    CiphertextFormat,
    CiphertextSerializer,
    SerializedCiphertext,
    create_ciphertext_bundle,
    deserialize_ciphertext,
    serialize_ciphertext,
)

__all__ = [
    # Core
    "N2HEScheme",
    "N2HEContext",
    "HESchemeParams",
    "LWECiphertext",
    "RLWECiphertext",
    "ToyN2HEScheme",
    "ToyModeNotEnabledError",
    "create_context",
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
    "CiphertextSerializer",
    "CiphertextBundle",
    "SerializedCiphertext",
    "serialize_ciphertext",
    "deserialize_ciphertext",
    "create_ciphertext_bundle",
    # Factory functions
    "create_encrypted_runtime",
    "create_private_inference_mode",
]
