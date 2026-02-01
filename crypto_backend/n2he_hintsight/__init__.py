"""
HintSight N2HE Backend for TenSafe.

Provides LWE-based homomorphic encryption using the HintSight N2HE library.
This implementation uses FasterNTT for polynomial operations, making it
portable across different CPU architectures (not Intel-specific like HEXL).

Repository: https://github.com/HintSight-Technology/N2HE
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Add lib directory to path
_LIB_DIR = Path(__file__).parent / "lib"
if _LIB_DIR.exists():
    sys.path.insert(0, str(_LIB_DIR))

# Try to import the native module
_NATIVE_AVAILABLE = False
_NATIVE_MODULE = None

try:
    import n2he_hintsight_native
    _NATIVE_MODULE = n2he_hintsight_native
    _NATIVE_AVAILABLE = True
    logger.info(f"HintSight N2HE native module loaded: version {n2he_hintsight_native.__version__}")
except ImportError as e:
    logger.warning(f"HintSight N2HE native module not available: {e}")
    logger.warning("Run ./scripts/build_n2he_hintsight.sh to build the native module")


class HEBackendNotAvailableError(Exception):
    """Raised when the HE backend is not available."""
    pass


@dataclass
class N2HEParams:
    """N2HE LWE encryption parameters."""
    n: int = 1024                    # Lattice dimension
    q: int = 2**32                   # Ciphertext modulus
    t: int = 2**16                   # Plaintext modulus
    std_dev: float = 3.2             # Gaussian noise standard deviation
    security_level: int = 128        # Security bits

    @classmethod
    def default_lora_params(cls) -> "N2HEParams":
        """Default parameters optimized for LoRA computation."""
        return cls(
            n=1024,
            q=2**32,
            t=2**16,
            std_dev=3.2,
            security_level=128
        )

    @classmethod
    def high_security_params(cls) -> "N2HEParams":
        """Higher security parameters (slower but more secure)."""
        return cls(
            n=2048,
            q=2**54,
            t=2**20,
            std_dev=3.2,
            security_level=192
        )


class N2HECiphertext:
    """Wrapper for N2HE ciphertext with metadata."""

    def __init__(self, native_ct, context: "N2HEHintSightBackend"):
        if not _NATIVE_AVAILABLE:
            raise HEBackendNotAvailableError("HintSight N2HE backend not available")
        self._ct = native_ct
        self._ctx = context

    @property
    def noise_budget(self) -> float:
        """Get remaining noise budget estimate."""
        if isinstance(self._ct, list):
            return min(ct.noise_budget for ct in self._ct) if self._ct else 0.0
        return self._ct.noise_budget

    @property
    def level(self) -> int:
        """Get current level (multiplicative depth)."""
        if isinstance(self._ct, list):
            return max(ct.level for ct in self._ct) if self._ct else 0
        return self._ct.level

    def serialize(self) -> bytes:
        """Serialize ciphertext to bytes."""
        if isinstance(self._ct, list):
            import struct
            data = struct.pack("<I", len(self._ct))
            for ct in self._ct:
                ct_bytes = bytes(ct.serialize())
                data += struct.pack("<I", len(ct_bytes)) + ct_bytes
            return data
        return bytes(self._ct.serialize())

    @classmethod
    def deserialize(cls, data: bytes, context: "N2HEHintSightBackend") -> "N2HECiphertext":
        """Deserialize ciphertext from bytes."""
        import struct
        # Check if it's a vector
        if len(data) >= 4:
            count = struct.unpack("<I", data[:4])[0]
            if count > 0 and count < 10000:  # Sanity check
                offset = 4
                cts = []
                for _ in range(count):
                    ct_len = struct.unpack("<I", data[offset:offset+4])[0]
                    offset += 4
                    ct_bytes = data[offset:offset+ct_len]
                    offset += ct_len
                    cts.append(_NATIVE_MODULE.LWECiphertext.deserialize(ct_bytes))
                return cls(cts, context)
        # Single ciphertext
        native_ct = _NATIVE_MODULE.LWECiphertext.deserialize(data)
        return cls(native_ct, context)


class N2HEHintSightBackend:
    """
    HintSight N2HE Backend for LWE-based homomorphic encryption.

    This backend uses the N2HE library from HintSight Technology,
    which provides neural network-optimized HE operations using
    FasterNTT for polynomial arithmetic.

    Key features:
    - LWE-based encryption for weighted sums
    - FHEW ciphertexts for non-polynomial activations
    - Portable across CPU architectures (no Intel HEXL dependency)
    """

    def __init__(self, params: Optional[N2HEParams] = None):
        if not _NATIVE_AVAILABLE:
            raise HEBackendNotAvailableError(
                "HintSight N2HE native module not available.\n"
                "Build with: ./scripts/build_n2he_hintsight.sh"
            )

        self._params = params or N2HEParams.default_lora_params()
        self._native_ctx: Optional[Any] = None
        self._setup_complete = False

    def is_available(self) -> bool:
        """Check if the backend is available."""
        return _NATIVE_AVAILABLE

    def get_backend_name(self) -> str:
        """Get backend name."""
        return "HintSight-N2HE"

    def setup_context(self) -> None:
        """Initialize N2HE context with parameters."""
        self._native_ctx = _NATIVE_MODULE.N2HEContext(
            self._params.n,
            self._params.q,
            self._params.t,
            self._params.std_dev,
            self._params.security_level
        )
        self._setup_complete = True
        logger.info(
            f"N2HE context initialized: "
            f"n={self._params.n}, q=2^{int(np.log2(self._params.q))}, "
            f"t=2^{int(np.log2(self._params.t))}"
        )

    def generate_keys(self) -> Tuple[bytes, bytes, bytes]:
        """Generate encryption keys (secret, public, evaluation)."""
        if not self._setup_complete:
            raise RuntimeError("Call setup_context() first")

        self._native_ctx.generate_keys()
        sk_bytes, pk_bytes, ek_bytes = self._native_ctx.get_keys()

        logger.info("Keys generated successfully")
        return bytes(sk_bytes), bytes(pk_bytes), bytes(ek_bytes)

    def set_keys(self, sk_bytes: bytes, pk_bytes: bytes, ek_bytes: bytes) -> None:
        """Set pre-generated keys."""
        if not self._setup_complete:
            raise RuntimeError("Call setup_context() first")
        self._native_ctx.set_keys(sk_bytes, pk_bytes, ek_bytes)
        logger.info("Keys loaded successfully")

    def get_context_params(self) -> Dict[str, Any]:
        """Get context parameters for verification."""
        if not self._setup_complete:
            return {}
        return dict(self._native_ctx.get_params())

    def encrypt(self, plaintext: np.ndarray) -> N2HECiphertext:
        """Encrypt a plaintext vector."""
        if self._native_ctx is None or not self._native_ctx.keys_generated:
            raise RuntimeError("Keys not generated. Call generate_keys() first.")

        native_cts = self._native_ctx.encrypt(plaintext.astype(np.float64).flatten())
        return N2HECiphertext(native_cts, self)

    def decrypt(self, ciphertext: N2HECiphertext, output_size: int = 0) -> np.ndarray:
        """Decrypt a ciphertext to plaintext vector."""
        if self._native_ctx is None or not self._native_ctx.keys_generated:
            raise RuntimeError("Keys not available for decryption")

        result = np.array(self._native_ctx.decrypt(ciphertext._ct))
        if output_size > 0 and output_size < len(result):
            result = result[:output_size]
        return result

    def add(self, ct1: N2HECiphertext, ct2: N2HECiphertext) -> N2HECiphertext:
        """Homomorphic addition of two ciphertexts."""
        if isinstance(ct1._ct, list) and isinstance(ct2._ct, list):
            native_result = self._native_ctx.add_vectors(ct1._ct, ct2._ct)
        else:
            native_result = self._native_ctx.add(ct1._ct, ct2._ct)
        return N2HECiphertext(native_result, self)

    def multiply_plain(
        self,
        ct: N2HECiphertext,
        plaintext: np.ndarray
    ) -> N2HECiphertext:
        """Multiply ciphertext by plaintext."""
        native_result = self._native_ctx.multiply_plain(
            ct._ct,
            plaintext.astype(np.float64).flatten()
        )
        return N2HECiphertext(native_result, self)

    def matmul(
        self,
        ct: N2HECiphertext,
        weight: np.ndarray
    ) -> N2HECiphertext:
        """
        Encrypted matrix multiplication: ct @ weight^T.

        This is the core operation for computing LoRA deltas.
        """
        native_result = self._native_ctx.matmul(
            ct._ct,
            weight.astype(np.float64)
        )
        return N2HECiphertext(native_result, self)

    def lora_delta(
        self,
        ct_x: N2HECiphertext,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0
    ) -> N2HECiphertext:
        """
        Compute LoRA delta: scaling * (x @ A^T @ B^T).

        Args:
            ct_x: Encrypted activation vector
            lora_a: LoRA A matrix [rank, in_features]
            lora_b: LoRA B matrix [out_features, rank]
            scaling: LoRA scaling factor

        Returns:
            Encrypted LoRA delta
        """
        native_result = self._native_ctx.lora_delta(
            ct_x._ct,
            lora_a.astype(np.float64),
            lora_b.astype(np.float64),
            scaling
        )
        return N2HECiphertext(native_result, self)

    def get_operation_stats(self) -> Dict[str, int]:
        """Get operation statistics."""
        if self._native_ctx is None:
            return {"operations": 0, "additions": 0, "multiplications": 0}
        return dict(self._native_ctx.get_stats())

    def reset_stats(self) -> None:
        """Reset operation counters."""
        if self._native_ctx is not None:
            self._native_ctx.reset_stats()

    def get_noise_budget(self, ct: N2HECiphertext) -> float:
        """Get noise budget estimate for ciphertext."""
        return ct.noise_budget


def verify_backend() -> Dict[str, Any]:
    """
    Verify the HintSight N2HE backend is properly installed and functional.

    Returns dict with verification results. Raises if backend not available.
    """
    if not _NATIVE_AVAILABLE:
        raise HEBackendNotAvailableError(
            "HintSight N2HE native module not available.\n"
            "Build with: ./scripts/build_n2he_hintsight.sh"
        )

    backend = N2HEHintSightBackend()
    backend.setup_context()
    backend.generate_keys()

    params = backend.get_context_params()

    # Test encrypt/decrypt
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    ct = backend.encrypt(test_data)
    decrypted = backend.decrypt(ct, len(test_data))

    error = np.max(np.abs(test_data - decrypted))

    # Test LoRA delta
    lora_a = np.random.randn(8, 4).astype(np.float64) * 0.1  # rank=8, in=4
    lora_b = np.random.randn(4, 8).astype(np.float64) * 0.1  # out=4, rank=8
    scaling = 0.5

    ct_x = backend.encrypt(test_data)
    ct_delta = backend.lora_delta(ct_x, lora_a, lora_b, scaling)
    delta_decrypted = backend.decrypt(ct_delta, 4)

    # Compute expected delta
    expected_delta = scaling * (test_data @ lora_a.T @ lora_b.T)
    delta_error = np.max(np.abs(expected_delta - delta_decrypted))

    return {
        "backend": "HintSight-N2HE",
        "available": True,
        "params": params,
        "test_encrypt_decrypt": {
            "input": test_data.tolist(),
            "output": decrypted.tolist(),
            "max_error": float(error),
            "passed": error < 0.1,  # LWE has higher error than CKKS
        },
        "test_lora_delta": {
            "input_dim": 4,
            "rank": 8,
            "scaling": scaling,
            "max_error": float(delta_error),
            "passed": delta_error < 0.5,  # Higher tolerance for LWE
        }
    }


# Export public API
__all__ = [
    "N2HEHintSightBackend",
    "N2HECiphertext",
    "N2HEParams",
    "HEBackendNotAvailableError",
    "verify_backend",
]
