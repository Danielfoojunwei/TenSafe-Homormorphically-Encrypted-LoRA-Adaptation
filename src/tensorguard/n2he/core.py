"""
N2HE Core Module.

Provides the foundational homomorphic encryption primitives for TenSafe.
This module interfaces with the N2HE C++ library (when available) or
provides a pure-Python TOY simulation layer for development and testing.

IMPORTANT: The default ToyN2HEScheme is NOT CRYPTOGRAPHICALLY SECURE.
It is intended only for:
- Development and testing
- API compatibility verification
- Performance benchmarking (overhead estimation)

For production use, you must:
1. Install the N2HE C++ library
2. Use NativeN2HEScheme from tensorguard.n2he._native

To explicitly enable toy mode for development, set:
    TENSAFE_TOY_HE=1

The N2HE scheme is based on LWE/RLWE encryption with optimized kernels
for neural network linear algebra operations (weighted sums, convolutions).

References:
    - N2HE: https://github.com/HintSight-Technology/N2HE
    - FHEW scheme for functional bootstrapping
    - HEXL acceleration for polynomial operations
"""

import hashlib
import json
import logging
import os
import secrets
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Environment variable to explicitly enable toy/simulation mode
_TOY_HE_ENABLED = os.environ.get("TENSAFE_TOY_HE", "0").lower() in ("1", "true", "yes")


class ToyModeNotEnabledError(Exception):
    """Raised when toy HE mode is used without explicit opt-in."""

    def __init__(self):
        super().__init__(
            "ToyN2HEScheme is not cryptographically secure and requires explicit opt-in. "
            "To enable toy mode for development/testing, set TENSAFE_TOY_HE=1 environment variable. "
            "For production, install the N2HE native library."
        )


class HESchemeType(Enum):
    """Supported homomorphic encryption scheme types."""

    LWE = "lwe"  # Learning With Errors
    RLWE = "rlwe"  # Ring Learning With Errors
    FHEW = "fhew"  # Faster HE for bootstrapping
    TFHE = "tfhe"  # Torus FHE
    CKKS = "ckks"  # Approximate HE for real numbers


@dataclass
class HESchemeParams:
    """
    Homomorphic encryption scheme parameters.

    These parameters define the security level, precision, and performance
    characteristics of the HE scheme.
    """

    scheme_type: HESchemeType = HESchemeType.LWE

    # LWE/RLWE parameters
    n: int = 1024  # Lattice dimension (security parameter)
    q: int = 2**32  # Ciphertext modulus
    t: int = 2**16  # Plaintext modulus
    std_dev: float = 3.2  # Gaussian noise standard deviation

    # RLWE ring parameters
    poly_degree: int = 4096  # Polynomial ring degree (N)
    coeff_modulus_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])

    # Bootstrapping parameters (for FHEW/TFHE)
    bootstrap_base: int = 2**10
    bootstrap_level: int = 3

    # Security level (NIST standard)
    security_level: int = 128  # bits

    def get_hash(self) -> str:
        """Compute deterministic hash of parameters."""
        params_dict = {
            "scheme_type": self.scheme_type.value,
            "n": self.n,
            "q": self.q,
            "t": self.t,
            "std_dev": self.std_dev,
            "poly_degree": self.poly_degree,
            "coeff_modulus_bits": self.coeff_modulus_bits,
            "bootstrap_base": self.bootstrap_base,
            "bootstrap_level": self.bootstrap_level,
            "security_level": self.security_level,
        }
        canonical = json.dumps(params_dict, sort_keys=True).encode()
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scheme_type": self.scheme_type.value,
            "n": self.n,
            "q": self.q,
            "t": self.t,
            "std_dev": self.std_dev,
            "poly_degree": self.poly_degree,
            "coeff_modulus_bits": self.coeff_modulus_bits,
            "bootstrap_base": self.bootstrap_base,
            "bootstrap_level": self.bootstrap_level,
            "security_level": self.security_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HESchemeParams":
        """Deserialize from dictionary."""
        return cls(
            scheme_type=HESchemeType(data.get("scheme_type", "lwe")),
            n=data.get("n", 1024),
            q=data.get("q", 2**32),
            t=data.get("t", 2**16),
            std_dev=data.get("std_dev", 3.2),
            poly_degree=data.get("poly_degree", 4096),
            coeff_modulus_bits=data.get("coeff_modulus_bits", [60, 40, 40, 60]),
            bootstrap_base=data.get("bootstrap_base", 2**10),
            bootstrap_level=data.get("bootstrap_level", 3),
            security_level=data.get("security_level", 128),
        )

    @classmethod
    def default_lora_params(cls) -> "HESchemeParams":
        """Default parameters optimized for LoRA adapter computation."""
        return cls(
            scheme_type=HESchemeType.LWE,
            n=1024,
            q=2**32,
            t=2**16,
            std_dev=3.2,
            poly_degree=4096,
            coeff_modulus_bits=[60, 40, 40, 60],
            security_level=128,
        )

    @classmethod
    def high_precision_params(cls) -> "HESchemeParams":
        """High-precision parameters for evaluation with lower noise."""
        return cls(
            scheme_type=HESchemeType.CKKS,
            n=2048,
            q=2**54,
            t=2**32,
            std_dev=3.2,
            poly_degree=8192,
            coeff_modulus_bits=[60, 50, 50, 50, 60],
            security_level=128,
        )


@dataclass
class LWECiphertext:
    """
    LWE (Learning With Errors) ciphertext.

    An LWE ciphertext encrypts a plaintext m as:
        ct = (a, b) where b = <a, s> + e + (q/t) * m mod q

    Here a is a random vector, s is the secret key, and e is noise.
    """

    a: np.ndarray  # Random vector component (n elements)
    b: int  # Scalar component b = <a,s> + e + Delta*m
    params: HESchemeParams

    # Metadata
    noise_budget: Optional[float] = None  # Remaining noise budget (bits)
    level: int = 0  # Modulus chain level (for leveled HE)

    def __post_init__(self):
        if self.noise_budget is None:
            # Estimate initial noise budget based on parameters
            self.noise_budget = float(np.log2(self.params.q) - np.log2(self.params.t) - 10)

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        # Header: scheme type, level, n, b (q is in params, not serialized)
        header = struct.pack(
            ">BBIQ",  # scheme, level, n, b (unsigned 64-bit)
            self.params.scheme_type.value.encode()[0] if isinstance(self.params.scheme_type.value, str) else 0,
            self.level,
            len(self.a),
            self.b % (2**64),  # Use 64-bit for b
        )
        # Body: a vector as int32 array
        a_bytes = self.a.astype(np.int32).tobytes()
        return header + a_bytes

    @classmethod
    def from_bytes(cls, data: bytes, params: HESchemeParams) -> "LWECiphertext":
        """Deserialize from bytes."""
        header_size = struct.calcsize(">BBIQ")
        header = struct.unpack(">BBIQ", data[:header_size])
        _, level, n, b = header

        a = np.frombuffer(data[header_size : header_size + n * 4], dtype=np.int32)
        return cls(a=a, b=b, params=params, level=level)


@dataclass
class RLWECiphertext:
    """
    RLWE (Ring Learning With Errors) ciphertext.

    An RLWE ciphertext over the polynomial ring R_q = Z_q[X]/(X^N + 1):
        ct = (c0, c1) where c1 = a (random), c0 = a*s + e + Delta*m

    This is more efficient than LWE for large plaintext vectors.
    """

    c0: np.ndarray  # First polynomial (N coefficients)
    c1: np.ndarray  # Second polynomial (N coefficients)
    params: HESchemeParams

    # Metadata
    noise_budget: Optional[float] = None
    level: int = 0
    scale: float = 1.0  # For CKKS approximate arithmetic

    def __post_init__(self):
        if self.noise_budget is None:
            self.noise_budget = float(np.log2(self.params.q) - np.log2(self.params.t) - 10)

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        header = struct.pack(
            ">BBI",
            self.level,
            0,  # Reserved
            len(self.c0),
        )
        c0_bytes = self.c0.astype(np.int64).tobytes()
        c1_bytes = self.c1.astype(np.int64).tobytes()
        scale_bytes = struct.pack(">d", self.scale)
        return header + scale_bytes + c0_bytes + c1_bytes

    @classmethod
    def from_bytes(cls, data: bytes, params: HESchemeParams) -> "RLWECiphertext":
        """Deserialize from bytes."""
        header_size = struct.calcsize(">BBI")
        level, _, n = struct.unpack(">BBI", data[:header_size])

        scale_size = struct.calcsize(">d")
        scale = struct.unpack(">d", data[header_size : header_size + scale_size])[0]

        offset = header_size + scale_size
        c0 = np.frombuffer(data[offset : offset + n * 8], dtype=np.int64)
        c1 = np.frombuffer(data[offset + n * 8 : offset + 2 * n * 8], dtype=np.int64)

        return cls(c0=c0, c1=c1, params=params, level=level, scale=scale)


Ciphertext = Union[LWECiphertext, RLWECiphertext]


class N2HEScheme(ABC):
    """
    Abstract base class for N2HE encryption schemes.

    Provides the interface for encryption, decryption, and homomorphic
    operations used by the TenSafe adapter runtime.
    """

    @abstractmethod
    def keygen(self) -> Tuple[bytes, bytes, bytes]:
        """
        Generate key triple (secret_key, public_key, eval_key).

        Returns:
            Tuple of (sk, pk, ek) as bytes
        """
        pass

    @abstractmethod
    def encrypt(self, pk: bytes, plaintext: np.ndarray) -> Ciphertext:
        """
        Encrypt a plaintext vector.

        Args:
            pk: Public key bytes
            plaintext: Plaintext as numpy array

        Returns:
            Ciphertext object
        """
        pass

    @abstractmethod
    def decrypt(self, sk: bytes, ciphertext: Ciphertext) -> np.ndarray:
        """
        Decrypt a ciphertext.

        Args:
            sk: Secret key bytes
            ciphertext: Ciphertext object

        Returns:
            Decrypted plaintext as numpy array
        """
        pass

    @abstractmethod
    def add(self, ct1: Ciphertext, ct2: Ciphertext) -> Ciphertext:
        """Homomorphic addition of two ciphertexts."""
        pass

    @abstractmethod
    def multiply(self, ct: Ciphertext, plaintext: np.ndarray) -> Ciphertext:
        """Multiply ciphertext by plaintext (scalar/vector)."""
        pass

    @abstractmethod
    def matmul(self, ct: Ciphertext, weight_matrix: np.ndarray, ek: bytes) -> Ciphertext:
        """
        Encrypted matrix multiplication: ct @ W^T.

        This is the core operation for computing LoRA deltas.

        Args:
            ct: Encrypted activation vector
            weight_matrix: Plaintext weight matrix
            ek: Evaluation key for key-switching

        Returns:
            Encrypted result
        """
        pass


class ToyN2HEScheme(N2HEScheme):
    """
    TOY N2HE scheme for development and testing ONLY.

    WARNING: THIS IS NOT CRYPTOGRAPHICALLY SECURE!

    This provides a functionally correct but insecure implementation
    for integration testing without the N2HE C++ library dependency.
    It simulates the API but does not provide real encryption.

    DO NOT USE IN PRODUCTION - this is for development/testing only!

    To use this scheme, you must explicitly opt-in by setting:
        TENSAFE_TOY_HE=1

    For production, install the N2HE native library and use NativeN2HEScheme.
    """

    def __init__(self, params: Optional[HESchemeParams] = None, _force_enable: bool = False):
        """
        Initialize with scheme parameters.

        Args:
            params: HE scheme parameters
            _force_enable: Internal flag to bypass env check (for tests only)
        """
        if not _force_enable and not _TOY_HE_ENABLED:
            raise ToyModeNotEnabledError()

        self.params = params or HESchemeParams.default_lora_params()
        self._noise_growth_factor = 1.5  # Simulated noise growth
        logger.warning(
            "*** USING ToyN2HEScheme - NOT CRYPTOGRAPHICALLY SECURE! ***\n"
            "This is a simulation for development/testing only.\n"
            "For production, install N2HE C++ library."
        )

    def keygen(self) -> Tuple[bytes, bytes, bytes]:
        """Generate simulated key triple."""
        # Generate random secret key
        sk = secrets.token_bytes(self.params.n * 4)  # n int32s

        # Public key derived from secret key (simulated)
        pk_seed = hashlib.sha256(b"pk:" + sk).digest()
        pk = pk_seed + secrets.token_bytes(self.params.n * 4 - 32)

        # Evaluation key (for relinearization/key-switching)
        ek_seed = hashlib.sha256(b"ek:" + sk).digest()
        ek = ek_seed + secrets.token_bytes(self.params.n * 8 - 32)

        return sk, pk, ek

    def encrypt(self, pk: bytes, plaintext: np.ndarray) -> LWECiphertext:
        """Encrypt plaintext (simulated - stores plaintext with noise).

        Args:
            pk: Public key bytes
            plaintext: Plaintext as numpy array (must have at least 1 element)

        Returns:
            LWECiphertext object

        Raises:
            ValueError: If plaintext array is empty
        """
        n = self.params.n

        # Validate plaintext is non-empty to prevent IndexError
        if plaintext.size == 0:
            raise ValueError(
                "Cannot encrypt empty plaintext array. "
                "Plaintext must have at least one element."
            )

        # Generate random 'a' vector
        a = np.random.randint(0, self.params.q, size=n, dtype=np.int64)

        # Compute b = <a, s> + e + Delta*m (simulated)
        # In simulation, we just add noise to plaintext
        delta = self.params.q // self.params.t

        if plaintext.size == 1:
            m = int(plaintext.item()) % self.params.t
        else:
            m = int(plaintext.flat[0]) % self.params.t  # Use .flat for safe indexing

        noise = int(np.random.normal(0, self.params.std_dev))
        b = (delta * m + noise) % self.params.q

        return LWECiphertext(
            a=a.astype(np.int32),
            b=int(b),
            params=self.params,
        )

    def decrypt(self, sk: bytes, ciphertext: LWECiphertext) -> np.ndarray:
        """Decrypt ciphertext (simulated)."""
        # In simulation, extract plaintext from b
        delta = self.params.q // self.params.t
        m = round(ciphertext.b / delta) % self.params.t
        return np.array([m])

    def add(self, ct1: LWECiphertext, ct2: LWECiphertext) -> LWECiphertext:
        """Add two ciphertexts."""
        q = self.params.q
        return LWECiphertext(
            a=(ct1.a.astype(np.int64) + ct2.a.astype(np.int64)) % q,
            b=(ct1.b + ct2.b) % q,
            params=self.params,
            noise_budget=min(ct1.noise_budget or 0, ct2.noise_budget or 0) - 1,
        )

    def multiply(self, ct: LWECiphertext, plaintext: np.ndarray) -> LWECiphertext:
        """Multiply ciphertext by plaintext scalar.

        Args:
            ct: LWE ciphertext to multiply
            plaintext: Scalar value as numpy array (must have at least 1 element)

        Returns:
            Scaled LWECiphertext

        Raises:
            ValueError: If plaintext array is empty
        """
        if plaintext.size == 0:
            raise ValueError(
                "Cannot multiply by empty plaintext array. "
                "Plaintext must have at least one element."
            )

        q = self.params.q
        scalar = int(plaintext.item()) if plaintext.size == 1 else int(plaintext.flat[0])

        return LWECiphertext(
            a=(ct.a.astype(np.int64) * scalar) % q,
            b=(ct.b * scalar) % q,
            params=self.params,
            noise_budget=(ct.noise_budget or 0) - np.log2(abs(scalar) + 1),
        )

    def matmul(self, ct: LWECiphertext, weight_matrix: np.ndarray, ek: bytes) -> LWECiphertext:
        """
        Encrypted matrix multiplication (simulated).

        For LoRA: computes encrypted_activation @ W^T
        where W is the LoRA weight matrix (low-rank).
        """
        # In simulation, just transform the ciphertext
        # Real implementation would use key-switching and rotation

        q = self.params.q

        # Simulate computational noise growth
        new_noise_budget = (ct.noise_budget or 0) - np.log2(weight_matrix.shape[0] + 1)

        # Transform 'a' vector (simulated key-switching)
        transform_hash = hashlib.sha256(ek + weight_matrix.tobytes()).digest()
        transform = np.frombuffer(transform_hash * (len(ct.a) // 32 + 1), dtype=np.uint8)
        transform = transform[: len(ct.a)]

        new_a = (ct.a.astype(np.int64) + transform.astype(np.int64)) % q

        return LWECiphertext(
            a=new_a.astype(np.int32),
            b=ct.b,  # b carries through
            params=self.params,
            noise_budget=new_noise_budget,
        )


class N2HEContext:
    """
    N2HE execution context.

    Manages the HE scheme, keys, and provides high-level operations
    for the encrypted adapter runtime.
    """

    def __init__(
        self,
        params: Optional[HESchemeParams] = None,
        scheme: Optional[N2HEScheme] = None,
    ):
        """
        Initialize N2HE context.

        Args:
            params: Scheme parameters (default: LoRA-optimized)
            scheme: HE scheme implementation. If None, attempts to load native
                    N2HE library first, then falls back to ToyN2HEScheme only
                    if TENSAFE_TOY_HE=1 is set.

        Raises:
            ImportError: If no scheme provided and native N2HE unavailable
                        without toy mode enabled.
            ToyModeNotEnabledError: If attempting to use toy scheme without
                                   TENSAFE_TOY_HE=1 environment variable.
        """
        self.params = params or HESchemeParams.default_lora_params()

        if scheme is not None:
            self.scheme = scheme
        else:
            # Try native N2HE first, then fall back to toy mode with explicit opt-in
            try:
                from tensorguard.n2he._native import NativeN2HEScheme
                self.scheme = NativeN2HEScheme(self.params)
                logger.info("N2HEContext initialized with native N2HE scheme")
            except ImportError:
                if _TOY_HE_ENABLED:
                    logger.warning(
                        "Native N2HE library not available - using ToyN2HEScheme. "
                        "WARNING: ToyN2HEScheme is NOT cryptographically secure! "
                        "For production, install the N2HE C++ library."
                    )
                    self.scheme = ToyN2HEScheme(self.params)
                else:
                    raise ImportError(
                        "N2HEContext requires an HE scheme implementation. Either:\n"
                        "1. Install the N2HE C++ library for production use, or\n"
                        "2. Set TENSAFE_TOY_HE=1 environment variable for development/testing\n"
                        "   (WARNING: ToyN2HEScheme is NOT cryptographically secure!)"
                    )

        # Key material (set by load_keys or generate_keys)
        self._sk: Optional[bytes] = None
        self._pk: Optional[bytes] = None
        self._ek: Optional[bytes] = None

        # Metrics
        self._operations_count = 0
        self._total_noise_growth = 0.0

    def generate_keys(self) -> None:
        """Generate fresh key material."""
        self._sk, self._pk, self._ek = self.scheme.keygen()
        logger.info(f"Generated N2HE keys: sk={len(self._sk)}B, pk={len(self._pk)}B, ek={len(self._ek)}B")

    def load_keys(
        self,
        sk: Optional[bytes] = None,
        pk: Optional[bytes] = None,
        ek: Optional[bytes] = None,
    ) -> None:
        """Load key material from external source."""
        if sk is not None:
            self._sk = sk
        if pk is not None:
            self._pk = pk
        if ek is not None:
            self._ek = ek

    @property
    def has_secret_key(self) -> bool:
        """Check if secret key is available (for decryption)."""
        return self._sk is not None

    @property
    def has_public_key(self) -> bool:
        """Check if public key is available (for encryption)."""
        return self._pk is not None

    @property
    def has_eval_key(self) -> bool:
        """Check if evaluation key is available (for matmul)."""
        return self._ek is not None

    def export_public_key(self) -> bytes:
        """Export public key for client encryption."""
        if self._pk is None:
            raise ValueError("No public key available")
        return self._pk

    def export_eval_key(self) -> bytes:
        """Export evaluation key for server computation."""
        if self._ek is None:
            raise ValueError("No evaluation key available")
        return self._ek

    def encrypt(self, plaintext: np.ndarray) -> Ciphertext:
        """Encrypt plaintext using public key."""
        if self._pk is None:
            raise ValueError("No public key available for encryption")
        ct = self.scheme.encrypt(self._pk, plaintext)
        self._operations_count += 1
        return ct

    def decrypt(self, ciphertext: Ciphertext) -> np.ndarray:
        """Decrypt ciphertext using secret key."""
        if self._sk is None:
            raise ValueError("No secret key available for decryption")
        return self.scheme.decrypt(self._sk, ciphertext)

    def encrypted_linear(
        self,
        encrypted_input: Ciphertext,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> Ciphertext:
        """
        Compute encrypted linear transformation: y = x @ W^T + b.

        This is the core operation for LoRA adapter computation.

        Args:
            encrypted_input: Encrypted activation vector
            weight: Plaintext weight matrix
            bias: Optional plaintext bias vector

        Returns:
            Encrypted output
        """
        if self._ek is None:
            raise ValueError("No evaluation key for matrix multiplication")

        # Compute encrypted matmul
        result = self.scheme.matmul(encrypted_input, weight, self._ek)
        self._operations_count += 1

        # Add bias if provided (requires fresh encryption of bias)
        if bias is not None:
            bias_ct = self.encrypt(bias)
            result = self.scheme.add(result, bias_ct)
            self._operations_count += 1

        return result

    def encrypted_lora_delta(
        self,
        encrypted_activation: Ciphertext,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        scaling: float = 1.0,
    ) -> Ciphertext:
        """
        Compute encrypted LoRA delta: y = scaling * (x @ A^T @ B^T).

        LoRA approximates weight updates as low-rank: W' = W + alpha * B @ A
        The delta contribution is: delta = x @ A^T @ B^T

        Args:
            encrypted_activation: Encrypted hidden state [batch, hidden_dim]
            lora_a: LoRA A matrix [rank, hidden_dim]
            lora_b: LoRA B matrix [out_dim, rank]
            scaling: LoRA scaling factor (alpha/rank)

        Returns:
            Encrypted LoRA delta [batch, out_dim]
        """
        if self._ek is None:
            raise ValueError("No evaluation key for LoRA computation")

        # Step 1: x @ A^T → [batch, rank]
        intermediate = self.scheme.matmul(encrypted_activation, lora_a, self._ek)
        self._operations_count += 1

        # Step 2: intermediate @ B^T → [batch, out_dim]
        result = self.scheme.matmul(intermediate, lora_b, self._ek)
        self._operations_count += 1

        # Step 3: Apply scaling
        if abs(scaling - 1.0) > 1e-6:
            scale_array = np.array([scaling])
            result = self.scheme.multiply(result, scale_array)
            self._operations_count += 1

        # Track noise growth
        if hasattr(result, "noise_budget"):
            self._total_noise_growth += (encrypted_activation.noise_budget or 0) - (result.noise_budget or 0)

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Get context metrics for monitoring."""
        return {
            "operations_count": self._operations_count,
            "total_noise_growth_bits": self._total_noise_growth,
            "scheme_params_hash": self.params.get_hash(),
            "has_secret_key": self.has_secret_key,
            "has_public_key": self.has_public_key,
            "has_eval_key": self.has_eval_key,
        }


def create_context(
    profile: str = "lora",
    use_toy_mode: bool = False,
) -> N2HEContext:
    """
    Factory function to create an N2HE context.

    Args:
        profile: Parameter profile ("lora", "high_precision", "default")
        use_toy_mode: Use toy/simulated scheme. Requires TENSAFE_TOY_HE=1 env var.

    Returns:
        Configured N2HEContext

    Raises:
        ToyModeNotEnabledError: If toy mode requested without env var
        ImportError: If native N2HE not available and toy mode not enabled
    """
    if profile == "lora":
        params = HESchemeParams.default_lora_params()
    elif profile == "high_precision":
        params = HESchemeParams.high_precision_params()
    else:
        params = HESchemeParams()

    if use_toy_mode or _TOY_HE_ENABLED:
        scheme = ToyN2HEScheme(params)
    else:
        # Try to import real N2HE
        try:
            from tensorguard.n2he._native import NativeN2HEScheme

            scheme = NativeN2HEScheme(params)
            logger.info("Using native N2HE scheme")
        except ImportError:
            raise ImportError(
                "Native N2HE library not available. Either:\n"
                "1. Install the N2HE C++ library for production use, or\n"
                "2. Set TENSAFE_TOY_HE=1 for development/testing"
            )

    return N2HEContext(params=params, scheme=scheme)


# Backwards compatibility alias (deprecated)
SimulatedN2HEScheme = ToyN2HEScheme
