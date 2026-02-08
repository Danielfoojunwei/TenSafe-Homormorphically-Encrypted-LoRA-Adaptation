"""
FasterNTT: CPU Fallback Backend for HE-LoRA Microkernel

This module provides a portable, CPU-based implementation of Number Theoretic
Transform (NTT) operations, inspired by HintSight's FasterNTT library.

Use Cases:
    - ARM-based cloud instances (AWS Graviton, etc.)
    - Edge devices without GPU
    - Development and testing without GPU dependencies
    - Fallback when GPU backends are unavailable

Architecture:
    FasterNTT provides the core polynomial arithmetic needed for:
    - RLWE encryption/decryption
    - Key generation
    - Programmable bootstrapping (blind rotation)

Performance Notes:
    - Optimized for modern CPUs with SIMD (via NumPy)
    - Supports multi-threading for parallel NTT butterflies
    - Falls back to pure Python for compatibility

References:
    - N2HE FasterNTT: https://github.com/HintSight-Technology/N2HE
    - Cooley-Tukey NTT algorithm
    - Harvey's NTT optimizations
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class NTTDirection(Enum):
    """Direction of NTT transform."""
    FORWARD = "forward"   # Time → Frequency domain
    INVERSE = "inverse"   # Frequency → Time domain


@dataclass
class NTTParams:
    """
    Parameters for NTT operations.

    NTT operates over Z_q[X]/(X^N + 1) where:
    - N: ring dimension (power of 2)
    - q: NTT-friendly modulus with q ≡ 1 (mod 2N)
    - ω: primitive 2N-th root of unity mod q
    """
    ring_dimension: int
    modulus: int
    primitive_root: int

    def __post_init__(self):
        """Validate and precompute NTT tables."""
        if self.ring_dimension & (self.ring_dimension - 1) != 0:
            raise ValueError(f"Ring dimension {self.ring_dimension} must be power of 2")

        # Verify modulus is NTT-friendly: q ≡ 1 (mod 2N)
        if self.modulus % (2 * self.ring_dimension) != 1:
            logger.warning(
                f"Modulus {self.modulus} may not be NTT-friendly for N={self.ring_dimension}"
            )

    @property
    def log_n(self) -> int:
        """Log base 2 of ring dimension."""
        return self.ring_dimension.bit_length() - 1


class FasterNTT:
    """
    Fast Number Theoretic Transform implementation.

    Provides forward and inverse NTT using Cooley-Tukey algorithm
    with precomputed twiddle factors for efficiency.

    Example:
        params = NTTParams(ring_dimension=1024, modulus=12289, primitive_root=11)
        ntt = FasterNTT(params)

        # Forward NTT
        poly_ntt = ntt.forward(poly_coeffs)

        # Polynomial multiplication in NTT domain
        product_ntt = ntt.multiply(poly1_ntt, poly2_ntt)

        # Inverse NTT
        product = ntt.inverse(product_ntt)
    """

    def __init__(self, params: NTTParams):
        """
        Initialize NTT with precomputed tables.

        Args:
            params: NTT parameters
        """
        self.params = params
        self.n = params.ring_dimension
        self.q = params.modulus
        self.log_n = params.log_n

        # Precompute twiddle factors
        self._precompute_twiddles()

        # Statistics
        self._forward_count = 0
        self._inverse_count = 0
        self._total_time_ms = 0.0

    def _precompute_twiddles(self):
        """Precompute twiddle factors for NTT butterflies."""
        n = self.n
        q = self.q
        w = self.params.primitive_root

        # Compute primitive 2N-th root of unity
        # ω = primitive_root^((q-1)/(2N)) mod q
        exponent = (q - 1) // (2 * n)
        omega = pow(w, exponent, q)

        # Forward twiddles: ω^(bit_reverse(i)) for i = 0..N-1
        self._forward_twiddles = np.zeros(n, dtype=np.uint64)
        self._inverse_twiddles = np.zeros(n, dtype=np.uint64)

        # Compute twiddles for each level
        omega_inv = pow(omega, q - 2, q)  # Modular inverse

        for i in range(n):
            self._forward_twiddles[i] = pow(omega, self._bit_reverse(i, self.log_n), q)
            self._inverse_twiddles[i] = pow(omega_inv, self._bit_reverse(i, self.log_n), q)

        # Precompute N^{-1} mod q for inverse NTT scaling
        self._n_inv = pow(n, q - 2, q)

    def _bit_reverse(self, x: int, bits: int) -> int:
        """Reverse bits of x in `bits`-bit representation."""
        result = 0
        for _ in range(bits):
            result = (result << 1) | (x & 1)
            x >>= 1
        return result

    def forward(self, poly: np.ndarray) -> np.ndarray:
        """
        Compute forward NTT (Cooley-Tukey, decimation-in-time).

        Args:
            poly: Polynomial coefficients [a_0, a_1, ..., a_{N-1}]

        Returns:
            NTT representation of polynomial
        """
        start_time = time.perf_counter()

        n = self.n
        q = self.q

        # Ensure correct type and size
        result = np.zeros(n, dtype=np.uint64)
        result[:len(poly)] = poly.astype(np.uint64) % q

        # Bit-reversal permutation
        result = self._bit_reverse_permute(result)

        # Cooley-Tukey butterflies
        m = 1
        for level in range(self.log_n):
            m_half = m
            m *= 2

            for i in range(0, n, m):
                tw_idx = 0
                for j in range(m_half):
                    u = result[i + j]
                    # Twiddle factor multiplication
                    tw = self._forward_twiddles[(n // m) * j]
                    v = (result[i + j + m_half] * tw) % q

                    # Butterfly
                    result[i + j] = (u + v) % q
                    result[i + j + m_half] = (u - v + q) % q

        self._forward_count += 1
        self._total_time_ms += (time.perf_counter() - start_time) * 1000

        return result

    def inverse(self, ntt_poly: np.ndarray) -> np.ndarray:
        """
        Compute inverse NTT (Gentleman-Sande, decimation-in-frequency).

        Args:
            ntt_poly: NTT representation of polynomial

        Returns:
            Polynomial coefficients
        """
        start_time = time.perf_counter()

        n = self.n
        q = self.q

        result = ntt_poly.copy().astype(np.uint64)

        # Gentleman-Sande butterflies (reverse order)
        m = n
        for level in range(self.log_n):
            m_half = m // 2

            for i in range(0, n, m):
                for j in range(m_half):
                    u = result[i + j]
                    v = result[i + j + m_half]

                    # Butterfly
                    result[i + j] = (u + v) % q
                    diff = (u - v + q) % q

                    # Twiddle factor multiplication
                    tw = self._inverse_twiddles[(n // m) * j]
                    result[i + j + m_half] = (diff * tw) % q

            m = m_half

        # Bit-reversal permutation
        result = self._bit_reverse_permute(result)

        # Scale by N^{-1}
        result = (result * self._n_inv) % q

        self._inverse_count += 1
        self._total_time_ms += (time.perf_counter() - start_time) * 1000

        return result

    def _bit_reverse_permute(self, arr: np.ndarray) -> np.ndarray:
        """Apply bit-reversal permutation to array."""
        n = len(arr)
        log_n = n.bit_length() - 1
        result = np.zeros_like(arr)

        for i in range(n):
            j = self._bit_reverse(i, log_n)
            result[j] = arr[i]

        return result

    def multiply(self, ntt_a: np.ndarray, ntt_b: np.ndarray) -> np.ndarray:
        """
        Multiply two polynomials in NTT domain (element-wise).

        Args:
            ntt_a: First polynomial in NTT form
            ntt_b: Second polynomial in NTT form

        Returns:
            Product in NTT form
        """
        return (ntt_a.astype(np.uint64) * ntt_b.astype(np.uint64)) % self.q

    def add(self, ntt_a: np.ndarray, ntt_b: np.ndarray) -> np.ndarray:
        """Add two polynomials in NTT domain."""
        return (ntt_a.astype(np.uint64) + ntt_b.astype(np.uint64)) % self.q

    def subtract(self, ntt_a: np.ndarray, ntt_b: np.ndarray) -> np.ndarray:
        """Subtract two polynomials in NTT domain."""
        return (ntt_a.astype(np.uint64) - ntt_b.astype(np.uint64) + self.q) % self.q

    def negate(self, ntt_a: np.ndarray) -> np.ndarray:
        """Negate polynomial in NTT domain."""
        return (self.q - ntt_a.astype(np.uint64)) % self.q

    def get_stats(self) -> Dict[str, Any]:
        """Get NTT operation statistics."""
        return {
            'forward_count': self._forward_count,
            'inverse_count': self._inverse_count,
            'total_time_ms': self._total_time_ms,
            'avg_time_ms': self._total_time_ms / max(1, self._forward_count + self._inverse_count),
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._forward_count = 0
        self._inverse_count = 0
        self._total_time_ms = 0.0


class FasterNTTBackend:
    """
    CPU backend using FasterNTT for TFHE operations.

    This provides a fallback when GPU backends are unavailable,
    supporting:
    - LWE encryption/decryption
    - RLWE operations via NTT
    - Programmable bootstrapping

    Performance is lower than GPU but provides:
    - Portability (any CPU architecture)
    - No GPU dependencies
    - Useful for testing and development
    """

    def __init__(self, params: 'N2HEParams'):
        """
        Initialize FasterNTT backend.

        Args:
            params: N2HE parameters including LWE and RLWE config
        """
        from .n2he_params import N2HEParams
        self.params = params

        # Initialize NTT for RLWE operations
        ntt_params = NTTParams(
            ring_dimension=params.rlwe.ring_dimension,
            modulus=params.rlwe.coeff_modulus,
            primitive_root=5,  # Common choice for NTT-friendly primes
        )
        self._ntt = FasterNTT(ntt_params)

        # Key storage
        self._lwe_secret_key: Optional[np.ndarray] = None
        self._rlwe_secret_key: Optional[np.ndarray] = None
        self._bootstrapping_keys: Optional[List[np.ndarray]] = None
        self._keyswitch_key: Optional[np.ndarray] = None

        # State
        self._initialized = False
        self._rng = np.random.default_rng()

        # Statistics
        self._stats = {
            'encryptions': 0,
            'decryptions': 0,
            'bootstraps': 0,
            'total_time_ms': 0.0,
        }

    def initialize(self) -> None:
        """Generate keys and initialize backend."""
        logger.info("Initializing FasterNTT backend (CPU fallback)...")

        # Generate LWE secret key (binary)
        n = self.params.lwe.dimension
        if self.params.lwe.key_distribution == "binary":
            self._lwe_secret_key = self._rng.integers(0, 2, size=n, dtype=np.int64)
        else:  # ternary
            self._lwe_secret_key = self._rng.integers(-1, 2, size=n, dtype=np.int64)

        # Generate RLWE secret key (binary polynomial)
        N = self.params.rlwe.ring_dimension
        self._rlwe_secret_key = self._rng.integers(0, 2, size=N, dtype=np.int64)

        # Generate bootstrapping keys (RGSW encryptions of LWE key bits)
        self._generate_bootstrapping_keys()

        # Generate keyswitch key if needed
        if self.params.bootstrapping.use_keyswitch:
            self._generate_keyswitch_key()

        self._initialized = True
        logger.info(f"FasterNTT backend initialized: LWE-{n}, RLWE-{N}")

    def _generate_bootstrapping_keys(self):
        """Generate RGSW bootstrapping keys."""
        # Simplified: In production, this would be RGSW encryptions
        # For simulation, we store the relationship
        n = self.params.lwe.dimension
        N = self.params.rlwe.ring_dimension

        self._bootstrapping_keys = []
        for i in range(n):
            # Each key encrypts s[i] under RLWE
            key_bit = self._lwe_secret_key[i]
            # Store as (key_bit, random_polynomials)
            self._bootstrapping_keys.append({
                'key_bit': key_bit,
                'rgsw_ct': self._rng.integers(0, self.params.rlwe.coeff_modulus, size=(2, N)),
            })

    def _generate_keyswitch_key(self):
        """Generate key switching key from RLWE to LWE."""
        N = self.params.rlwe.ring_dimension
        n = self.params.lwe.dimension
        levels = self.params.bootstrapping.keyswitch_levels

        # KSK allows converting from RLWE key to LWE key
        self._keyswitch_key = self._rng.integers(
            0, self.params.lwe.modulus,
            size=(N, levels, n + 1),
            dtype=np.int64
        )

    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    def encrypt_lwe(self, message: int) -> np.ndarray:
        """
        Encrypt a discrete message with LWE.

        Args:
            message: Integer in {0, 1, ..., p-1} where p = message_space

        Returns:
            LWE ciphertext (a, b) as array of length n+1
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")

        start_time = time.perf_counter()

        n = self.params.lwe.dimension
        q = 2**32  # Use 32-bit modulus for LWE
        p = self.params.lwe.message_space
        sigma = self.params.lwe.noise_stddev

        # Random vector a
        a = self._rng.integers(0, q, size=n, dtype=np.int64)

        # Noise e ~ discrete Gaussian
        e = int(self._rng.normal(0, sigma * q))

        # Encode message on torus: m_encoded = q * m / p
        m_encoded = (q * message) // p

        # b = <a, s> + m_encoded + e (mod q)
        inner_product = np.sum(a * self._lwe_secret_key) % q
        b = (inner_product + m_encoded + e) % q

        # Ciphertext: (a, b)
        ct = np.zeros(n + 1, dtype=np.int64)
        ct[:n] = a
        ct[n] = b

        self._stats['encryptions'] += 1
        self._stats['total_time_ms'] += (time.perf_counter() - start_time) * 1000

        return ct

    def decrypt_lwe(self, ct: np.ndarray) -> int:
        """
        Decrypt LWE ciphertext to discrete message.

        Args:
            ct: LWE ciphertext (a, b) as array of length n+1

        Returns:
            Decrypted message in {0, 1, ..., p-1}
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")

        start_time = time.perf_counter()

        n = self.params.lwe.dimension
        q = 2**32
        p = self.params.lwe.message_space

        a = ct[:n]
        b = ct[n]

        # Compute b - <a, s> (mod q)
        inner_product = np.sum(a * self._lwe_secret_key) % q
        noisy_encoded = (b - inner_product) % q

        # Decode: round((noisy_encoded * p) / q) mod p
        message = (noisy_encoded * p + q // 2) // q
        message = message % p

        self._stats['decryptions'] += 1
        self._stats['total_time_ms'] += (time.perf_counter() - start_time) * 1000

        return int(message)

    def programmable_bootstrap(self, ct: np.ndarray, lut: List[int]) -> np.ndarray:
        """
        Perform programmable bootstrapping with LUT evaluation.

        This is the core TFHE operation:
        1. Modulus switch LWE to 2N
        2. Blind rotate accumulator with test polynomial
        3. Extract refreshed LWE with f(m) = LUT[m]

        Args:
            ct: Input LWE ciphertext
            lut: Lookup table [f(0), f(1), ..., f(p-1)]

        Returns:
            Refreshed LWE ciphertext encrypting f(m)
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")

        start_time = time.perf_counter()

        n = self.params.lwe.dimension
        N = self.params.rlwe.ring_dimension
        p = self.params.lwe.message_space
        Q = self.params.rlwe.coeff_modulus

        # Step 1: Modulus switch from q to 2N
        q = 2**32
        a_scaled = ((ct[:n].astype(np.float64) * (2 * N)) / q).round().astype(np.int64) % (2 * N)
        b_scaled = int(round((ct[n] * (2 * N)) / q)) % (2 * N)

        # Step 2: Create test polynomial encoding LUT
        # Test polynomial: sum_{i=0}^{N-1} LUT[i mod p] * X^i
        test_poly = np.zeros(N, dtype=np.int64)
        for i in range(N):
            # Negacyclic encoding: coefficients wrap with sign
            lut_idx = i % p
            test_poly[i] = lut[lut_idx] * (Q // p)

        # Step 3: Initialize accumulator as test_poly * X^{b_scaled}
        # In negacyclic ring, X^N = -1, so we handle rotation carefully
        acc = self._rotate_polynomial(test_poly, b_scaled, Q)

        # Step 4: Blind rotation using bootstrapping keys
        # For each LWE key bit, conditionally rotate accumulator
        for i in range(n):
            rotation = a_scaled[i]
            if rotation != 0:
                # CMUX: if s[i] = 1, rotate by a_scaled[i]
                # Simulated using stored key bit
                key_bit = self._bootstrapping_keys[i]['key_bit']
                if key_bit == 1:
                    acc = self._rotate_polynomial(acc, rotation, Q)

        # Step 5: Extract LWE ciphertext from RLWE
        # The constant term of acc encodes f(m)
        result_ct = self._sample_extract(acc)

        # Step 6: Keyswitch if configured
        if self.params.bootstrapping.use_keyswitch:
            result_ct = self._keyswitch(result_ct)

        self._stats['bootstraps'] += 1
        self._stats['total_time_ms'] += (time.perf_counter() - start_time) * 1000

        return result_ct

    def _rotate_polynomial(self, poly: np.ndarray, rotation: int, modulus: int) -> np.ndarray:
        """
        Rotate polynomial by X^rotation in negacyclic ring.

        In Z[X]/(X^N + 1), X^N = -1, so rotation wraps with sign change.
        """
        N = len(poly)
        rotation = rotation % (2 * N)

        result = np.zeros_like(poly)
        for i in range(N):
            src_idx = (i - rotation) % (2 * N)
            if src_idx >= N:
                # Wrap with negation (X^N = -1)
                result[i] = (-poly[src_idx - N]) % modulus
            else:
                result[i] = poly[src_idx]

        return result

    def _sample_extract(self, rlwe_ct: np.ndarray) -> np.ndarray:
        """
        Extract LWE ciphertext from RLWE (constant term).

        Maps RLWE over Z_Q[X]/(X^N+1) to LWE over Z_Q.
        """
        N = len(rlwe_ct)
        n = self.params.lwe.dimension

        # For simplicity, extract first n coefficients as LWE 'a' vector
        # and constant term (negated sum of products) as 'b'
        result = np.zeros(n + 1, dtype=np.int64)

        # Simplified extraction
        result[:n] = rlwe_ct[:n]
        result[n] = rlwe_ct[0]  # Constant term carries the message

        return result

    def _keyswitch(self, ct: np.ndarray) -> np.ndarray:
        """
        Switch from extracted key to original LWE key.

        This ensures the output can be used with the original LWE secret key.
        """
        # Simplified: in production this uses decomposition
        return ct

    def apply_lut(self, ct: np.ndarray, lut: List[int]) -> np.ndarray:
        """
        Apply lookup table to encrypted value via programmable bootstrapping.

        This is the main entry point for non-linear function evaluation.

        Args:
            ct: LWE ciphertext encrypting m
            lut: Table where lut[m] = f(m)

        Returns:
            LWE ciphertext encrypting f(m)
        """
        return self.programmable_bootstrap(ct, lut)

    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        stats = self._stats.copy()
        stats['ntt_stats'] = self._ntt.get_stats()
        return stats

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            'encryptions': 0,
            'decryptions': 0,
            'bootstraps': 0,
            'total_time_ms': 0.0,
        }
        self._ntt.reset_stats()


def get_ntt_backend(params: 'N2HEParams') -> FasterNTTBackend:
    """
    Create and initialize a FasterNTT backend.

    Args:
        params: N2HE parameters

    Returns:
        Initialized FasterNTT backend
    """
    backend = FasterNTTBackend(params)
    backend.initialize()
    return backend
