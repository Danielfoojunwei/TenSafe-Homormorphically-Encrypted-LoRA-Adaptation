"""
N2HE/TFHE Parameter Profiles for HE-LoRA Microkernel

This module defines parameters for TFHE-style programmable bootstrapping,
which provides EXACT computation on discrete plaintexts (not approximate).

Key Distinction from CKKS:
    - CKKS: Approximate arithmetic on real numbers (inherent precision loss)
    - TFHE: EXACT arithmetic on discrete message space (with overwhelming probability)

TFHE Programmable Bootstrapping:
    - Operates on discrete messages (bits / small integers on the torus T = R/Z)
    - Bootstrapping = noise refresh + LUT evaluation in ONE operation
    - Correctness via rounding to nearest valid message during decryption
    - Parameters designed to ensure correctness with overwhelming probability

Message Space:
    - Torus T = R/Z (real numbers modulo 1)
    - Messages encoded as discrete points: m/p for m in {0, 1, ..., p-1}
    - Decryption rounds to nearest valid message point

References:
    - TFHE: Fast Fully Homomorphic Encryption over the Torus
    - N2HE: https://github.com/HintSight-Technology/N2HE
    - Programmable Bootstrapping surveys
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import math


class N2HEProfile(Enum):
    """N2HE/TFHE security/performance profile selector."""
    FAST = "fast"          # Speed-first, 80-bit security
    BALANCED = "balanced"  # 128-bit security, good performance
    SECURE = "secure"      # 192-bit security, production


class MessageSpace(Enum):
    """Discrete message space for TFHE operations."""
    BINARY = 2        # {0, 1} - for boolean circuits
    TERNARY = 3       # {-1, 0, 1} - for ternary networks
    SMALL_INT_4 = 16  # 4-bit integers {0, ..., 15}
    SMALL_INT_8 = 256 # 8-bit integers {0, ..., 255}


@dataclass(frozen=True)
class LWEParams:
    """
    LWE (Learning With Errors) parameters for TFHE.

    LWE encrypts a torus element: c = (a, b) where b = <a, s> + m + e
    - s: secret key (binary or ternary)
    - m: message encoded on torus (m/p for discrete message space p)
    - e: Gaussian noise with standard deviation σ

    Correctness: Decryption succeeds when |e| < 1/(2p)
    """
    # LWE dimension (secret key length)
    dimension: int

    # Noise standard deviation (as fraction of torus)
    noise_stddev: float

    # Message space cardinality
    message_space: int

    # Key distribution: "binary" (0/1) or "ternary" (-1/0/1)
    key_distribution: str = "binary"

    def __post_init__(self):
        """Validate parameters for correctness."""
        if self.dimension < 256:
            raise ValueError(f"LWE dimension {self.dimension} too small for security")

        # Correctness check: noise must be small enough for decryption
        # Decryption fails if noise exceeds 1/(2p) where p = message_space
        max_noise_for_correctness = 1.0 / (2 * self.message_space)
        # Allow 6σ margin for overwhelming probability
        if 6 * self.noise_stddev > max_noise_for_correctness:
            raise ValueError(
                f"Noise σ={self.noise_stddev} too large for message space {self.message_space}. "
                f"Need 6σ < 1/(2p) = {max_noise_for_correctness:.6f}"
            )

    @property
    def security_bits(self) -> int:
        """Estimate security level using LWE estimator heuristics."""
        n = self.dimension
        sigma = self.noise_stddev
        # Conservative estimate based on lattice estimator
        # Security ≈ n * log2(1/σ) for binary secrets
        if self.key_distribution == "binary":
            return max(0, int(n * math.log2(1 / sigma) * 0.265))
        else:  # ternary
            return max(0, int(n * math.log2(1 / sigma) * 0.292))

    @property
    def correctness_probability(self) -> float:
        """Probability that decryption is correct (single operation)."""
        # P(|e| < 1/(2p)) where e ~ N(0, σ²)
        threshold = 1.0 / (2 * self.message_space)
        # Using error function: P = erf(threshold / (σ * sqrt(2)))
        import math
        return math.erf(threshold / (self.noise_stddev * math.sqrt(2)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dimension': self.dimension,
            'noise_stddev': self.noise_stddev,
            'message_space': self.message_space,
            'key_distribution': self.key_distribution,
            'security_bits': self.security_bits,
            'correctness_probability': self.correctness_probability,
        }


@dataclass(frozen=True)
class RLWEParams:
    """
    RLWE (Ring LWE) parameters for bootstrapping accumulator.

    RLWE operates over the ring Z_Q[X]/(X^N + 1):
    - N: ring dimension (power of 2)
    - Q: coefficient modulus

    Used for:
    - Bootstrapping key (RGSW encryptions of LWE secret key bits)
    - Accumulator during programmable bootstrapping
    """
    # Ring dimension (must be power of 2)
    ring_dimension: int

    # Coefficient modulus
    coeff_modulus: int

    # Noise standard deviation
    noise_stddev: float

    # Decomposition base for RGSW (controls noise vs key size tradeoff)
    decomposition_base: int = 1024  # B_g

    # Number of decomposition levels
    decomposition_levels: int = 3   # ℓ

    def __post_init__(self):
        """Validate parameters."""
        if self.ring_dimension & (self.ring_dimension - 1) != 0:
            raise ValueError(f"Ring dimension {self.ring_dimension} must be power of 2")
        if self.ring_dimension < 512:
            raise ValueError(f"Ring dimension {self.ring_dimension} too small")

    @property
    def slot_count(self) -> int:
        """Number of coefficients (= ring dimension for negacyclic)."""
        return self.ring_dimension

    @property
    def rgsw_key_size(self) -> int:
        """Size of one RGSW ciphertext in ring elements."""
        return 2 * (self.decomposition_levels + 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ring_dimension': self.ring_dimension,
            'coeff_modulus': self.coeff_modulus,
            'noise_stddev': self.noise_stddev,
            'decomposition_base': self.decomposition_base,
            'decomposition_levels': self.decomposition_levels,
            'slot_count': self.slot_count,
        }


@dataclass(frozen=True)
class BootstrappingParams:
    """
    Parameters specific to programmable bootstrapping.

    Programmable bootstrapping in TFHE:
    1. Modulus switch LWE ciphertext to smaller modulus (2N)
    2. Initialize accumulator with test polynomial encoding LUT
    3. Blind rotate accumulator using RGSW bootstrapping keys
    4. Extract refreshed LWE ciphertext with LUT applied

    The LUT is encoded in the test polynomial coefficients.
    """
    # Test polynomial modulus (usually 2 * ring_dimension)
    test_polynomial_size: int

    # LUT precision (number of distinct output values)
    lut_output_precision: int

    # Whether to use keyswitch after extraction
    use_keyswitch: bool = True

    # Keyswitch decomposition base
    keyswitch_base: int = 32

    # Keyswitch decomposition levels
    keyswitch_levels: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_polynomial_size': self.test_polynomial_size,
            'lut_output_precision': self.lut_output_precision,
            'use_keyswitch': self.use_keyswitch,
            'keyswitch_base': self.keyswitch_base,
            'keyswitch_levels': self.keyswitch_levels,
        }


@dataclass(frozen=True)
class N2HEParams:
    """
    Complete N2HE/TFHE parameter set for programmable bootstrapping.

    This combines:
    - LWE parameters for input/output ciphertexts
    - RLWE parameters for bootstrapping accumulator
    - Bootstrapping-specific parameters

    Key Property: EXACTNESS
        For discrete message space p, decryption is EXACT (not approximate)
        with overwhelming probability when parameters are correctly chosen.
    """
    # LWE parameters for encrypted data
    lwe: LWEParams

    # RLWE parameters for bootstrapping
    rlwe: RLWEParams

    # Bootstrapping configuration
    bootstrapping: BootstrappingParams

    # Profile identifier
    profile: N2HEProfile

    # Whether to use GPU acceleration
    use_gpu: bool = True

    def validate(self) -> None:
        """Validate complete parameter set for correctness."""
        # Check that test polynomial size matches ring dimension
        expected_test_size = 2 * self.rlwe.ring_dimension
        if self.bootstrapping.test_polynomial_size != expected_test_size:
            raise ValueError(
                f"Test polynomial size ({self.bootstrapping.test_polynomial_size}) "
                f"should be 2N = {expected_test_size}"
            )

        # Check LUT precision fits in message space
        if self.bootstrapping.lut_output_precision > self.lwe.message_space:
            raise ValueError(
                f"LUT output precision ({self.bootstrapping.lut_output_precision}) "
                f"exceeds message space ({self.lwe.message_space})"
            )

        # Verify noise growth during bootstrapping stays bounded
        # This is a simplified check; real analysis requires noise estimator
        rgsw_noise_factor = self.rlwe.decomposition_base ** self.rlwe.decomposition_levels
        if rgsw_noise_factor > self.rlwe.coeff_modulus / 1000:
            raise ValueError(
                f"RGSW decomposition may cause excessive noise growth"
            )

    @property
    def message_space_size(self) -> int:
        """Cardinality of discrete message space."""
        return self.lwe.message_space

    @property
    def is_binary(self) -> bool:
        """Check if operating on binary messages."""
        return self.lwe.message_space == 2

    @property
    def estimated_bootstrapping_time_ms(self) -> float:
        """Rough estimate of bootstrapping time (GPU, single ciphertext)."""
        # Based on typical TFHE/concrete benchmarks
        n = self.lwe.dimension
        N = self.rlwe.ring_dimension
        # Approximately O(n * log N) NTT operations
        base_time = 0.01  # 10μs per NTT on GPU
        return base_time * n * math.log2(N)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'profile': self.profile.value,
            'lwe': self.lwe.to_dict(),
            'rlwe': self.rlwe.to_dict(),
            'bootstrapping': self.bootstrapping.to_dict(),
            'use_gpu': self.use_gpu,
            'message_space_size': self.message_space_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'N2HEParams':
        """Deserialize from dictionary."""
        return cls(
            lwe=LWEParams(
                dimension=d['lwe']['dimension'],
                noise_stddev=d['lwe']['noise_stddev'],
                message_space=d['lwe']['message_space'],
                key_distribution=d['lwe'].get('key_distribution', 'binary'),
            ),
            rlwe=RLWEParams(
                ring_dimension=d['rlwe']['ring_dimension'],
                coeff_modulus=d['rlwe']['coeff_modulus'],
                noise_stddev=d['rlwe']['noise_stddev'],
                decomposition_base=d['rlwe'].get('decomposition_base', 1024),
                decomposition_levels=d['rlwe'].get('decomposition_levels', 3),
            ),
            bootstrapping=BootstrappingParams(
                test_polynomial_size=d['bootstrapping']['test_polynomial_size'],
                lut_output_precision=d['bootstrapping']['lut_output_precision'],
                use_keyswitch=d['bootstrapping'].get('use_keyswitch', True),
                keyswitch_base=d['bootstrapping'].get('keyswitch_base', 32),
                keyswitch_levels=d['bootstrapping'].get('keyswitch_levels', 4),
            ),
            profile=N2HEProfile(d['profile']),
            use_gpu=d.get('use_gpu', True),
        )


# =============================================================================
# TFHE PROFILE DEFINITIONS
# =============================================================================

def get_fast_n2he_profile() -> N2HEParams:
    """
    FAST profile: ~80-bit security, optimized for speed.

    Message space: 16 values (4-bit integers)
    Best for: Development, testing, latency-critical applications

    Correctness: >1 - 2^{-40} probability per bootstrapping
    """
    # LWE: n=630 for ~80-bit security
    # σ chosen for correctness with p=16: need 6σ < 1/32
    lwe = LWEParams(
        dimension=630,
        noise_stddev=2**(-15),  # ≈ 3e-5, well below 1/32 ≈ 0.031
        message_space=16,
        key_distribution="binary",
    )

    # RLWE: N=1024, Q=2^32
    rlwe = RLWEParams(
        ring_dimension=1024,
        coeff_modulus=2**32,
        noise_stddev=2**(-25),
        decomposition_base=128,
        decomposition_levels=3,
    )

    bootstrapping = BootstrappingParams(
        test_polynomial_size=2048,  # 2N
        lut_output_precision=16,
        use_keyswitch=True,
        keyswitch_base=32,
        keyswitch_levels=3,
    )

    return N2HEParams(
        lwe=lwe,
        rlwe=rlwe,
        bootstrapping=bootstrapping,
        profile=N2HEProfile.FAST,
        use_gpu=True,
    )


def get_balanced_n2he_profile() -> N2HEParams:
    """
    BALANCED profile: 128-bit security with good performance.

    Message space: 256 values (8-bit integers)
    Best for: Production workloads, standard security requirements

    Correctness: >1 - 2^{-64} probability per bootstrapping
    """
    # LWE: n=1024 for ~128-bit security
    # σ chosen for correctness with p=256: need 6σ < 1/512
    lwe = LWEParams(
        dimension=1024,
        noise_stddev=2**(-20),  # ≈ 9.5e-7, well below 1/512 ≈ 0.002
        message_space=256,
        key_distribution="binary",
    )

    # RLWE: N=2048, Q=2^64
    rlwe = RLWEParams(
        ring_dimension=2048,
        coeff_modulus=2**64,
        noise_stddev=2**(-35),
        decomposition_base=256,
        decomposition_levels=4,
    )

    bootstrapping = BootstrappingParams(
        test_polynomial_size=4096,  # 2N
        lut_output_precision=256,
        use_keyswitch=True,
        keyswitch_base=64,
        keyswitch_levels=4,
    )

    return N2HEParams(
        lwe=lwe,
        rlwe=rlwe,
        bootstrapping=bootstrapping,
        profile=N2HEProfile.BALANCED,
        use_gpu=True,
    )


def get_secure_n2he_profile() -> N2HEParams:
    """
    SECURE profile: 192-bit security for high-assurance applications.

    Message space: 256 values (8-bit integers)
    Best for: Compliance-critical, sensitive data, long-term security

    Correctness: >1 - 2^{-80} probability per bootstrapping
    """
    # LWE: n=2048 for ~192-bit security
    lwe = LWEParams(
        dimension=2048,
        noise_stddev=2**(-25),
        message_space=256,
        key_distribution="binary",
    )

    # RLWE: N=4096, Q=2^128 (multi-limb)
    rlwe = RLWEParams(
        ring_dimension=4096,
        coeff_modulus=2**128,
        noise_stddev=2**(-45),
        decomposition_base=512,
        decomposition_levels=5,
    )

    bootstrapping = BootstrappingParams(
        test_polynomial_size=8192,  # 2N
        lut_output_precision=256,
        use_keyswitch=True,
        keyswitch_base=64,
        keyswitch_levels=5,
    )

    return N2HEParams(
        lwe=lwe,
        rlwe=rlwe,
        bootstrapping=bootstrapping,
        profile=N2HEProfile.SECURE,
        use_gpu=True,
    )


def get_n2he_profile(profile: N2HEProfile) -> N2HEParams:
    """Get N2HE/TFHE parameters for specified profile."""
    profiles = {
        N2HEProfile.FAST: get_fast_n2he_profile,
        N2HEProfile.BALANCED: get_balanced_n2he_profile,
        N2HEProfile.SECURE: get_secure_n2he_profile,
    }
    if profile not in profiles:
        raise ValueError(f"Unknown N2HE profile: {profile}")

    params = profiles[profile]()
    params.validate()
    return params


def select_optimal_n2he_profile(
    activation_precision_bits: int = 8,
    security_requirement: int = 128,
    latency_critical: bool = False,
) -> N2HEParams:
    """
    Automatically select the best N2HE profile for given requirements.

    Args:
        activation_precision_bits: Bits of precision for activation values
        security_requirement: Minimum security bits required
        latency_critical: If True, prioritize speed over security margin

    Returns:
        Optimal N2HEParams for the workload.
    """
    # Determine required message space
    required_message_space = 2 ** activation_precision_bits

    # Select profile based on security and latency requirements
    if latency_critical and security_requirement < 100:
        params = get_fast_n2he_profile()
    elif security_requirement >= 192:
        params = get_secure_n2he_profile()
    else:
        params = get_balanced_n2he_profile()

    # Verify message space is sufficient
    if params.message_space_size < required_message_space:
        # Upgrade to profile with larger message space
        if params.profile == N2HEProfile.FAST:
            params = get_balanced_n2he_profile()

    params.validate()
    return params


# =============================================================================
# CKKS-TFHE BRIDGE FOR HYBRID COMPUTATION
# =============================================================================

@dataclass
class CKKSToTFHEBridge:
    """
    Bridge parameters for hybrid CKKS + TFHE computation.

    Architecture:
        CKKS (MOAI) for linear operations → Fast, rotation-free Ct×Pt
        TFHE for non-linear activations → Exact LUT evaluation

    Data Flow:
        1. CKKS ciphertext from MOAI matmul
        2. Decrypt to plaintext (client-side for privacy)
        3. Quantize real values to discrete message space
        4. Encrypt with LWE (TFHE)
        5. Programmable bootstrap with activation LUT
        6. Decrypt LWE result
        7. Re-scale and re-encrypt with CKKS
        8. Continue with next MOAI layer

    Note: Steps 2-7 happen on client for true privacy preservation.
    For semi-honest model, server can perform quantization on encrypted data
    using comparison circuits (expensive).
    """
    # CKKS scale for import/export
    ckks_scale_bits: int

    # TFHE parameters
    tfhe_params: N2HEParams

    # Quantization parameters
    input_min: float   # Expected minimum activation value
    input_max: float   # Expected maximum activation value

    def quantize_to_message_space(self, value: float) -> int:
        """
        Quantize a real value to discrete message space.

        Maps [input_min, input_max] → {0, 1, ..., p-1}
        """
        p = self.tfhe_params.message_space_size
        # Clamp to valid range
        clamped = max(self.input_min, min(self.input_max, value))
        # Normalize to [0, 1]
        normalized = (clamped - self.input_min) / (self.input_max - self.input_min)
        # Map to discrete space
        discrete = int(round(normalized * (p - 1)))
        return max(0, min(p - 1, discrete))

    def dequantize_from_message_space(self, message: int) -> float:
        """
        Dequantize a discrete message to real value.

        Maps {0, 1, ..., p-1} → [input_min, input_max]
        """
        p = self.tfhe_params.message_space_size
        normalized = message / (p - 1)
        return self.input_min + normalized * (self.input_max - self.input_min)

    def quantize_lut(self, func, output_min: float, output_max: float) -> List[int]:
        """
        Create quantized LUT for an activation function.

        Args:
            func: Activation function f: R → R (e.g., relu, gelu)
            output_min: Minimum expected output value
            output_max: Maximum expected output value

        Returns:
            List of p integers representing quantized LUT entries.
        """
        p = self.tfhe_params.message_space_size
        lut = []

        for i in range(p):
            # Dequantize input
            x = self.dequantize_from_message_space(i)
            # Apply function
            y = func(x)
            # Quantize output (using output range)
            y_clamped = max(output_min, min(output_max, y))
            y_normalized = (y_clamped - output_min) / (output_max - output_min)
            y_discrete = int(round(y_normalized * (p - 1)))
            lut.append(max(0, min(p - 1, y_discrete)))

        return lut


def create_ckks_tfhe_bridge(
    ckks_scale_bits: int = 40,
    tfhe_profile: N2HEProfile = N2HEProfile.BALANCED,
    activation_range: Tuple[float, float] = (-10.0, 10.0),
) -> CKKSToTFHEBridge:
    """
    Create bridge for hybrid CKKS+TFHE computation.

    Args:
        ckks_scale_bits: CKKS scale (2^scale_bits)
        tfhe_profile: TFHE profile to use for activations
        activation_range: Expected (min, max) range of activation values

    Returns:
        Configured bridge parameters.
    """
    tfhe_params = get_n2he_profile(tfhe_profile)

    return CKKSToTFHEBridge(
        ckks_scale_bits=ckks_scale_bits,
        tfhe_params=tfhe_params,
        input_min=activation_range[0],
        input_max=activation_range[1],
    )
