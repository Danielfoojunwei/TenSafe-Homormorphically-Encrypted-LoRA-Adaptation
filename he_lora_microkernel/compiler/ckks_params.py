"""
CKKS Parameter Profiles for HE-LoRA Microkernel

This module defines the CKKS encryption parameters for the microkernel.
Two profiles are provided:
  - FAST: Speed-first, minimal depth
  - SAFE: Precision-first, extra headroom

NO BOOTSTRAPPING is supported - schedules must fit within the modulus chain.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import math


class CKKSProfile(Enum):
    """CKKS security/performance profile selector."""
    FAST = "fast"
    SAFE = "safe"
    TURBO = "turbo"  # For large batch/hidden configurations
    PRODUCTION = "production"  # N=32768 for production Llama-scale models


@dataclass(frozen=True)
class CKKSParams:
    """
    Immutable CKKS encryption parameters.

    These parameters define the encryption scheme's security, precision,
    and computational depth. Once set, they cannot be modified.
    """

    # Polynomial ring degree (N) - determines slot count (N/2)
    poly_modulus_degree: int

    # Coefficient modulus bit lengths - defines multiplicative depth
    coeff_modulus_bits: Tuple[int, ...]

    # Scale for encoding (2^scale_bits)
    scale_bits: int

    # Profile identifier
    profile: CKKSProfile

    # Maximum multiplicative depth before rescale exhaustion
    max_depth: int = field(init=False)

    # Number of SIMD slots (N/2)
    slot_count: int = field(init=False)

    # Actual scale value
    scale: float = field(init=False)

    def __post_init__(self):
        """Compute derived parameters."""
        # max_depth = number of intermediate primes (exclude first and last)
        object.__setattr__(self, 'max_depth', len(self.coeff_modulus_bits) - 2)
        object.__setattr__(self, 'slot_count', self.poly_modulus_degree // 2)
        object.__setattr__(self, 'scale', 2.0 ** self.scale_bits)

    def validate(self) -> None:
        """
        Validate CKKS parameters for correctness and security.

        Raises:
            ValueError: If parameters are invalid or insecure.
        """
        # Validate polynomial degree (must be power of 2)
        if self.poly_modulus_degree < 4096:
            raise ValueError(
                f"poly_modulus_degree={self.poly_modulus_degree} too small, "
                f"minimum 4096 for security"
            )
        if self.poly_modulus_degree & (self.poly_modulus_degree - 1) != 0:
            raise ValueError(
                f"poly_modulus_degree={self.poly_modulus_degree} must be power of 2"
            )

        # Validate coefficient modulus
        if len(self.coeff_modulus_bits) < 2:
            raise ValueError("Need at least 2 primes in coefficient modulus")

        total_bits = sum(self.coeff_modulus_bits)

        # Security bounds based on polynomial degree
        # These are conservative estimates based on HE standard
        max_bits_map = {
            4096: 109,
            8192: 218,
            16384: 438,
            32768: 881,
            65536: 1770,
        }

        max_allowed = max_bits_map.get(self.poly_modulus_degree, 218)
        if total_bits > max_allowed:
            raise ValueError(
                f"Total coeff modulus bits ({total_bits}) exceeds "
                f"security bound ({max_allowed}) for N={self.poly_modulus_degree}"
            )

        # Validate scale
        if self.scale_bits < 20 or self.scale_bits > 60:
            raise ValueError(
                f"scale_bits={self.scale_bits} out of valid range [20, 60]"
            )

        # Ensure scale fits in intermediate primes
        for i, bits in enumerate(self.coeff_modulus_bits[1:-1], start=1):
            if bits < self.scale_bits:
                raise ValueError(
                    f"Intermediate prime {i} ({bits} bits) smaller than "
                    f"scale ({self.scale_bits} bits) - rescale will fail"
                )

        # Validate depth is positive
        if self.max_depth < 1:
            raise ValueError(
                f"max_depth={self.max_depth} - need at least 1 for any computation"
            )

    def depth_for_lora(self, rank: int) -> int:
        """
        Compute required multiplicative depth for LoRA computation.

        LoRA: Δy = A(Bx)
        - 1 multiplication for Bx
        - 1 multiplication for A(Bx)
        - Possible additional for packing operations

        Args:
            rank: LoRA rank (r)

        Returns:
            Required multiplicative depth.
        """
        # Base: 2 multiplications for A @ B @ x
        base_depth = 2

        # Additional depth for rotation-based accumulation
        # For column packing with rotation reduction, we need log2(blocks)
        # This is a conservative estimate
        rotation_depth = 0  # MOAI-style packing avoids rotation depth

        return base_depth + rotation_depth

    def can_compute_lora(self, rank: int) -> bool:
        """Check if parameters support LoRA computation at given rank."""
        required = self.depth_for_lora(rank)
        return required <= self.max_depth

    def remaining_depth(self, consumed: int) -> int:
        """Return remaining multiplicative depth."""
        return max(0, self.max_depth - consumed)

    def to_dict(self) -> dict:
        """Serialize to dictionary for artifact emission."""
        return {
            'poly_modulus_degree': self.poly_modulus_degree,
            'coeff_modulus_bits': list(self.coeff_modulus_bits),
            'scale_bits': self.scale_bits,
            'profile': self.profile.value,
            'max_depth': self.max_depth,
            'slot_count': self.slot_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'CKKSParams':
        """Deserialize from dictionary."""
        return cls(
            poly_modulus_degree=d['poly_modulus_degree'],
            coeff_modulus_bits=tuple(d['coeff_modulus_bits']),
            scale_bits=d['scale_bits'],
            profile=CKKSProfile(d['profile']),
        )


# =============================================================================
# PROFILE DEFINITIONS
# =============================================================================

def get_fast_profile() -> CKKSParams:
    """
    FAST profile: Speed-first, minimal depth.

    - poly_modulus_degree = 16384 (8192 slots)
    - coeff_modulus_bits = [60, 40, 40, 60]
    - scale = 2^40
    - max_depth = 2

    Best for:
    - Low-rank LoRA (r <= 16)
    - Single-layer computations
    - Latency-sensitive workloads
    """
    return CKKSParams(
        poly_modulus_degree=16384,
        coeff_modulus_bits=(60, 40, 40, 60),
        scale_bits=40,
        profile=CKKSProfile.FAST,
    )


def get_safe_profile() -> CKKSParams:
    """
    SAFE profile: Precision-first, extra headroom.

    - poly_modulus_degree = 16384 (8192 slots)
    - coeff_modulus_bits = [60, 45, 45, 45, 60]
    - scale = 2^45
    - max_depth = 3

    Best for:
    - Higher precision requirements
    - Larger ranks (r up to 32)
    - When noise margin is critical
    """
    return CKKSParams(
        poly_modulus_degree=16384,
        coeff_modulus_bits=(60, 45, 45, 45, 60),
        scale_bits=45,
        profile=CKKSProfile.SAFE,
    )


def get_turbo_profile() -> CKKSParams:
    """
    TURBO profile: Maximum depth for large configurations.

    - poly_modulus_degree = 16384 (8192 slots)
    - coeff_modulus_bits = [60, 40, 40, 40, 40, 60]
    - scale = 2^40
    - max_depth = 4

    Best for:
    - Large batch sizes (b >= 8)
    - Large hidden dimensions (h >= 1024)
    - Configurations that exceed FAST profile depth
    """
    return CKKSParams(
        poly_modulus_degree=16384,
        coeff_modulus_bits=(60, 40, 40, 40, 40, 60),
        scale_bits=40,
        profile=CKKSProfile.TURBO,
    )


def get_production_profile() -> CKKSParams:
    """
    PRODUCTION profile: Optimized for Llama-scale models (h=4096+).

    - poly_modulus_degree = 32768 (16384 slots) - 2x more SIMD parallelism
    - coeff_modulus_bits = [60, 45, 45, 45, 45, 60] - balanced precision/depth
    - scale = 2^45
    - max_depth = 4

    Key benefits:
    - 16384 slots enables block_size=1024 (vs 512 with N=16384)
    - Fewer blocks for h=4096: 4 blocks instead of 8
    - Fewer rotations: log2(4)=2 instead of log2(8)=3
    - ~33% rotation reduction for production Llama 8B workloads

    Best for:
    - Production inference with Llama 8B/70B scale models
    - hidden_size >= 2048
    - Throughput-optimized deployments

    Trade-offs:
    - 2x more memory per ciphertext
    - Slightly higher per-operation latency
    - Amortized by fewer total operations

    Security: 128-bit (881 bits modulus budget at N=32768)
    """
    return CKKSParams(
        poly_modulus_degree=32768,
        coeff_modulus_bits=(60, 45, 45, 45, 45, 60),
        scale_bits=45,
        profile=CKKSProfile.PRODUCTION,
    )


def get_profile(profile: CKKSProfile) -> CKKSParams:
    """Get CKKS parameters for specified profile."""
    if profile == CKKSProfile.FAST:
        return get_fast_profile()
    elif profile == CKKSProfile.SAFE:
        return get_safe_profile()
    elif profile == CKKSProfile.TURBO:
        return get_turbo_profile()
    elif profile == CKKSProfile.PRODUCTION:
        return get_production_profile()
    else:
        raise ValueError(f"Unknown profile: {profile}")


def select_optimal_profile(
    hidden_size: int,
    lora_rank: int,
    batch_size: int,
    precision_requirement: float = 1e-2,
    prefer_production: bool = True,
) -> CKKSParams:
    """
    Automatically select the best CKKS profile for given parameters.

    Args:
        hidden_size: Model hidden dimension
        lora_rank: LoRA rank (r)
        batch_size: Batch size
        precision_requirement: Maximum acceptable relative error
        prefer_production: Use PRODUCTION profile for large models (h>=2048)

    Returns:
        Optimal CKKSParams for the workload.

    Profile Selection Logic:
    1. PRODUCTION (N=32768): h >= 2048, optimizes rotation count for Llama-scale
    2. TURBO (N=16384): h >= 1024 with b >= 8, or h >= 512 with b >= 16
    3. SAFE (N=16384): Higher precision requirements
    4. FAST (N=16384): Default for smaller workloads
    """
    # For Llama-scale models (h >= 2048), use PRODUCTION profile
    # This provides 2x more slots, enabling larger blocks and fewer rotations
    if prefer_production and hidden_size >= 2048:
        params = get_production_profile()
    # For large configurations, use TURBO profile
    # This handles h>=1024 with b>=8 which exceeds FAST depth
    elif hidden_size >= 1024 and batch_size >= 8:
        params = get_turbo_profile()
    elif hidden_size >= 512 and batch_size >= 16:
        params = get_turbo_profile()
    else:
        # Start with FAST profile
        params = get_fast_profile()

    # Check if current profile can handle the workload
    if not params.can_compute_lora(lora_rank):
        if params.profile == CKKSProfile.FAST:
            params = get_safe_profile()
        elif params.profile == CKKSProfile.SAFE:
            params = get_turbo_profile()
        elif params.profile == CKKSProfile.TURBO:
            params = get_production_profile()

    # Check slot requirements
    required_slots = batch_size * max(hidden_size // 512, 1)  # Block packing
    if required_slots > params.slot_count:
        # Try upgrading to PRODUCTION profile for more slots
        if params.profile != CKKSProfile.PRODUCTION:
            params = get_production_profile()
            if required_slots > params.slot_count:
                raise ValueError(
                    f"Workload requires {required_slots} slots but "
                    f"max available is {params.slot_count}. "
                    f"Reduce batch_size or hidden_size."
                )
        else:
            raise ValueError(
                f"Workload requires {required_slots} slots but "
                f"profile only has {params.slot_count}. "
                f"Reduce batch_size or hidden_size."
            )

    # Upgrade to SAFE if precision is critical (but not if using PRODUCTION)
    if precision_requirement < 5e-3 and params.profile == CKKSProfile.FAST:
        params = get_safe_profile()

    params.validate()
    return params


# =============================================================================
# PARAMETER VERIFICATION
# =============================================================================

@dataclass
class ScheduleCompatibility:
    """Result of schedule compatibility check."""
    compatible: bool
    required_depth: int
    available_depth: int
    required_slots: int
    available_slots: int
    error_message: Optional[str] = None


def verify_schedule_fits(
    params: CKKSParams,
    hidden_size: int,
    lora_rank: int,
    batch_size: int,
    block_size: int = 512,
) -> ScheduleCompatibility:
    """
    Verify that a compiled schedule fits within CKKS parameters.

    This is called by the compiler to reject builds that would
    require bootstrapping (which is NOT supported).

    Args:
        params: CKKS parameters
        hidden_size: Model hidden dimension
        lora_rank: LoRA rank
        batch_size: Batch size
        block_size: Block size for blocked packing

    Returns:
        ScheduleCompatibility result.
    """
    # Calculate required depth
    required_depth = params.depth_for_lora(lora_rank)

    # Calculate required slots
    num_blocks = math.ceil(hidden_size / block_size)
    required_slots = batch_size * block_size  # Slots per block

    # Check compatibility
    depth_ok = required_depth <= params.max_depth
    slots_ok = required_slots <= params.slot_count

    if depth_ok and slots_ok:
        return ScheduleCompatibility(
            compatible=True,
            required_depth=required_depth,
            available_depth=params.max_depth,
            required_slots=required_slots,
            available_slots=params.slot_count,
        )

    error_parts = []
    if not depth_ok:
        error_parts.append(
            f"Depth overflow: need {required_depth}, have {params.max_depth}. "
            f"Bootstrapping NOT supported - reduce lora_rank or use SAFE profile."
        )
    if not slots_ok:
        error_parts.append(
            f"Slot overflow: need {required_slots}, have {params.slot_count}. "
            f"Reduce batch_size or block_size."
        )

    return ScheduleCompatibility(
        compatible=False,
        required_depth=required_depth,
        available_depth=params.max_depth,
        required_slots=required_slots,
        available_slots=params.slot_count,
        error_message=" ".join(error_parts),
    )
