"""
IR Type System for Hybrid CKKS-TFHE Compiler

Defines the type system for scheme-aware IR values including:
- Scheme domains (CKKS vs TFHE)
- Value types (approximate reals vs discrete integers)
- Shape representations
- Scheme-specific metadata (precision, noise budgets)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, Union, List, Dict, Any
import math


class Scheme(Enum):
    """
    Encryption scheme domain.

    CKKS: Approximate arithmetic on real numbers
        - Used for linear algebra (MatMul, Add, Mul)
        - No conditionals, no activations
        - MOAI column packing enabled

    TFHE: Exact arithmetic on discrete values
        - Used for control flow (gates, thresholds)
        - Programmable bootstrapping
        - Limited to scalars / tiny vectors
    """
    CKKS = auto()
    TFHE = auto()

    def is_linear_only(self) -> bool:
        """CKKS is restricted to linear operations."""
        return self == Scheme.CKKS

    def supports_lut(self) -> bool:
        """Only TFHE supports lookup tables."""
        return self == Scheme.TFHE


class ValueType(Enum):
    """
    Value type for encrypted data.

    REAL_APPROX: Approximate real numbers (CKKS only)
        - Subject to precision loss from rescaling
        - Precision tracked via precision_budget

    BIT: Single bit {0, 1} (TFHE only)
        - Exact computation
        - Result of comparisons and thresholds

    INT_K: K-bit integer (TFHE only)
        - Exact computation on discrete values
        - K typically 4-12 bits for LUT compatibility
    """
    REAL_APPROX = "real_approx"
    BIT = "bit"
    INT_4 = "int_4"
    INT_8 = "int_8"
    INT_12 = "int_12"
    INT_16 = "int_16"

    @property
    def bits(self) -> int:
        """Number of bits for integer types."""
        bits_map = {
            ValueType.REAL_APPROX: 0,  # Not applicable
            ValueType.BIT: 1,
            ValueType.INT_4: 4,
            ValueType.INT_8: 8,
            ValueType.INT_12: 12,
            ValueType.INT_16: 16,
        }
        return bits_map[self]

    @property
    def bit_width(self) -> int:
        """Alias for bits property."""
        return self.bits

    @property
    def is_ckks_compatible(self) -> bool:
        """Check if type is compatible with CKKS scheme."""
        return self == ValueType.REAL_APPROX

    @property
    def is_tfhe_compatible(self) -> bool:
        """Check if type is compatible with TFHE scheme."""
        return self != ValueType.REAL_APPROX

    @property
    def is_discrete(self) -> bool:
        """Check if type is discrete (exact arithmetic)."""
        return self != ValueType.REAL_APPROX

    @property
    def message_space_size(self) -> int:
        """Size of discrete message space."""
        if self == ValueType.REAL_APPROX:
            raise ValueError("REAL_APPROX has no discrete message space")
        return 2 ** self.bits

    def compatible_scheme(self) -> Scheme:
        """Get the compatible scheme for this value type."""
        if self == ValueType.REAL_APPROX:
            return Scheme.CKKS
        return Scheme.TFHE


@dataclass(frozen=True)
class Shape:
    """
    Shape of an IR value.

    Enforces constraints:
    - TFHE values must be scalar or very small vectors (<=16)
    - CKKS values can be arbitrary size (within slot limits)
    """
    dims: Tuple[int, ...]

    # Maximum elements allowed for TFHE operations
    MAX_TFHE_ELEMENTS = 16

    @property
    def is_scalar(self) -> bool:
        """Check if shape is scalar."""
        return len(self.dims) == 0 or self.dims == (1,)

    @property
    def numel(self) -> int:
        """Total number of elements."""
        if len(self.dims) == 0:
            return 1
        result = 1
        for d in self.dims:
            result *= d
        return result

    @property
    def size(self) -> int:
        """Alias for numel - total number of elements."""
        return self.numel

    @property
    def is_tfhe_compatible(self) -> bool:
        """Check if shape is valid for TFHE operations."""
        return self.numel <= self.MAX_TFHE_ELEMENTS

    def validate_for_scheme(self, scheme: Scheme) -> None:
        """Validate shape is compatible with scheme."""
        if scheme == Scheme.TFHE and not self.is_tfhe_compatible:
            raise ValueError(
                f"TFHE operations limited to <={self.MAX_TFHE_ELEMENTS} elements, "
                f"got shape {self.dims} with {self.numel} elements"
            )

    @classmethod
    def scalar(cls) -> 'Shape':
        """Create scalar shape."""
        return cls(dims=())

    @classmethod
    def vector(cls, n: int) -> 'Shape':
        """Create vector shape."""
        return cls(dims=(n,))

    @classmethod
    def matrix(cls, rows: int, cols: int) -> 'Shape':
        """Create matrix shape."""
        return cls(dims=(rows, cols))

    def __str__(self) -> str:
        if self.is_scalar:
            return "scalar"
        return f"[{', '.join(map(str, self.dims))}]"


@dataclass
class CKKSMetadata:
    """
    CKKS-specific metadata for IR values.

    Tracks:
    - Precision budget (bits remaining before noise overwhelms signal)
    - Scale (encoding scale factor)
    - Level (position in modulus chain)
    - MOAI packing status
    """
    # Precision budget in bits (approximate SNR)
    precision_budget: float

    # Current scale (2^scale_bits)
    scale: float

    # Level in modulus chain (0 = fresh, increases with rescales)
    level: int = 0

    # Whether value is in MOAI column-packed format
    is_moai_packed: bool = False

    # Original dimensions before packing
    original_shape: Optional[Tuple[int, ...]] = None

    def after_multiply(self) -> 'CKKSMetadata':
        """Compute metadata after multiplication (before rescale)."""
        return CKKSMetadata(
            precision_budget=self.precision_budget - 1,  # Lose ~1 bit per mult
            scale=self.scale * self.scale,
            level=self.level,
            is_moai_packed=self.is_moai_packed,
            original_shape=self.original_shape,
        )

    def after_rescale(self, scale_bits: int) -> 'CKKSMetadata':
        """Compute metadata after rescaling."""
        return CKKSMetadata(
            precision_budget=self.precision_budget,
            scale=2.0 ** scale_bits,
            level=self.level + 1,
            is_moai_packed=self.is_moai_packed,
            original_shape=self.original_shape,
        )

    def after_add(self, other: 'CKKSMetadata') -> 'CKKSMetadata':
        """Compute metadata after addition."""
        return CKKSMetadata(
            precision_budget=min(self.precision_budget, other.precision_budget),
            scale=self.scale,  # Scales must match for add
            level=max(self.level, other.level),
            is_moai_packed=self.is_moai_packed and other.is_moai_packed,
            original_shape=self.original_shape,
        )


@dataclass
class TFHEMetadata:
    """
    TFHE-specific metadata for IR values.

    Tracks:
    - Noise budget (margin before decryption failure)
    - Bootstrap cost estimate
    - LUT complexity
    """
    # Noise budget estimate (bits of margin)
    noise_budget: float

    # Estimated bootstrap cost (milliseconds)
    bootstrap_cost_ms: float = 0.0

    # Whether value has been bootstrapped recently
    is_fresh: bool = True

    # LUT that was applied (if any)
    applied_lut: Optional[str] = None

    def after_bootstrap(self, lut_name: Optional[str] = None) -> 'TFHEMetadata':
        """Compute metadata after programmable bootstrapping."""
        return TFHEMetadata(
            noise_budget=128.0,  # Full refresh
            bootstrap_cost_ms=self.bootstrap_cost_ms + 10.0,  # ~10ms per bootstrap
            is_fresh=True,
            applied_lut=lut_name,
        )

    def after_operation(self, noise_growth: float = 1.0) -> 'TFHEMetadata':
        """Compute metadata after a non-bootstrapping operation."""
        return TFHEMetadata(
            noise_budget=self.noise_budget - noise_growth,
            bootstrap_cost_ms=self.bootstrap_cost_ms,
            is_fresh=False,
            applied_lut=self.applied_lut,
        )


@dataclass
class IRValue:
    """
    IR value representation with scheme-aware typing.

    Every value in the IR carries full type information including:
    - Scheme domain (CKKS or TFHE)
    - Value type (approximate real or discrete)
    - Shape
    - Scheme-specific metadata
    """
    # Unique identifier
    name: str

    # Encryption scheme
    scheme: Scheme

    # Value type
    value_type: ValueType

    # Shape
    shape: Shape

    # Scheme-specific metadata
    ckks_meta: Optional[CKKSMetadata] = None
    tfhe_meta: Optional[TFHEMetadata] = None

    def __post_init__(self):
        """Validate value consistency."""
        self._validate()

    def _validate(self):
        """Ensure value is internally consistent."""
        # Check scheme/type compatibility
        if self.scheme == Scheme.CKKS and self.value_type != ValueType.REAL_APPROX:
            raise ValueError(
                f"CKKS scheme requires REAL_APPROX type, got {self.value_type}"
            )
        if self.scheme == Scheme.TFHE and self.value_type == ValueType.REAL_APPROX:
            raise ValueError(
                f"TFHE scheme requires discrete type, got {self.value_type}"
            )

        # Check shape constraints
        self.shape.validate_for_scheme(self.scheme)

        # Check metadata presence
        if self.scheme == Scheme.CKKS and self.ckks_meta is None:
            raise ValueError("CKKS values require ckks_meta")
        if self.scheme == Scheme.TFHE and self.tfhe_meta is None:
            raise ValueError("TFHE values require tfhe_meta")

    @property
    def precision_budget(self) -> float:
        """Get precision budget (CKKS only)."""
        if self.scheme != Scheme.CKKS:
            raise ValueError("precision_budget only defined for CKKS")
        return self.ckks_meta.precision_budget

    @property
    def noise_budget(self) -> float:
        """Get noise budget (TFHE only)."""
        if self.scheme != Scheme.TFHE:
            raise ValueError("noise_budget only defined for TFHE")
        return self.tfhe_meta.noise_budget

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            'name': self.name,
            'scheme': self.scheme.name,
            'value_type': self.value_type.value,
            'shape': str(self.shape),
        }
        if self.ckks_meta:
            result['ckks_meta'] = {
                'precision_budget': self.ckks_meta.precision_budget,
                'scale': self.ckks_meta.scale,
                'level': self.ckks_meta.level,
                'is_moai_packed': self.ckks_meta.is_moai_packed,
            }
        if self.tfhe_meta:
            result['tfhe_meta'] = {
                'noise_budget': self.tfhe_meta.noise_budget,
                'bootstrap_cost_ms': self.tfhe_meta.bootstrap_cost_ms,
                'is_fresh': self.tfhe_meta.is_fresh,
            }
        return result

    def __str__(self) -> str:
        meta = ""
        if self.ckks_meta:
            meta = f", prec={self.ckks_meta.precision_budget:.1f}b"
        if self.tfhe_meta:
            meta = f", noise={self.tfhe_meta.noise_budget:.1f}b"
        return f"{self.name}: {self.scheme.name}::{self.value_type.value}{self.shape}{meta}"


def create_ckks_value(
    name: str,
    shape: Shape,
    precision_budget: float = 40.0,
    scale_bits: int = 40,
    is_moai_packed: bool = False,
) -> IRValue:
    """Helper to create a CKKS IR value."""
    return IRValue(
        name=name,
        scheme=Scheme.CKKS,
        value_type=ValueType.REAL_APPROX,
        shape=shape,
        ckks_meta=CKKSMetadata(
            precision_budget=precision_budget,
            scale=2.0 ** scale_bits,
            is_moai_packed=is_moai_packed,
        ),
    )


def create_tfhe_value(
    name: str,
    shape: Shape,
    value_type: ValueType = ValueType.BIT,
    noise_budget: float = 128.0,
) -> IRValue:
    """Helper to create a TFHE IR value."""
    if value_type == ValueType.REAL_APPROX:
        raise ValueError("TFHE requires discrete value type")
    return IRValue(
        name=name,
        scheme=Scheme.TFHE,
        value_type=value_type,
        shape=shape,
        tfhe_meta=TFHEMetadata(noise_budget=noise_budget),
    )
