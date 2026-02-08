"""
TFHE Lookup Table Library Implementation

Provides discrete LUTs for TFHE programmable bootstrapping.

Key Properties:
- All functions map discrete inputs to discrete outputs
- LUT evaluation is EXACT (no approximation error)
- LUT size = 2^input_bits entries

Standard LUTs:
- step(x): x >= 0 ? 1 : 0
- sign(x): x > 0 ? 1 : (x < 0 ? -1 : 0)
- clip(x, lo, hi): max(lo, min(hi, x))
- argmax_2(a, b): a > b ? 0 : 1
- relu(x): max(0, x)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class LUTEntry:
    """A lookup table entry."""
    name: str
    entries: List[int]
    input_bits: int
    output_bits: int
    is_signed_input: bool = True
    is_signed_output: bool = False
    description: str = ""

    @property
    def size(self) -> int:
        """Number of entries in LUT."""
        return len(self.entries)

    def lookup(self, value: int) -> int:
        """Lookup a value in the table."""
        if self.is_signed_input:
            # Convert signed to index
            offset = self.size // 2
            index = value + offset
        else:
            index = value

        # Clamp to valid range
        index = max(0, min(self.size - 1, index))
        return self.entries[index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'entries': self.entries,
            'input_bits': self.input_bits,
            'output_bits': self.output_bits,
            'is_signed_input': self.is_signed_input,
            'is_signed_output': self.is_signed_output,
            'description': self.description,
        }


def step_lut(bits: int = 8) -> LUTEntry:
    """
    Create step function LUT.

    step(x) = 1 if x >= 0 else 0

    This is the primary gating function for gated LoRA.
    """
    size = 1 << bits
    half = size // 2

    entries = []
    for i in range(size):
        # Signed interpretation: i - half
        signed_val = i - half
        entries.append(1 if signed_val >= 0 else 0)

    return LUTEntry(
        name="step",
        entries=entries,
        input_bits=bits,
        output_bits=1,
        is_signed_input=True,
        is_signed_output=False,
        description="Step function: 1 if x >= 0, else 0",
    )


def sign_lut(bits: int = 8) -> LUTEntry:
    """
    Create sign function LUT.

    sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0

    Uses 2-bit signed output: {-1, 0, 1}
    """
    size = 1 << bits
    half = size // 2

    entries = []
    for i in range(size):
        signed_val = i - half
        if signed_val > 0:
            entries.append(1)
        elif signed_val < 0:
            entries.append(-1 + 256)  # Encode -1 as 255 for unsigned storage
        else:
            entries.append(0)

    return LUTEntry(
        name="sign",
        entries=entries,
        input_bits=bits,
        output_bits=2,
        is_signed_input=True,
        is_signed_output=True,
        description="Sign function: 1, 0, or -1",
    )


def clip_lut(bits: int = 8, lo: int = -64, hi: int = 64) -> LUTEntry:
    """
    Create clipping function LUT.

    clip(x) = max(lo, min(hi, x))

    Useful for keeping values in a bounded range.
    """
    size = 1 << bits
    half = size // 2

    entries = []
    for i in range(size):
        signed_val = i - half
        clipped = max(lo, min(hi, signed_val))
        # Store as unsigned
        entries.append(clipped + half)

    return LUTEntry(
        name=f"clip_{lo}_{hi}",
        entries=entries,
        input_bits=bits,
        output_bits=bits,
        is_signed_input=True,
        is_signed_output=True,
        description=f"Clip to [{lo}, {hi}]",
    )


def argmax_2_lut(bits: int = 8) -> LUTEntry:
    """
    Create binary argmax LUT.

    For input encoding (a, b) where a is high bits and b is low bits:
    argmax_2 = 0 if a > b else 1

    This is useful for routing between two adapter paths.
    """
    size = 1 << bits
    # Assume 4 bits each for a and b (if bits=8)
    half_bits = bits // 2
    mask = (1 << half_bits) - 1

    entries = []
    for i in range(size):
        a = (i >> half_bits) & mask  # High bits
        b = i & mask  # Low bits
        entries.append(0 if a > b else 1)

    return LUTEntry(
        name="argmax_2",
        entries=entries,
        input_bits=bits,
        output_bits=1,
        is_signed_input=False,
        is_signed_output=False,
        description="Binary argmax: 0 if a > b else 1 (a,b packed in input)",
    )


def relu_lut(bits: int = 8) -> LUTEntry:
    """
    Create ReLU function LUT.

    relu(x) = max(0, x)

    Note: For gated LoRA, use step_lut instead.
    This is provided for completeness.
    """
    size = 1 << bits
    half = size // 2

    entries = []
    for i in range(size):
        signed_val = i - half
        relu_val = max(0, signed_val)
        # Store as unsigned
        entries.append(relu_val)

    return LUTEntry(
        name="relu",
        entries=entries,
        input_bits=bits,
        output_bits=bits - 1,  # Output is non-negative
        is_signed_input=True,
        is_signed_output=False,
        description="ReLU: max(0, x)",
    )


def create_custom_lut(
    name: str,
    func: Callable[[int], int],
    bits: int = 8,
    is_signed: bool = True,
    description: str = "",
) -> LUTEntry:
    """
    Create a custom LUT from a function.

    Args:
        name: LUT name
        func: Function mapping int -> int
        bits: Input bit width
        is_signed: Whether input is signed
        description: Description of the function

    Returns:
        LUTEntry for the function
    """
    size = 1 << bits
    half = size // 2 if is_signed else 0

    entries = []
    for i in range(size):
        input_val = i - half if is_signed else i
        output_val = func(input_val)
        entries.append(output_val)

    # Determine output bits
    max_output = max(entries)
    min_output = min(entries)
    if min_output < 0:
        output_bits = max(abs(min_output), abs(max_output)).bit_length() + 1
        is_signed_output = True
    else:
        output_bits = max_output.bit_length() if max_output > 0 else 1
        is_signed_output = False

    return LUTEntry(
        name=name,
        entries=entries,
        input_bits=bits,
        output_bits=output_bits,
        is_signed_input=is_signed,
        is_signed_output=is_signed_output,
        description=description,
    )


class LUTLibrary:
    """
    Library of precomputed LUTs for TFHE operations.

    Provides:
    - Standard LUTs (step, sign, clip, argmax_2)
    - Custom LUT registration
    - LUT caching for reuse
    """

    def __init__(self, default_bits: int = 8):
        self.default_bits = default_bits
        self._luts: Dict[str, LUTEntry] = {}

        # Register standard LUTs
        self.register(step_lut(default_bits))
        self.register(sign_lut(default_bits))
        self.register(clip_lut(default_bits))
        self.register(argmax_2_lut(default_bits))
        self.register(relu_lut(default_bits))

    def register(self, lut: LUTEntry) -> None:
        """Register a LUT in the library."""
        self._luts[lut.name] = lut

    def get(self, name: str) -> Optional[LUTEntry]:
        """Get a LUT by name."""
        return self._luts.get(name)

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], LUTEntry],
    ) -> LUTEntry:
        """Get a LUT or create it using the factory."""
        if name not in self._luts:
            lut = factory()
            self.register(lut)
        return self._luts[name]

    def list_luts(self) -> List[str]:
        """List all registered LUT names."""
        return list(self._luts.keys())

    def get_entries(self, name: str) -> Optional[List[int]]:
        """Get LUT entries by name."""
        lut = self.get(name)
        return lut.entries if lut else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize library to dictionary."""
        return {
            'default_bits': self.default_bits,
            'luts': {name: lut.to_dict() for name, lut in self._luts.items()},
        }
