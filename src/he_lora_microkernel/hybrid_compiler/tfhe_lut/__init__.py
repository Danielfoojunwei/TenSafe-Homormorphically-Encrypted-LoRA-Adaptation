"""
TFHE Lookup Table Library

Provides precomputed LUTs for common functions:
- step: Step function (threshold at 0)
- sign: Sign function (-1, 0, 1)
- clip: Clipping function
- argmax_2: Binary argmax

All LUTs operate on discrete message space and are
evaluated EXACTLY via TFHE programmable bootstrapping.
"""

from .lut_library import (
    LUTLibrary,
    LUTEntry,
    step_lut,
    sign_lut,
    clip_lut,
    argmax_2_lut,
    relu_lut,
    create_custom_lut,
)

__all__ = [
    "LUTLibrary",
    "LUTEntry",
    "step_lut",
    "sign_lut",
    "clip_lut",
    "argmax_2_lut",
    "relu_lut",
    "create_custom_lut",
]
