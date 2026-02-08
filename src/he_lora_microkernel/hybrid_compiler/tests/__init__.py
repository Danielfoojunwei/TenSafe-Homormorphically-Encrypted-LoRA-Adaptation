"""
Tests for Hybrid CKKS-TFHE Compiler

Test Categories:
1. Functional: Verify gated LoRA correctness vs plaintext reference
2. Validation: Ensure scheme violations are caught
3. Precision: Measure CKKS error before/after gating
4. Performance: Compare latency with baseline
"""

from .test_gated_lora import *
from .test_ir import *
from .test_precision import *
