"""
IR Validation for Hybrid CKKS-TFHE Compiler

Enforces critical invariants:
1. No direct scheme mixing (CKKS op cannot take TFHE input directly)
2. All scheme switches go through bridge ops
3. TFHE operations are limited to scalars/tiny vectors
4. No more than MAX_BOOTSTRAPS_PER_LAYER TFHE bootstraps

Validation is performed at compile time to prevent invalid programs.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from .types import Scheme, ValueType, Shape, IRValue
from .nodes import (
    IRNode, IRProgram,
    CKKSMatMul, CKKSAdd, CKKSMul, CKKSRescale, CKKSRotate, CKKSPackMOAI,
    TFHELUT, TFHECompare, TFHEMUX, TFHEBootstrap,
    CKKSQuantizeToInt, CKKSToTFHE, TFHEToCKKS, CKKSApplyMask,
)


# =============================================================================
# VALIDATION ERRORS
# =============================================================================

class ValidationError(Exception):
    """Base class for IR validation errors."""
    pass


class SchemeViolationError(ValidationError):
    """Raised when scheme mixing rules are violated."""
    def __init__(self, message: str, node_id: str, expected: Scheme, actual: Scheme):
        self.node_id = node_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"[{node_id}] {message}: expected {expected.name}, got {actual.name}")


class TypeMismatchError(ValidationError):
    """Raised when value types don't match."""
    def __init__(self, message: str, node_id: str, expected: ValueType, actual: ValueType):
        self.node_id = node_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"[{node_id}] {message}: expected {expected.value}, got {actual.value}")


class ShapeError(ValidationError):
    """Raised when shapes are incompatible."""
    def __init__(self, message: str, node_id: str, shape: Shape):
        self.node_id = node_id
        self.shape = shape
        super().__init__(f"[{node_id}] {message}: shape {shape}")


class BridgeRequiredError(ValidationError):
    """Raised when a bridge op is missing between scheme domains."""
    def __init__(self, node_id: str, from_scheme: Scheme, to_scheme: Scheme):
        self.node_id = node_id
        self.from_scheme = from_scheme
        self.to_scheme = to_scheme
        super().__init__(
            f"[{node_id}] Bridge required: {from_scheme.name} -> {to_scheme.name}. "
            f"Use CKKSToTFHE or TFHEToCKKS explicitly."
        )


class BootstrapBudgetError(ValidationError):
    """Raised when bootstrap count exceeds budget."""
    def __init__(self, count: int, budget: int):
        self.count = count
        self.budget = budget
        super().__init__(
            f"Bootstrap budget exceeded: {count} > {budget}. "
            f"Reduce TFHE operations or restructure computation."
        )


# =============================================================================
# VALIDATION RULES
# =============================================================================

# Maximum TFHE bootstraps allowed per program/layer
MAX_BOOTSTRAPS_PER_LAYER = 2

# Ops that require specific input schemes
CKKS_OPS = {CKKSMatMul, CKKSAdd, CKKSMul, CKKSRescale, CKKSRotate, CKKSPackMOAI, CKKSApplyMask}
TFHE_OPS = {TFHELUT, TFHECompare, TFHEMUX, TFHEBootstrap}
BRIDGE_OPS = {CKKSQuantizeToInt, CKKSToTFHE, TFHEToCKKS}

# Ops that cost a bootstrap
BOOTSTRAP_OPS = {TFHELUT, TFHECompare, TFHEBootstrap}


@dataclass
class ValidationResult:
    """Result of program validation."""
    valid: bool
    errors: List[ValidationError]
    warnings: List[str]

    # Statistics
    ckks_op_count: int = 0
    tfhe_op_count: int = 0
    bootstrap_count: int = 0
    bridge_count: int = 0
    rotation_count: int = 0

    def __str__(self) -> str:
        if self.valid:
            return (
                f"Valid program: {self.ckks_op_count} CKKS ops, "
                f"{self.tfhe_op_count} TFHE ops, {self.bootstrap_count} bootstraps"
            )
        return f"Invalid program: {len(self.errors)} errors"


def validate_node(
    node: IRNode,
    value_env: Dict[str, IRValue],
) -> Tuple[List[ValidationError], List[str]]:
    """
    Validate a single IR node.

    Args:
        node: The IR node to validate
        value_env: Current value environment

    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []

    # Get input values
    input_names = node.get_inputs()
    inputs = {}

    for name in input_names:
        if name not in value_env:
            errors.append(ValidationError(f"[{node.node_id}] Unknown input: {name}"))
            continue
        inputs[name] = value_env[name]

    if errors:
        return errors, warnings

    # Check scheme compatibility
    node_type = type(node)

    # CKKS ops require CKKS inputs
    if node_type in CKKS_OPS:
        for name, value in inputs.items():
            if value.scheme != Scheme.CKKS:
                errors.append(SchemeViolationError(
                    f"CKKS op {node.op_name} requires CKKS input '{name}'",
                    node.node_id, Scheme.CKKS, value.scheme
                ))

    # TFHE ops require TFHE inputs (except for constants)
    if node_type in TFHE_OPS:
        for name, value in inputs.items():
            if value.scheme != Scheme.TFHE:
                errors.append(SchemeViolationError(
                    f"TFHE op {node.op_name} requires TFHE input '{name}'",
                    node.node_id, Scheme.TFHE, value.scheme
                ))

            # Check TFHE size constraints
            if not value.shape.is_tfhe_compatible:
                errors.append(ShapeError(
                    f"TFHE input '{name}' exceeds size limit ({Shape.MAX_TFHE_ELEMENTS})",
                    node.node_id, value.shape
                ))

    # Bridge ops: CKKSToTFHE takes CKKS, outputs TFHE
    if node_type == CKKSToTFHE:
        for name, value in inputs.items():
            if value.scheme != Scheme.CKKS:
                errors.append(SchemeViolationError(
                    f"CKKSToTFHE requires CKKS input",
                    node.node_id, Scheme.CKKS, value.scheme
                ))
            # Warn if not quantized first
            if not hasattr(value, '_quantization_bits'):
                warnings.append(
                    f"[{node.node_id}] CKKSToTFHE input should be quantized first "
                    f"via CKKSQuantizeToInt for precision control"
                )

    # Bridge ops: TFHEToCKKS takes TFHE, outputs CKKS
    if node_type == TFHEToCKKS:
        for name, value in inputs.items():
            if value.scheme != Scheme.TFHE:
                errors.append(SchemeViolationError(
                    f"TFHEToCKKS requires TFHE input",
                    node.node_id, Scheme.TFHE, value.scheme
                ))

    # Run node-specific validation
    node_errors = node.validate(inputs)
    for err_msg in node_errors:
        errors.append(ValidationError(f"[{node.node_id}] {err_msg}"))

    # Check for rotation (MOAI aims to eliminate these)
    if node_type == CKKSRotate:
        warnings.append(
            f"[{node.node_id}] CKKSRotate detected. "
            f"Consider MOAI column packing to eliminate rotations."
        )

    return errors, warnings


def validate_program(
    program: IRProgram,
    max_bootstraps: int = MAX_BOOTSTRAPS_PER_LAYER,
) -> ValidationResult:
    """
    Validate an entire IR program.

    Checks:
    1. All nodes are individually valid
    2. No direct scheme mixing without bridges
    3. Bootstrap count within budget
    4. All outputs are defined

    Args:
        program: The IR program to validate
        max_bootstraps: Maximum allowed bootstraps

    Returns:
        ValidationResult with errors, warnings, and statistics
    """
    errors = []
    warnings = []

    # Statistics
    ckks_ops = 0
    tfhe_ops = 0
    bootstraps = 0
    bridges = 0
    rotations = 0

    # Initialize value environment with inputs
    value_env = dict(program.inputs)

    # Validate each node in order
    for node in program.nodes:
        # Validate the node
        node_errors, node_warnings = validate_node(node, value_env)
        errors.extend(node_errors)
        warnings.extend(node_warnings)

        # Update statistics
        node_type = type(node)
        if node_type in CKKS_OPS:
            ckks_ops += 1
        if node_type in TFHE_OPS:
            tfhe_ops += 1
        if node_type in BOOTSTRAP_OPS:
            bootstraps += 1
        if node_type in BRIDGE_OPS:
            bridges += 1
        if node_type == CKKSRotate:
            rotations += 1

        # Add outputs to environment (even if errors, for subsequent validation)
        if not node_errors:
            for output in node.get_outputs():
                value_env[output.name] = output

    # Check bootstrap budget
    if bootstraps > max_bootstraps:
        errors.append(BootstrapBudgetError(bootstraps, max_bootstraps))

    # Check all outputs are defined
    for output_name in program.outputs:
        if output_name not in value_env:
            errors.append(ValidationError(f"Output '{output_name}' not defined"))

    # Check for suspicious patterns (warnings)
    if tfhe_ops > 0 and bridges == 0:
        warnings.append(
            "TFHE ops present but no bridge ops. "
            "This may indicate missing CKKS->TFHE conversion."
        )

    if rotations > 0:
        warnings.append(
            f"{rotations} rotation(s) detected. "
            f"MOAI column packing should eliminate all rotations for optimal performance."
        )

    # Update program's value environment
    program._value_env = value_env

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        ckks_op_count=ckks_ops,
        tfhe_op_count=tfhe_ops,
        bootstrap_count=bootstraps,
        bridge_count=bridges,
        rotation_count=rotations,
    )


def check_scheme_boundaries(program: IRProgram) -> List[Tuple[str, str, Scheme, Scheme]]:
    """
    Find all scheme boundary crossings in a program.

    Returns list of (producer_node, consumer_node, from_scheme, to_scheme)
    for any edges where schemes differ without an intervening bridge.

    Useful for debugging scheme mixing issues.
    """
    boundaries = []

    # Build producer map: value_name -> (node_id, scheme)
    producers: Dict[str, Tuple[str, Scheme]] = {}

    # Inputs are initial producers
    for name, value in program.inputs.items():
        producers[name] = ("input", value.scheme)

    for node in program.nodes:
        # Check all inputs for scheme mismatch
        for input_name in node.get_inputs():
            if input_name in producers:
                prod_node, prod_scheme = producers[input_name]

                # Expected scheme for this node type
                if type(node) in CKKS_OPS:
                    expected = Scheme.CKKS
                elif type(node) in TFHE_OPS:
                    expected = Scheme.TFHE
                elif type(node) == CKKSToTFHE:
                    expected = Scheme.CKKS
                elif type(node) == TFHEToCKKS:
                    expected = Scheme.TFHE
                else:
                    continue

                if prod_scheme != expected:
                    boundaries.append((prod_node, node.node_id, prod_scheme, expected))

        # Register outputs
        for output in node.get_outputs():
            producers[output.name] = (node.node_id, output.scheme)

    return boundaries
