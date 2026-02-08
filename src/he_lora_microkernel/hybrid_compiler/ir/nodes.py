"""
IR Nodes for Hybrid CKKS-TFHE Compiler

Defines all IR operations including:
- CKKS ops (linear algebra only)
- TFHE ops (discrete logic only)
- Bridge ops (mandatory scheme switching)

All scheme switching MUST go through bridge ops.
Direct scheme mixing is a compile-time error.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from .types import Scheme, ValueType, Shape, IRValue, CKKSMetadata, TFHEMetadata
import numpy as np


# =============================================================================
# BASE IR NODE
# =============================================================================

@dataclass
class IRNode(ABC):
    """
    Base class for all IR operations.

    Each node must declare:
    - scheme: Which encryption scheme it operates in
    - inputs: List of input value names
    - outputs: List of output values
    """
    # Unique node ID
    node_id: str

    @property
    @abstractmethod
    def scheme(self) -> Scheme:
        """Get the scheme this operation runs in."""
        pass

    @property
    @abstractmethod
    def op_name(self) -> str:
        """Get operation name for display/debugging."""
        pass

    @abstractmethod
    def get_inputs(self) -> List[str]:
        """Get names of input values."""
        pass

    @abstractmethod
    def get_outputs(self) -> List[IRValue]:
        """Get output values with computed metadata."""
        pass

    @abstractmethod
    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        """
        Validate node with resolved inputs.

        Returns list of error messages (empty if valid).
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'node_id': self.node_id,
            'op_name': self.op_name,
            'scheme': self.scheme.name,
            'inputs': self.get_inputs(),
            'outputs': [o.to_dict() for o in self.get_outputs()],
        }


# =============================================================================
# CKKS OPERATIONS (Linear Algebra Only)
# =============================================================================

@dataclass
class CKKSMatMul(IRNode):
    """
    CKKS matrix multiplication: output = input @ weight^T

    Uses MOAI column packing when possible for rotation-free computation.

    Constraints:
    - Input must be CKKS REAL_APPROX
    - Weight is plaintext (Ct x Pt multiplication)
    - Consumes 1 multiplication depth
    """
    input_name: str
    weight_name: str  # Plaintext weight reference
    output_name: str

    # Output shape
    output_shape: Shape

    # Plaintext weight metadata
    weight_shape: Tuple[int, int]  # (out_features, in_features)

    # MOAI optimization flags
    use_column_packing: bool = True

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS

    @property
    def op_name(self) -> str:
        return "ckks_matmul"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        # Output metadata computed during validation
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None:
            errors.append(f"Input '{self.input_name}' not found")
            return errors

        if inp.scheme != Scheme.CKKS:
            errors.append(f"CKKSMatMul requires CKKS input, got {inp.scheme}")

        if inp.value_type != ValueType.REAL_APPROX:
            errors.append(f"CKKSMatMul requires REAL_APPROX, got {inp.value_type}")

        # Compute output metadata
        if not errors:
            out_meta = inp.ckks_meta.after_multiply()
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.CKKS,
                value_type=ValueType.REAL_APPROX,
                shape=self.output_shape,
                ckks_meta=CKKSMetadata(
                    precision_budget=out_meta.precision_budget,
                    scale=out_meta.scale,
                    level=out_meta.level,
                    is_moai_packed=self.use_column_packing,
                    original_shape=self.output_shape.dims,
                ),
            )

        return errors


@dataclass
class CKKSAdd(IRNode):
    """
    CKKS addition: output = lhs + rhs

    Constraints:
    - Both inputs must be CKKS REAL_APPROX
    - Scales must match (or be aligned via rescale)
    """
    lhs_name: str
    rhs_name: str
    output_name: str

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS

    @property
    def op_name(self) -> str:
        return "ckks_add"

    def get_inputs(self) -> List[str]:
        return [self.lhs_name, self.rhs_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        lhs = inputs.get(self.lhs_name)
        rhs = inputs.get(self.rhs_name)

        if lhs is None:
            errors.append(f"LHS '{self.lhs_name}' not found")
        if rhs is None:
            errors.append(f"RHS '{self.rhs_name}' not found")
        if errors:
            return errors

        if lhs.scheme != Scheme.CKKS:
            errors.append(f"CKKSAdd requires CKKS inputs, LHS is {lhs.scheme}")
        if rhs.scheme != Scheme.CKKS:
            errors.append(f"CKKSAdd requires CKKS inputs, RHS is {rhs.scheme}")

        if not errors:
            out_meta = lhs.ckks_meta.after_add(rhs.ckks_meta)
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.CKKS,
                value_type=ValueType.REAL_APPROX,
                shape=lhs.shape,  # Broadcasting rules would apply in real impl
                ckks_meta=out_meta,
            )

        return errors


@dataclass
class CKKSMul(IRNode):
    """
    CKKS element-wise multiplication: output = lhs * rhs

    Constraints:
    - Both inputs must be CKKS REAL_APPROX
    - Consumes 1 multiplication depth
    - Requires rescale afterward
    """
    lhs_name: str
    rhs_name: str
    output_name: str

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS

    @property
    def op_name(self) -> str:
        return "ckks_mul"

    def get_inputs(self) -> List[str]:
        return [self.lhs_name, self.rhs_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        lhs = inputs.get(self.lhs_name)
        rhs = inputs.get(self.rhs_name)

        if lhs is None or rhs is None:
            errors.append("Missing inputs")
            return errors

        if lhs.scheme != Scheme.CKKS or rhs.scheme != Scheme.CKKS:
            errors.append("CKKSMul requires CKKS inputs")

        if not errors:
            out_meta = lhs.ckks_meta.after_multiply()
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.CKKS,
                value_type=ValueType.REAL_APPROX,
                shape=lhs.shape,
                ckks_meta=out_meta,
            )

        return errors


@dataclass
class CKKSRescale(IRNode):
    """
    CKKS rescaling: reduces scale and consumes one level.

    Required after multiplication to manage scale growth.
    """
    input_name: str
    output_name: str
    scale_bits: int = 40

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS

    @property
    def op_name(self) -> str:
        return "ckks_rescale"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None:
            errors.append(f"Input '{self.input_name}' not found")
            return errors

        if inp.scheme != Scheme.CKKS:
            errors.append("CKKSRescale requires CKKS input")

        if not errors:
            out_meta = inp.ckks_meta.after_rescale(self.scale_bits)
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.CKKS,
                value_type=ValueType.REAL_APPROX,
                shape=inp.shape,
                ckks_meta=out_meta,
            )

        return errors


@dataclass
class CKKSRotate(IRNode):
    """
    CKKS slot rotation.

    Note: MOAI column packing aims to ELIMINATE rotations.
    This op should rarely appear in optimized IR.
    """
    input_name: str
    output_name: str
    rotation_amount: int

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS

    @property
    def op_name(self) -> str:
        return "ckks_rotate"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None or inp.scheme != Scheme.CKKS:
            errors.append("CKKSRotate requires CKKS input")
            return errors

        # Rotation doesn't change metadata significantly
        self._output_value = IRValue(
            name=self.output_name,
            scheme=Scheme.CKKS,
            value_type=ValueType.REAL_APPROX,
            shape=inp.shape,
            ckks_meta=inp.ckks_meta,
        )

        return errors


@dataclass
class CKKSPackMOAI(IRNode):
    """
    Pack values into MOAI column-packed format.

    This transforms the layout for rotation-free matrix multiplication.
    """
    input_name: str
    output_name: str
    block_size: int = 512

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS

    @property
    def op_name(self) -> str:
        return "ckks_pack_moai"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None or inp.scheme != Scheme.CKKS:
            errors.append("CKKSPackMOAI requires CKKS input")
            return errors

        self._output_value = IRValue(
            name=self.output_name,
            scheme=Scheme.CKKS,
            value_type=ValueType.REAL_APPROX,
            shape=inp.shape,
            ckks_meta=CKKSMetadata(
                precision_budget=inp.ckks_meta.precision_budget,
                scale=inp.ckks_meta.scale,
                level=inp.ckks_meta.level,
                is_moai_packed=True,
                original_shape=inp.shape.dims,
            ),
        )

        return errors


# =============================================================================
# TFHE OPERATIONS (Discrete Logic Only)
# =============================================================================

@dataclass
class TFHELUT(IRNode):
    """
    TFHE lookup table evaluation via programmable bootstrapping.

    Computes: output = LUT[input]

    This is the core TFHE operation for non-linear functions.
    The LUT is applied EXACTLY to discrete plaintext values.

    Constraints:
    - Input must be TFHE discrete type
    - Input shape must be scalar or tiny vector (<=16 elements)
    - Each LUT evaluation costs one bootstrap
    """
    input_name: str
    output_name: str

    # LUT specification
    lut_name: str  # Reference to precomputed LUT
    lut_entries: Optional[List[int]] = None  # Inline LUT (optional)

    # Output type (LUT can change bit width)
    output_type: ValueType = ValueType.BIT

    @property
    def scheme(self) -> Scheme:
        return Scheme.TFHE

    @property
    def op_name(self) -> str:
        return "tfhe_lut"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None:
            errors.append(f"Input '{self.input_name}' not found")
            return errors

        if inp.scheme != Scheme.TFHE:
            errors.append(f"TFHELUT requires TFHE input, got {inp.scheme}")

        if not inp.shape.is_tfhe_compatible:
            errors.append(
                f"TFHELUT input too large: {inp.shape.numel} > {Shape.MAX_TFHE_ELEMENTS}"
            )

        if not errors:
            # Bootstrap refreshes noise and applies LUT
            out_meta = inp.tfhe_meta.after_bootstrap(self.lut_name)
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.TFHE,
                value_type=self.output_type,
                shape=inp.shape,
                tfhe_meta=out_meta,
            )

        return errors


@dataclass
class TFHECompare(IRNode):
    """
    TFHE comparison: output = (lhs > rhs) ? 1 : 0

    Returns a single bit result.
    """
    lhs_name: str
    rhs_name: str  # Can be plaintext constant
    output_name: str
    comparison: str = "gt"  # "gt", "lt", "eq", "ge", "le"

    @property
    def scheme(self) -> Scheme:
        return Scheme.TFHE

    @property
    def op_name(self) -> str:
        return "tfhe_compare"

    def get_inputs(self) -> List[str]:
        return [self.lhs_name, self.rhs_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        lhs = inputs.get(self.lhs_name)

        if lhs is None:
            errors.append(f"LHS '{self.lhs_name}' not found")
            return errors

        if lhs.scheme != Scheme.TFHE:
            errors.append(f"TFHECompare requires TFHE input, got {lhs.scheme}")

        if not errors:
            # Comparison typically done via LUT, costs one bootstrap
            out_meta = lhs.tfhe_meta.after_bootstrap("compare")
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.TFHE,
                value_type=ValueType.BIT,
                shape=Shape.scalar(),
                tfhe_meta=out_meta,
            )

        return errors


@dataclass
class TFHEMUX(IRNode):
    """
    TFHE multiplexer: output = sel ? if_true : if_false

    Selector must be a bit. Branches can be larger discrete values.
    """
    selector_name: str
    if_true_name: str
    if_false_name: str
    output_name: str

    @property
    def scheme(self) -> Scheme:
        return Scheme.TFHE

    @property
    def op_name(self) -> str:
        return "tfhe_mux"

    def get_inputs(self) -> List[str]:
        return [self.selector_name, self.if_true_name, self.if_false_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        sel = inputs.get(self.selector_name)
        t_val = inputs.get(self.if_true_name)
        f_val = inputs.get(self.if_false_name)

        if sel is None or t_val is None or f_val is None:
            errors.append("Missing MUX inputs")
            return errors

        if sel.value_type != ValueType.BIT:
            errors.append("MUX selector must be BIT type")

        if t_val.value_type != f_val.value_type:
            errors.append("MUX branches must have same type")

        if not errors:
            out_meta = sel.tfhe_meta.after_operation(2.0)  # MUX grows noise
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.TFHE,
                value_type=t_val.value_type,
                shape=t_val.shape,
                tfhe_meta=out_meta,
            )

        return errors


@dataclass
class TFHEBootstrap(IRNode):
    """
    Explicit TFHE bootstrapping (noise refresh without LUT).

    Use when noise budget is low but no function evaluation needed.
    """
    input_name: str
    output_name: str

    @property
    def scheme(self) -> Scheme:
        return Scheme.TFHE

    @property
    def op_name(self) -> str:
        return "tfhe_bootstrap"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None or inp.scheme != Scheme.TFHE:
            errors.append("TFHEBootstrap requires TFHE input")
            return errors

        out_meta = inp.tfhe_meta.after_bootstrap()
        self._output_value = IRValue(
            name=self.output_name,
            scheme=Scheme.TFHE,
            value_type=inp.value_type,
            shape=inp.shape,
            tfhe_meta=out_meta,
        )

        return errors


# =============================================================================
# BRIDGE OPERATIONS (Mandatory Scheme Switching)
# =============================================================================

@dataclass
class CKKSQuantizeToInt(IRNode):
    """
    Quantize CKKS approximate value to discrete integer.

    This is the first step in CKKS -> TFHE conversion.
    Operates in CKKS domain but produces a conceptually discrete result.

    output = round(clamp(input, clip_min, clip_max) * scale)
    """
    input_name: str
    output_name: str

    # Quantization parameters
    bits: int = 8  # Target bit width
    clip_min: float = -1.0
    clip_max: float = 1.0

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS  # Still CKKS, but conceptually quantized

    @property
    def op_name(self) -> str:
        return "ckks_quantize_to_int"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None:
            errors.append(f"Input '{self.input_name}' not found")
            return errors

        if inp.scheme != Scheme.CKKS:
            errors.append("CKKSQuantizeToInt requires CKKS input")

        # Must be scalar or tiny vector for eventual TFHE conversion
        if not inp.shape.is_tfhe_compatible:
            errors.append(
                f"Quantization target must be TFHE-compatible size, "
                f"got {inp.shape.numel} elements"
            )

        if not errors:
            # Output is still CKKS but marked as quantized
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.CKKS,
                value_type=ValueType.REAL_APPROX,  # Still stored as CKKS
                shape=inp.shape,
                ckks_meta=CKKSMetadata(
                    precision_budget=inp.ckks_meta.precision_budget,
                    scale=inp.ckks_meta.scale,
                    level=inp.ckks_meta.level,
                    is_moai_packed=False,  # Unpacked for bridge
                ),
            )
            # Store quantization info for bridge
            self._output_value._quantization_bits = self.bits

        return errors


@dataclass
class CKKSToTFHE(IRNode):
    """
    Convert CKKS ciphertext to TFHE ciphertext.

    CRITICAL: This requires client interaction in a real system.
    The value must be decrypted with CKKS key, quantized, then
    re-encrypted with TFHE key.

    For security, this operation does NOT expose plaintext to server.
    In production, this is implemented via:
    - Client-side decryption + re-encryption, OR
    - Threshold cryptography, OR
    - Proxy re-encryption

    Constraints:
    - Input must be scalar or tiny vector (<=16 elements)
    - Input should be pre-quantized via CKKSQuantizeToInt
    """
    input_name: str
    output_name: str

    # Output discrete type
    output_type: ValueType = ValueType.INT_8

    @property
    def scheme(self) -> Scheme:
        return Scheme.TFHE  # Output is TFHE

    @property
    def op_name(self) -> str:
        return "ckks_to_tfhe"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None:
            errors.append(f"Input '{self.input_name}' not found")
            return errors

        if inp.scheme != Scheme.CKKS:
            errors.append("CKKSToTFHE requires CKKS input")

        if not inp.shape.is_tfhe_compatible:
            errors.append(
                f"CKKS->TFHE bridge limited to <={Shape.MAX_TFHE_ELEMENTS} elements"
            )

        if not errors:
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.TFHE,
                value_type=self.output_type,
                shape=inp.shape,
                tfhe_meta=TFHEMetadata(
                    noise_budget=128.0,  # Fresh encryption
                    is_fresh=True,
                ),
            )

        return errors


@dataclass
class TFHEToCKKS(IRNode):
    """
    Convert TFHE ciphertext to CKKS ciphertext.

    Similar security considerations as CKKSToTFHE.
    Requires client interaction for re-encryption.

    The discrete TFHE value is encoded as a CKKS approximate value.
    """
    input_name: str
    output_name: str

    # CKKS encoding parameters
    scale_bits: int = 40

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS  # Output is CKKS

    @property
    def op_name(self) -> str:
        return "tfhe_to_ckks"

    def get_inputs(self) -> List[str]:
        return [self.input_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        inp = inputs.get(self.input_name)

        if inp is None:
            errors.append(f"Input '{self.input_name}' not found")
            return errors

        if inp.scheme != Scheme.TFHE:
            errors.append("TFHEToCKKS requires TFHE input")

        if not errors:
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.CKKS,
                value_type=ValueType.REAL_APPROX,
                shape=inp.shape,
                ckks_meta=CKKSMetadata(
                    precision_budget=40.0,  # Fresh encryption
                    scale=2.0 ** self.scale_bits,
                    level=0,
                    is_moai_packed=False,
                ),
            )

        return errors


@dataclass
class CKKSApplyMask(IRNode):
    """
    Apply TFHE-computed gate/mask to CKKS tensor.

    output = gate * tensor

    This is the final step in gated computation:
    - gate: Computed via TFHE, converted back to CKKS (scalar or tiny)
    - tensor: Full CKKS tensor (the LoRA delta)

    The gate is broadcast across the tensor.
    """
    gate_name: str    # CKKS scalar/tiny (from TFHE conversion)
    tensor_name: str  # CKKS full tensor
    output_name: str

    @property
    def scheme(self) -> Scheme:
        return Scheme.CKKS

    @property
    def op_name(self) -> str:
        return "ckks_apply_mask"

    def get_inputs(self) -> List[str]:
        return [self.gate_name, self.tensor_name]

    def get_outputs(self) -> List[IRValue]:
        return [self._output_value]

    def validate(self, inputs: Dict[str, IRValue]) -> List[str]:
        errors = []
        gate = inputs.get(self.gate_name)
        tensor = inputs.get(self.tensor_name)

        if gate is None or tensor is None:
            errors.append("Missing inputs for CKKSApplyMask")
            return errors

        if gate.scheme != Scheme.CKKS:
            errors.append(f"Gate must be CKKS (after TFHE->CKKS), got {gate.scheme}")

        if tensor.scheme != Scheme.CKKS:
            errors.append(f"Tensor must be CKKS, got {tensor.scheme}")

        # Gate should be scalar or broadcastable
        if not gate.shape.is_scalar and gate.shape.numel > Shape.MAX_TFHE_ELEMENTS:
            errors.append("Gate should be scalar or tiny for efficient gating")

        if not errors:
            out_meta = tensor.ckks_meta.after_multiply()
            self._output_value = IRValue(
                name=self.output_name,
                scheme=Scheme.CKKS,
                value_type=ValueType.REAL_APPROX,
                shape=tensor.shape,
                ckks_meta=out_meta,
            )

        return errors


# =============================================================================
# IR PROGRAM
# =============================================================================

@dataclass
class IRProgram:
    """
    Complete IR program with nodes and value definitions.

    Maintains:
    - Ordered list of operations
    - Input/output specifications
    - Value type environment for validation
    """
    name: str
    nodes: List[IRNode] = field(default_factory=list)

    # Program inputs (encrypted or plaintext)
    inputs: Dict[str, IRValue] = field(default_factory=dict)

    # Program outputs
    outputs: List[str] = field(default_factory=list)

    # Computed value environment (filled during validation)
    _value_env: Dict[str, IRValue] = field(default_factory=dict, repr=False)

    def add_input(self, value: IRValue) -> None:
        """Add an input value definition."""
        self.inputs[value.name] = value
        self._value_env[value.name] = value

    def add_node(self, node: IRNode) -> None:
        """Add an IR node."""
        self.nodes.append(node)

    def add_output(self, name: str) -> None:
        """Mark a value as program output."""
        self.outputs.append(name)

    def get_value(self, name: str) -> Optional[IRValue]:
        """Get a value from the environment."""
        return self._value_env.get(name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'inputs': {k: v.to_dict() for k, v in self.inputs.items()},
            'nodes': [n.to_dict() for n in self.nodes],
            'outputs': self.outputs,
        }

    def __str__(self) -> str:
        lines = [f"Program: {self.name}", "Inputs:"]
        for v in self.inputs.values():
            lines.append(f"  {v}")
        lines.append("Operations:")
        for n in self.nodes:
            lines.append(f"  [{n.node_id}] {n.op_name}")
        lines.append(f"Outputs: {self.outputs}")
        return "\n".join(lines)
