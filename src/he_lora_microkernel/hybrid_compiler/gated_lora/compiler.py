"""
Gated LoRA Compiler

Compiles gated LoRA configurations into optimized hybrid IR programs.

Gated LoRA Form:
    y = Wx + g(x) * Delta(x)

Where:
    - Delta(x) = B(Ax) - Standard LoRA delta (CKKS)
    - g(x) = step(w_g^T x + b_g) - Discrete gate (TFHE)

The compiler:
1. Creates IR for both CKKS and TFHE paths
2. Inserts bridge ops for scheme switching
3. Validates the complete program
4. Schedules for optimal execution
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

from ..ir import (
    IRProgram, IRValue, Shape, Scheme, ValueType,
    create_ckks_value, create_tfhe_value,
    CKKSMatMul, CKKSAdd, CKKSMul, CKKSRescale, CKKSPackMOAI, CKKSApplyMask,
    TFHELUT, CKKSQuantizeToInt, CKKSToTFHE, TFHEToCKKS,
    validate_program,
)
from ..scheduler import HybridScheduler, ScheduleConfig, ExecutionPlan
from ..tfhe_lut import LUTLibrary, step_lut


@dataclass
class GatedLoRAConfig:
    """Configuration for gated LoRA compilation."""
    # Model dimensions
    hidden_size: int
    lora_rank: int

    # Gate configuration
    gate_type: str = "step"  # "step", "sign", "sigmoid"
    gate_bias: bool = True

    # Quantization for TFHE
    quantization_bits: int = 8
    clip_range: Tuple[float, float] = (-10.0, 10.0)

    # CKKS parameters
    ckks_scale_bits: int = 40
    use_moai_packing: bool = True

    # Layer targeting
    layer_indices: Optional[List[int]] = None  # None = all layers
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Optimization
    fuse_matmul_rescale: bool = True

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        if self.hidden_size <= 0:
            errors.append("hidden_size must be positive")
        if self.lora_rank <= 0 or self.lora_rank > self.hidden_size:
            errors.append("lora_rank must be in (0, hidden_size]")
        if self.quantization_bits < 4 or self.quantization_bits > 12:
            errors.append("quantization_bits should be in [4, 12]")
        return errors


class GatedLoRACompiler:
    """
    Compiler for gated LoRA programs.

    Produces optimized hybrid CKKS-TFHE IR from configuration.
    """

    def __init__(
        self,
        config: GatedLoRAConfig,
        lut_library: Optional[LUTLibrary] = None,
    ):
        self.config = config
        self.lut_library = lut_library or LUTLibrary(config.quantization_bits)

        # Ensure required LUTs are available
        if config.gate_type not in self.lut_library.list_luts():
            if config.gate_type == "step":
                self.lut_library.register(step_lut(config.quantization_bits))

        self._node_counter = 0

    def _next_node_id(self, prefix: str = "node") -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def compile(self) -> Tuple[IRProgram, ExecutionPlan]:
        """
        Compile gated LoRA configuration to IR and execution plan.

        Returns:
            Tuple of (IR program, execution plan)
        """
        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {errors}")

        # Create program
        program = self._create_ir_program()

        # Validate program
        validation = validate_program(program)
        if not validation.valid:
            error_msgs = [str(e) for e in validation.errors]
            raise ValueError(f"IR validation failed: {error_msgs}")

        # Schedule
        scheduler = HybridScheduler(ScheduleConfig(
            enable_moai=self.config.use_moai_packing,
            fuse_ckks_ops=self.config.fuse_matmul_rescale,
        ))
        plan = scheduler.schedule(program)

        return program, plan

    def _create_ir_program(self) -> IRProgram:
        """Create the IR program for gated LoRA."""
        cfg = self.config
        program = IRProgram(name="gated_lora")

        # =====================================================================
        # INPUTS
        # =====================================================================

        # x: Input activation [hidden_size]
        x = create_ckks_value(
            name="x",
            shape=Shape.vector(cfg.hidden_size),
            precision_budget=40.0,
            scale_bits=cfg.ckks_scale_bits,
        )
        program.add_input(x)

        # base_output: Wx from base model [hidden_size]
        base_output = create_ckks_value(
            name="base_output",
            shape=Shape.vector(cfg.hidden_size),
            precision_budget=40.0,
            scale_bits=cfg.ckks_scale_bits,
        )
        program.add_input(base_output)

        # =====================================================================
        # PHASE 1: LoRA Delta (CKKS with MOAI)
        # =====================================================================

        # Pack input for MOAI if enabled
        if cfg.use_moai_packing:
            pack_x = CKKSPackMOAI(
                node_id=self._next_node_id("pack"),
                input_name="x",
                output_name="x_packed",
            )
            program.add_node(pack_x)
            lora_input = "x_packed"
        else:
            lora_input = "x"

        # u = A @ x [lora_rank]
        matmul_a = CKKSMatMul(
            node_id=self._next_node_id("matmul_a"),
            input_name=lora_input,
            weight_name="lora_A",
            output_name="u",
            output_shape=Shape.vector(cfg.lora_rank),
            weight_shape=(cfg.lora_rank, cfg.hidden_size),
            use_column_packing=cfg.use_moai_packing,
        )
        program.add_node(matmul_a)

        # Rescale u
        rescale_u = CKKSRescale(
            node_id=self._next_node_id("rescale"),
            input_name="u",
            output_name="u_rs",
            scale_bits=cfg.ckks_scale_bits,
        )
        program.add_node(rescale_u)

        # delta = B @ u [hidden_size]
        matmul_b = CKKSMatMul(
            node_id=self._next_node_id("matmul_b"),
            input_name="u_rs",
            weight_name="lora_B",
            output_name="delta",
            output_shape=Shape.vector(cfg.hidden_size),
            weight_shape=(cfg.hidden_size, cfg.lora_rank),
            use_column_packing=cfg.use_moai_packing,
        )
        program.add_node(matmul_b)

        # Rescale delta
        rescale_delta = CKKSRescale(
            node_id=self._next_node_id("rescale"),
            input_name="delta",
            output_name="delta_rs",
            scale_bits=cfg.ckks_scale_bits,
        )
        program.add_node(rescale_delta)

        # =====================================================================
        # PHASE 2: Gate Pre-activation (CKKS)
        # =====================================================================

        # z = w_g^T @ x + b_g [scalar]
        gate_matmul = CKKSMatMul(
            node_id=self._next_node_id("gate_matmul"),
            input_name="x",  # Use original x, not packed
            weight_name="w_gate",
            output_name="z_pre",
            output_shape=Shape.scalar(),
            weight_shape=(1, cfg.hidden_size),
            use_column_packing=False,  # Scalar output, no packing
        )
        program.add_node(gate_matmul)

        # Rescale z
        rescale_z = CKKSRescale(
            node_id=self._next_node_id("rescale"),
            input_name="z_pre",
            output_name="z",
            scale_bits=cfg.ckks_scale_bits,
        )
        program.add_node(rescale_z)

        # =====================================================================
        # PHASE 3: Bridge CKKS -> TFHE
        # =====================================================================

        # Quantize z to int
        quantize = CKKSQuantizeToInt(
            node_id=self._next_node_id("quantize"),
            input_name="z",
            output_name="z_q",
            bits=cfg.quantization_bits,
            clip_min=cfg.clip_range[0],
            clip_max=cfg.clip_range[1],
        )
        program.add_node(quantize)

        # Convert to TFHE
        to_tfhe = CKKSToTFHE(
            node_id=self._next_node_id("to_tfhe"),
            input_name="z_q",
            output_name="z_tfhe",
            output_type=ValueType.INT_8 if cfg.quantization_bits == 8 else ValueType.INT_4,
        )
        program.add_node(to_tfhe)

        # =====================================================================
        # PHASE 4: Gate Evaluation (TFHE)
        # =====================================================================

        # g = LUT[z] via programmable bootstrapping
        gate_lut = TFHELUT(
            node_id=self._next_node_id("gate_lut"),
            input_name="z_tfhe",
            output_name="g_tfhe",
            lut_name=cfg.gate_type,
            output_type=ValueType.BIT,
        )
        program.add_node(gate_lut)

        # =====================================================================
        # PHASE 5: Bridge TFHE -> CKKS
        # =====================================================================

        to_ckks = TFHEToCKKS(
            node_id=self._next_node_id("to_ckks"),
            input_name="g_tfhe",
            output_name="g_ckks",
            scale_bits=cfg.ckks_scale_bits,
        )
        program.add_node(to_ckks)

        # =====================================================================
        # PHASE 6: Apply Gate (CKKS)
        # =====================================================================

        # gated_delta = g * delta
        apply_gate = CKKSApplyMask(
            node_id=self._next_node_id("apply_gate"),
            gate_name="g_ckks",
            tensor_name="delta_rs",
            output_name="gated_delta",
        )
        program.add_node(apply_gate)

        # Rescale gated_delta
        rescale_gated = CKKSRescale(
            node_id=self._next_node_id("rescale"),
            input_name="gated_delta",
            output_name="gated_delta_rs",
            scale_bits=cfg.ckks_scale_bits,
        )
        program.add_node(rescale_gated)

        # =====================================================================
        # PHASE 7: Final Output (CKKS)
        # =====================================================================

        # y = base_output + gated_delta
        final_add = CKKSAdd(
            node_id=self._next_node_id("final_add"),
            lhs_name="base_output",
            rhs_name="gated_delta_rs",
            output_name="y",
        )
        program.add_node(final_add)

        # Mark output
        program.add_output("y")

        return program


def compile_gated_lora(
    hidden_size: int,
    lora_rank: int,
    gate_type: str = "step",
    **kwargs,
) -> Tuple[IRProgram, ExecutionPlan]:
    """
    Convenience function to compile gated LoRA.

    Args:
        hidden_size: Model hidden dimension
        lora_rank: LoRA rank
        gate_type: Type of gate function
        **kwargs: Additional GatedLoRAConfig parameters

    Returns:
        Tuple of (IR program, execution plan)
    """
    config = GatedLoRAConfig(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
        gate_type=gate_type,
        **kwargs,
    )
    compiler = GatedLoRACompiler(config)
    return compiler.compile()
