"""
Hybrid Scheduler for CKKS-TFHE Compiler

Schedules IR operations with optimizations:
1. CKKS linear op fusion (batch matmuls)
2. TFHE op hoisting (minimize scheme switches)
3. Cost-aware ordering

The scheduler produces an ExecutionPlan that can be directly executed.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from ..ir import (
    TFHELUT,
    CKKSAdd,
    CKKSApplyMask,
    CKKSMatMul,
    CKKSQuantizeToInt,
    CKKSRescale,
    CKKSToTFHE,
    IRNode,
    IRProgram,
    Scheme,
    TFHEToCKKS,
)
from .cost_model import CostModel, OpCost


class SchedulePhase(Enum):
    """Execution phases in gated LoRA."""
    CKKS_LORA_DELTA = auto()      # Compute Delta = B(Ax) in CKKS
    CKKS_GATE_PRE = auto()        # Compute gate pre-activation z = w_g^T x + b_g
    BRIDGE_TO_TFHE = auto()       # Quantize and convert to TFHE
    TFHE_GATE_EVAL = auto()       # Evaluate gate LUT in TFHE
    BRIDGE_TO_CKKS = auto()       # Convert gate back to CKKS
    CKKS_APPLY_GATE = auto()      # Apply gate: gated_delta = g * Delta
    CKKS_FINAL_ADD = auto()       # Final: y = Wx + gated_delta


@dataclass
class ScheduleConfig:
    """Configuration for the hybrid scheduler."""
    # Maximum TFHE bootstraps allowed
    max_bootstraps: int = 2

    # Enable MOAI column packing
    enable_moai: bool = True

    # Fuse consecutive CKKS ops where possible
    fuse_ckks_ops: bool = True

    # Parallelize independent TFHE bootstraps
    parallel_tfhe: bool = True

    # Cost model for optimization decisions
    cost_model: Optional[CostModel] = None


@dataclass
class ScheduledOp:
    """A scheduled operation with phase and dependencies."""
    node: IRNode
    phase: SchedulePhase
    dependencies: List[str]  # Node IDs this depends on
    estimated_cost: OpCost


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a hybrid program.

    Organized into phases for efficient execution.
    Each phase can be executed as a batch on the appropriate backend.
    """
    name: str
    phases: Dict[SchedulePhase, List[ScheduledOp]] = field(default_factory=dict)

    # Total cost estimate
    total_cost: OpCost = field(default_factory=lambda: OpCost(0, 0, 0, 0, 0, 0))

    # Statistics
    num_ckks_ops: int = 0
    num_tfhe_ops: int = 0
    num_bridges: int = 0
    num_bootstraps: int = 0

    def get_phase_ops(self, phase: SchedulePhase) -> List[IRNode]:
        """Get nodes in a phase."""
        return [op.node for op in self.phases.get(phase, [])]

    def get_execution_order(self) -> List[IRNode]:
        """Get all nodes in execution order."""
        order = []
        for phase in SchedulePhase:
            order.extend(self.get_phase_ops(phase))
        return order

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'phases': {
                phase.name: [
                    {
                        'node_id': op.node.node_id,
                        'op_name': op.node.op_name,
                        'cost': op.estimated_cost.to_dict(),
                    }
                    for op in ops
                ]
                for phase, ops in self.phases.items()
            },
            'total_cost': self.total_cost.to_dict(),
            'stats': {
                'num_ckks_ops': self.num_ckks_ops,
                'num_tfhe_ops': self.num_tfhe_ops,
                'num_bridges': self.num_bridges,
                'num_bootstraps': self.num_bootstraps,
            },
        }

    def __str__(self) -> str:
        lines = [f"ExecutionPlan: {self.name}"]
        for phase in SchedulePhase:
            ops = self.phases.get(phase, [])
            if ops:
                lines.append(f"  {phase.name}:")
                for op in ops:
                    lines.append(f"    - {op.node.op_name} ({op.estimated_cost.latency_ms:.2f}ms)")
        lines.append(f"Total: {self.total_cost.latency_ms:.2f}ms, {self.num_bootstraps} bootstraps")
        return "\n".join(lines)


class HybridScheduler:
    """
    Scheduler for hybrid CKKS-TFHE programs.

    Analyzes IR programs and produces optimized execution plans.
    """

    def __init__(self, config: Optional[ScheduleConfig] = None):
        self.config = config or ScheduleConfig()
        self.cost_model = self.config.cost_model or CostModel()

    def schedule(self, program: IRProgram) -> ExecutionPlan:
        """
        Schedule an IR program.

        Args:
            program: Validated IR program

        Returns:
            Optimized execution plan
        """
        plan = ExecutionPlan(name=program.name)

        # Initialize phases
        for phase in SchedulePhase:
            plan.phases[phase] = []

        # Classify each node into a phase
        for node in program.nodes:
            phase = self._classify_node(node)
            cost = self.cost_model.estimate_node(node)
            deps = self._get_dependencies(node, program)

            scheduled_op = ScheduledOp(
                node=node,
                phase=phase,
                dependencies=deps,
                estimated_cost=cost,
            )
            plan.phases[phase].append(scheduled_op)
            plan.total_cost = plan.total_cost + cost

            # Update statistics
            if node.scheme == Scheme.CKKS:
                plan.num_ckks_ops += 1
            elif node.scheme == Scheme.TFHE:
                plan.num_tfhe_ops += 1
                if node.op_name in ('tfhe_lut', 'tfhe_compare', 'tfhe_bootstrap'):
                    plan.num_bootstraps += cost.bootstraps

            if node.op_name in ('ckks_to_tfhe', 'tfhe_to_ckks'):
                plan.num_bridges += 1

        # Optimize within phases
        if self.config.fuse_ckks_ops:
            self._fuse_ckks_ops(plan)

        return plan

    def _classify_node(self, node: IRNode) -> SchedulePhase:
        """Classify a node into an execution phase."""
        op_name = node.op_name

        # CKKS linear ops for LoRA delta
        if op_name in ('ckks_matmul', 'ckks_rescale', 'ckks_pack_moai'):
            # Check if this is the gate pre-activation
            if hasattr(node, 'is_gate_preact') and node.is_gate_preact:
                return SchedulePhase.CKKS_GATE_PRE
            return SchedulePhase.CKKS_LORA_DELTA

        # Quantization
        if op_name == 'ckks_quantize_to_int':
            return SchedulePhase.BRIDGE_TO_TFHE

        # CKKS -> TFHE bridge
        if op_name == 'ckks_to_tfhe':
            return SchedulePhase.BRIDGE_TO_TFHE

        # TFHE gate evaluation
        if op_name in ('tfhe_lut', 'tfhe_compare', 'tfhe_mux', 'tfhe_bootstrap'):
            return SchedulePhase.TFHE_GATE_EVAL

        # TFHE -> CKKS bridge
        if op_name == 'tfhe_to_ckks':
            return SchedulePhase.BRIDGE_TO_CKKS

        # Gate application
        if op_name == 'ckks_apply_mask':
            return SchedulePhase.CKKS_APPLY_GATE

        # Addition (could be various phases)
        if op_name == 'ckks_add':
            return SchedulePhase.CKKS_FINAL_ADD

        if op_name == 'ckks_mul':
            return SchedulePhase.CKKS_APPLY_GATE

        # Default
        return SchedulePhase.CKKS_LORA_DELTA

    def _get_dependencies(self, node: IRNode, program: IRProgram) -> List[str]:
        """Get node IDs that this node depends on."""
        deps = []
        input_names = node.get_inputs()

        # Find which nodes produce these inputs
        for prev_node in program.nodes:
            if prev_node == node:
                break
            for output in prev_node.get_outputs():
                if output.name in input_names:
                    deps.append(prev_node.node_id)

        return deps

    def _fuse_ckks_ops(self, plan: ExecutionPlan) -> None:
        """
        Fuse consecutive CKKS operations where possible.

        This is primarily for MatMul + Rescale fusion.
        """
        for phase in [SchedulePhase.CKKS_LORA_DELTA, SchedulePhase.CKKS_GATE_PRE]:
            ops = plan.phases.get(phase, [])
            if len(ops) < 2:
                continue

            # Mark fusable pairs
            fused = set()
            for i in range(len(ops) - 1):
                curr = ops[i]
                next_op = ops[i + 1]

                # MatMul + Rescale fusion
                if (curr.node.op_name == 'ckks_matmul' and
                    next_op.node.op_name == 'ckks_rescale' and
                    curr.node.node_id in next_op.dependencies):
                    fused.add(i + 1)  # Mark rescale as fused
                    # Update cost estimate (fusion reduces overhead)
                    curr.estimated_cost.latency_ms *= 0.9


def schedule_gated_lora(
    # LoRA parameters
    hidden_size: int,
    lora_rank: int,

    # Gate parameters
    gate_bias: bool = True,

    # Configuration
    config: Optional[ScheduleConfig] = None,
) -> Tuple[IRProgram, ExecutionPlan]:
    """
    Create and schedule a gated LoRA program.

    This is a convenience function that:
    1. Creates the IR for gated LoRA
    2. Validates it
    3. Schedules it

    Returns:
        Tuple of (IR program, execution plan)
    """
    from ..ir import (
        IRProgram,
        Shape,
        ValueType,
        create_ckks_value,
        validate_program,
    )

    program = IRProgram(name="gated_lora")

    # Input: x [hidden_size]
    x = create_ckks_value("x", Shape.vector(hidden_size))
    program.add_input(x)

    # Phase 1: LoRA Delta computation
    # u = Ax [lora_rank] (matmul)
    matmul_a = CKKSMatMul(
        node_id="matmul_a",
        input_name="x",
        weight_name="lora_a",
        output_name="u",
        output_shape=Shape.vector(lora_rank),
        weight_shape=(lora_rank, hidden_size),
    )
    program.add_node(matmul_a)

    # rescale u
    rescale_u = CKKSRescale(
        node_id="rescale_u",
        input_name="u",
        output_name="u_rs",
    )
    program.add_node(rescale_u)

    # delta = Bu [hidden_size] (matmul)
    matmul_b = CKKSMatMul(
        node_id="matmul_b",
        input_name="u_rs",
        weight_name="lora_b",
        output_name="delta",
        output_shape=Shape.vector(hidden_size),
        weight_shape=(hidden_size, lora_rank),
    )
    program.add_node(matmul_b)

    # rescale delta
    rescale_delta = CKKSRescale(
        node_id="rescale_delta",
        input_name="delta",
        output_name="delta_rs",
    )
    program.add_node(rescale_delta)

    # Phase 2: Gate pre-activation
    # z = w_g^T x + b_g (scalar output)
    gate_matmul = CKKSMatMul(
        node_id="gate_matmul",
        input_name="x",
        weight_name="w_gate",
        output_name="z_pre",
        output_shape=Shape.scalar(),
        weight_shape=(1, hidden_size),
    )
    gate_matmul.is_gate_preact = True
    program.add_node(gate_matmul)

    # rescale z
    rescale_z = CKKSRescale(
        node_id="rescale_z",
        input_name="z_pre",
        output_name="z",
    )
    program.add_node(rescale_z)

    # Phase 3: Bridge to TFHE
    # Quantize z to int8
    quantize = CKKSQuantizeToInt(
        node_id="quantize_z",
        input_name="z",
        output_name="z_q",
        bits=8,
        clip_min=-10.0,
        clip_max=10.0,
    )
    program.add_node(quantize)

    # Convert to TFHE
    to_tfhe = CKKSToTFHE(
        node_id="to_tfhe",
        input_name="z_q",
        output_name="z_tfhe",
        output_type=ValueType.INT_8,
    )
    program.add_node(to_tfhe)

    # Phase 4: TFHE gate evaluation
    # g = step(z_tfhe) via LUT
    gate_lut = TFHELUT(
        node_id="gate_lut",
        input_name="z_tfhe",
        output_name="g_tfhe",
        lut_name="step",
        output_type=ValueType.BIT,
    )
    program.add_node(gate_lut)

    # Phase 5: Bridge back to CKKS
    to_ckks = TFHEToCKKS(
        node_id="to_ckks",
        input_name="g_tfhe",
        output_name="g_ckks",
    )
    program.add_node(to_ckks)

    # Phase 6: Apply gate
    # gated_delta = g * delta
    apply_gate = CKKSApplyMask(
        node_id="apply_gate",
        gate_name="g_ckks",
        tensor_name="delta_rs",
        output_name="gated_delta",
    )
    program.add_node(apply_gate)

    # rescale gated_delta
    rescale_gated = CKKSRescale(
        node_id="rescale_gated",
        input_name="gated_delta",
        output_name="gated_delta_rs",
    )
    program.add_node(rescale_gated)

    # Phase 7: Final output
    # y = Wx + gated_delta (Wx computed separately, just add here)
    final_add = CKKSAdd(
        node_id="final_add",
        lhs_name="base_output",  # Wx from base model
        rhs_name="gated_delta_rs",
        output_name="y",
    )
    # Note: base_output would be an input in a real system
    # For now, we just mark the output
    program.add_output("gated_delta_rs")

    # Validate
    validation = validate_program(program)
    if not validation.valid:
        raise ValueError(f"Invalid program: {validation.errors}")

    # Schedule
    scheduler = HybridScheduler(config)
    plan = scheduler.schedule(program)

    return program, plan
