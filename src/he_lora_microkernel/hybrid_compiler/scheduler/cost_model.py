"""
Cost Model for Hybrid CKKS-TFHE Operations

Provides estimates for:
- CKKS operation latencies
- TFHE bootstrap costs
- Scheme switching overhead
- Memory usage

Used by scheduler for cost-aware optimization.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
import math

from ..ir import Scheme, IRNode, IRProgram


class OpType(Enum):
    """Operation types for cost modeling."""
    # CKKS
    CKKS_MATMUL = "ckks_matmul"
    CKKS_ADD = "ckks_add"
    CKKS_MUL = "ckks_mul"
    CKKS_RESCALE = "ckks_rescale"
    CKKS_ROTATE = "ckks_rotate"
    CKKS_PACK = "ckks_pack"

    # TFHE
    TFHE_BOOTSTRAP = "tfhe_bootstrap"
    TFHE_LUT = "tfhe_lut"
    TFHE_COMPARE = "tfhe_compare"
    TFHE_MUX = "tfhe_mux"

    # Bridge
    BRIDGE_CKKS_TO_TFHE = "bridge_ckks_to_tfhe"
    BRIDGE_TFHE_TO_CKKS = "bridge_tfhe_to_ckks"
    BRIDGE_QUANTIZE = "bridge_quantize"


@dataclass
class OpCost:
    """Cost estimate for a single operation."""
    # Latency in milliseconds
    latency_ms: float

    # Memory in bytes (ciphertext size)
    memory_bytes: int

    # Multiplicative depth consumed
    depth_consumed: int

    # Key switching operations (expensive)
    keyswitches: int

    # Rotations required
    rotations: int

    # TFHE bootstraps required
    bootstraps: int

    def __add__(self, other: 'OpCost') -> 'OpCost':
        """Sum costs (sequential execution)."""
        return OpCost(
            latency_ms=self.latency_ms + other.latency_ms,
            memory_bytes=max(self.memory_bytes, other.memory_bytes),  # Peak
            depth_consumed=self.depth_consumed + other.depth_consumed,
            keyswitches=self.keyswitches + other.keyswitches,
            rotations=self.rotations + other.rotations,
            bootstraps=self.bootstraps + other.bootstraps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'latency_ms': self.latency_ms,
            'memory_bytes': self.memory_bytes,
            'depth_consumed': self.depth_consumed,
            'keyswitches': self.keyswitches,
            'rotations': self.rotations,
            'bootstraps': self.bootstraps,
        }


class CostModel:
    """
    Cost model for hybrid CKKS-TFHE operations.

    Provides estimates based on typical GPU/CPU performance.
    Values can be calibrated for specific hardware.
    """

    def __init__(
        self,
        # CKKS parameters
        poly_degree: int = 16384,
        num_slots: int = 8192,

        # Hardware assumptions
        gpu_enabled: bool = True,

        # Calibration factors
        ckks_matmul_ms_per_slot: float = 0.001,  # 1μs per slot
        ckks_rotation_ms: float = 0.5,  # 500μs per rotation
        tfhe_bootstrap_ms: float = 10.0,  # 10ms per bootstrap (CPU)
        bridge_overhead_ms: float = 1.0,  # 1ms for scheme switch
    ):
        self.poly_degree = poly_degree
        self.num_slots = num_slots
        self.gpu_enabled = gpu_enabled

        # Per-operation costs
        self._costs = {
            OpType.CKKS_MATMUL: lambda size: OpCost(
                latency_ms=ckks_matmul_ms_per_slot * size,
                memory_bytes=poly_degree * 8 * 2,  # Two polynomials
                depth_consumed=1,
                keyswitches=0,
                rotations=0,  # MOAI eliminates rotations
                bootstraps=0,
            ),
            OpType.CKKS_ADD: lambda size: OpCost(
                latency_ms=0.01,  # ~10μs
                memory_bytes=poly_degree * 8 * 2,
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=0,
            ),
            OpType.CKKS_MUL: lambda size: OpCost(
                latency_ms=0.1,  # ~100μs
                memory_bytes=poly_degree * 8 * 3,  # Grows before rescale
                depth_consumed=1,
                keyswitches=1,  # Relinearization
                rotations=0,
                bootstraps=0,
            ),
            OpType.CKKS_RESCALE: lambda size: OpCost(
                latency_ms=0.05,  # ~50μs
                memory_bytes=poly_degree * 8 * 2,
                depth_consumed=0,  # Consumes level, not depth
                keyswitches=0,
                rotations=0,
                bootstraps=0,
            ),
            OpType.CKKS_ROTATE: lambda size: OpCost(
                latency_ms=ckks_rotation_ms,
                memory_bytes=poly_degree * 8 * 2,
                depth_consumed=0,
                keyswitches=1,
                rotations=1,
                bootstraps=0,
            ),
            OpType.CKKS_PACK: lambda size: OpCost(
                latency_ms=0.1,
                memory_bytes=poly_degree * 8 * 2,
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=0,
            ),
            OpType.TFHE_BOOTSTRAP: lambda size: OpCost(
                latency_ms=tfhe_bootstrap_ms * size,  # Per element
                memory_bytes=1024 * size,  # LWE ciphertext
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=size,
            ),
            OpType.TFHE_LUT: lambda size: OpCost(
                latency_ms=tfhe_bootstrap_ms * size,  # LUT = bootstrap
                memory_bytes=1024 * size,
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=size,
            ),
            OpType.TFHE_COMPARE: lambda size: OpCost(
                latency_ms=tfhe_bootstrap_ms,  # One bootstrap
                memory_bytes=1024,
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=1,
            ),
            OpType.TFHE_MUX: lambda size: OpCost(
                latency_ms=0.1,  # No bootstrap for simple MUX
                memory_bytes=1024 * 2,
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=0,
            ),
            OpType.BRIDGE_CKKS_TO_TFHE: lambda size: OpCost(
                latency_ms=bridge_overhead_ms * size,
                memory_bytes=1024 * size,
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=0,
            ),
            OpType.BRIDGE_TFHE_TO_CKKS: lambda size: OpCost(
                latency_ms=bridge_overhead_ms * size,
                memory_bytes=poly_degree * 8 * 2,
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=0,
            ),
            OpType.BRIDGE_QUANTIZE: lambda size: OpCost(
                latency_ms=0.01,
                memory_bytes=0,  # In-place
                depth_consumed=0,
                keyswitches=0,
                rotations=0,
                bootstraps=0,
            ),
        }

    def estimate_op(self, op_type: OpType, size: int = 1) -> OpCost:
        """Estimate cost for a single operation."""
        return self._costs[op_type](size)

    def estimate_node(self, node: IRNode) -> OpCost:
        """Estimate cost for an IR node."""
        op_name = node.op_name

        # Map IR op to cost type
        op_map = {
            'ckks_matmul': OpType.CKKS_MATMUL,
            'ckks_add': OpType.CKKS_ADD,
            'ckks_mul': OpType.CKKS_MUL,
            'ckks_rescale': OpType.CKKS_RESCALE,
            'ckks_rotate': OpType.CKKS_ROTATE,
            'ckks_pack_moai': OpType.CKKS_PACK,
            'tfhe_lut': OpType.TFHE_LUT,
            'tfhe_compare': OpType.TFHE_COMPARE,
            'tfhe_mux': OpType.TFHE_MUX,
            'tfhe_bootstrap': OpType.TFHE_BOOTSTRAP,
            'ckks_quantize_to_int': OpType.BRIDGE_QUANTIZE,
            'ckks_to_tfhe': OpType.BRIDGE_CKKS_TO_TFHE,
            'tfhe_to_ckks': OpType.BRIDGE_TFHE_TO_CKKS,
            'ckks_apply_mask': OpType.CKKS_MUL,  # Essentially a multiply
        }

        op_type = op_map.get(op_name)
        if op_type is None:
            # Unknown op, return minimal cost
            return OpCost(0.01, 0, 0, 0, 0, 0)

        # Estimate size from node outputs
        outputs = node.get_outputs()
        size = 1
        if outputs:
            size = outputs[0].shape.numel

        return self.estimate_op(op_type, size)


def estimate_program_cost(program: IRProgram, cost_model: Optional[CostModel] = None) -> OpCost:
    """
    Estimate total cost for an IR program.

    Args:
        program: The IR program
        cost_model: Cost model to use (default created if None)

    Returns:
        Total cost estimate
    """
    if cost_model is None:
        cost_model = CostModel()

    total = OpCost(0, 0, 0, 0, 0, 0)
    for node in program.nodes:
        total = total + cost_model.estimate_node(node)

    return total
