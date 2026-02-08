"""
LoRA Intermediate Representation (IR) for HE-LoRA Microkernel

This module defines the IR for compiling LoRA adapters into efficient
HE execution schedules. The IR captures:

  1. LoRA computation structure: Δy = A(Bx)
  2. Packing requirements
  3. Operation sequence with explicit levels
  4. Rotation schedule
  5. Rescale/modswitch plan

The IR is designed to be:
  - Rank-aware (r is compile-time known)
  - Rotation-minimal (MOAI-aligned)
  - Deterministic (same inputs → identical schedules)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Any, Tuple, Set
import hashlib
import json

from .ckks_params import CKKSParams, CKKSProfile
from .packer import PackingLayout, PackingStrategy


# =============================================================================
# LoRA CONFIGURATION
# =============================================================================

class LoRATargets(Enum):
    """Which attention modules get LoRA adapters."""
    QKV = "qkv"      # Query, Key, Value only
    QKVO = "qkvo"    # Query, Key, Value, Output


@dataclass(frozen=True)
class LoRAConfig:
    """
    Compile-time LoRA configuration.

    All parameters are immutable and known at compile time.
    Changes to any parameter require recompilation.
    """
    # Model dimensions
    hidden_size: int        # e.g., 4096, 7168, 8192

    # LoRA parameters
    rank: int               # e.g., 8, 16, 32
    alpha: float            # Scaling factor (typically rank or 2*rank)
    targets: LoRATargets    # Which modules

    # Batch configuration
    batch_size: int         # e.g., 4, 8, 16, 32
    max_context_length: int # Hard limit on sequence length

    # CKKS profile
    ckks_profile: CKKSProfile

    def __post_init__(self):
        """Validate configuration."""
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive: {self.hidden_size}")
        if self.rank <= 0:
            raise ValueError(f"rank must be positive: {self.rank}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive: {self.batch_size}")
        if self.max_context_length <= 0:
            raise ValueError(f"max_context_length must be positive")

    @property
    def scaling_factor(self) -> float:
        """Compute alpha/rank scaling."""
        return self.alpha / self.rank

    @property
    def num_adapters(self) -> int:
        """Number of LoRA adapters per layer."""
        if self.targets == LoRATargets.QKV:
            return 3
        else:
            return 4

    def config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_str = f"{self.hidden_size}_{self.rank}_{self.alpha}_{self.targets.value}_"
        config_str += f"{self.batch_size}_{self.max_context_length}_{self.ckks_profile.value}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'hidden_size': self.hidden_size,
            'rank': self.rank,
            'alpha': self.alpha,
            'targets': self.targets.value,
            'batch_size': self.batch_size,
            'max_context_length': self.max_context_length,
            'ckks_profile': self.ckks_profile.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LoRAConfig':
        """Deserialize from dictionary."""
        return cls(
            hidden_size=d['hidden_size'],
            rank=d['rank'],
            alpha=d['alpha'],
            targets=LoRATargets(d['targets']),
            batch_size=d['batch_size'],
            max_context_length=d['max_context_length'],
            ckks_profile=CKKSProfile(d['ckks_profile']),
        )


# =============================================================================
# IR OPERATIONS
# =============================================================================

class IROpType(Enum):
    """Types of operations in the IR."""
    # Data movement
    ENCRYPT = auto()          # Encrypt activations
    DECRYPT = auto()          # Decrypt result
    ENCODE_PLAINTEXT = auto() # Encode weight as plaintext

    # Arithmetic
    MUL_PLAIN = auto()        # Ciphertext × Plaintext
    ADD = auto()              # Ciphertext + Ciphertext
    ADD_INPLACE = auto()      # Ciphertext += Ciphertext

    # Rotation (EXPENSIVE)
    ROTATE = auto()           # Rotate slots
    ROTATE_INPLACE = auto()   # Rotate in-place

    # Level management
    RESCALE = auto()          # Rescale after multiplication
    RESCALE_INPLACE = auto()  # Rescale in-place
    MODSWITCH = auto()        # Modulus switch
    MODSWITCH_TO_LEVEL = auto()  # Modswitch to specific level

    # Fused operations
    MUL_PLAIN_RESCALE = auto()     # Fused mul + rescale
    MUL_PLAIN_RESCALE_ADD = auto() # Fused mul + rescale + add

    # Control flow
    SYNC = auto()             # Synchronize streams
    BARRIER = auto()          # Memory barrier

    # Annotations
    LEVEL_CHECK = auto()      # Assert ciphertext level
    COMMENT = auto()          # Documentation


@dataclass
class IROperand:
    """Operand in an IR operation."""
    name: str           # Variable name (e.g., "ct_x", "pt_B_block0")
    operand_type: str   # "ciphertext", "plaintext", "scalar"
    level: Optional[int] = None    # Ciphertext level (if applicable)
    scale: Optional[float] = None  # Scale (if applicable)


@dataclass
class IROp:
    """Single operation in the IR."""
    op_type: IROpType
    result: Optional[IROperand] = None  # Output operand
    operands: List[IROperand] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Cost annotations
    rotation_count: int = 0
    keyswitch_count: int = 0
    rescale_count: int = 0

    def __str__(self) -> str:
        """String representation for debugging."""
        op_str = self.op_type.name
        if self.result:
            op_str = f"{self.result.name} = {op_str}"
        if self.operands:
            args = ", ".join(o.name for o in self.operands)
            op_str += f"({args})"
        if self.attributes:
            attrs = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
            op_str += f" [{attrs}]"
        return op_str


# =============================================================================
# IR BASIC BLOCK
# =============================================================================

@dataclass
class IRBasicBlock:
    """
    Basic block of IR operations.

    Operations within a block execute sequentially.
    Blocks can potentially be parallelized.
    """
    block_id: int
    name: str
    operations: List[IROp] = field(default_factory=list)

    # Dependencies for scheduling
    depends_on: Set[int] = field(default_factory=set)  # Block IDs
    produces: Set[str] = field(default_factory=set)    # Variable names

    # Aggregate costs
    total_rotations: int = 0
    total_keyswitches: int = 0
    total_rescales: int = 0

    def add_op(self, op: IROp) -> None:
        """Add operation to block."""
        self.operations.append(op)
        self.total_rotations += op.rotation_count
        self.total_keyswitches += op.keyswitch_count
        self.total_rescales += op.rescale_count

        if op.result:
            self.produces.add(op.result.name)


# =============================================================================
# LoRA IR MODULE
# =============================================================================

@dataclass
class LoRAIRModule:
    """
    Complete IR module for a LoRA adapter.

    This represents the compiled computation graph for one LoRA adapter
    (e.g., the Q projection's LoRA).
    """
    # Configuration
    config: LoRAConfig
    ckks_params: CKKSParams
    packing_layout: PackingLayout

    # IR structure
    blocks: List[IRBasicBlock] = field(default_factory=list)
    entry_block_id: int = 0

    # Variable registry
    variables: Dict[str, IROperand] = field(default_factory=dict)

    # Pre-encoded plaintexts (weight blocks)
    plaintext_registry: Dict[str, Any] = field(default_factory=dict)

    # Cost summary
    total_rotations: int = 0
    total_keyswitches: int = 0
    total_rescales: int = 0
    total_multiplications: int = 0
    total_additions: int = 0

    # Determinism
    module_hash: str = ""

    def compute_costs(self) -> None:
        """Recompute cost totals from blocks."""
        self.total_rotations = sum(b.total_rotations for b in self.blocks)
        self.total_keyswitches = sum(b.total_keyswitches for b in self.blocks)
        self.total_rescales = sum(b.total_rescales for b in self.blocks)

        self.total_multiplications = sum(
            1 for b in self.blocks for op in b.operations
            if op.op_type in (IROpType.MUL_PLAIN, IROpType.MUL_PLAIN_RESCALE,
                             IROpType.MUL_PLAIN_RESCALE_ADD)
        )
        self.total_additions = sum(
            1 for b in self.blocks for op in b.operations
            if op.op_type in (IROpType.ADD, IROpType.ADD_INPLACE,
                             IROpType.MUL_PLAIN_RESCALE_ADD)
        )

    def add_block(self, name: str) -> IRBasicBlock:
        """Create and add a new basic block."""
        block_id = len(self.blocks)
        block = IRBasicBlock(block_id=block_id, name=name)
        self.blocks.append(block)
        return block

    def register_variable(self, operand: IROperand) -> None:
        """Register a variable in the module."""
        self.variables[operand.name] = operand

    def compute_hash(self) -> str:
        """Compute deterministic hash of the module."""
        # Hash includes config, params, layout, and all operations
        hash_data = {
            'config': self.config.to_dict(),
            'ckks_params': self.ckks_params.to_dict(),
            'layout': self.packing_layout.to_dict(),
            'blocks': [
                {
                    'name': b.name,
                    'ops': [str(op) for op in b.operations],
                }
                for b in self.blocks
            ],
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        self.module_hash = hashlib.sha256(hash_str.encode()).hexdigest()
        return self.module_hash

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for artifact emission."""
        return {
            'config': self.config.to_dict(),
            'ckks_params': self.ckks_params.to_dict(),
            'packing_layout': self.packing_layout.to_dict(),
            'blocks': [
                {
                    'block_id': b.block_id,
                    'name': b.name,
                    'operations': [str(op) for op in b.operations],
                    'total_rotations': b.total_rotations,
                    'total_keyswitches': b.total_keyswitches,
                    'total_rescales': b.total_rescales,
                }
                for b in self.blocks
            ],
            'cost_summary': {
                'total_rotations': self.total_rotations,
                'total_keyswitches': self.total_keyswitches,
                'total_rescales': self.total_rescales,
                'total_multiplications': self.total_multiplications,
                'total_additions': self.total_additions,
            },
            'module_hash': self.module_hash,
        }


# =============================================================================
# IR BUILDER
# =============================================================================

class IRBuilder:
    """
    Builder for constructing LoRA IR modules.

    Provides a fluent API for creating IR operations with
    automatic cost tracking and level management.
    """

    def __init__(self, module: LoRAIRModule):
        self.module = module
        self.current_block: Optional[IRBasicBlock] = None
        self._var_counter = 0

    def set_block(self, block: IRBasicBlock) -> 'IRBuilder':
        """Set current basic block."""
        self.current_block = block
        return self

    def _fresh_var(self, prefix: str) -> str:
        """Generate fresh variable name."""
        self._var_counter += 1
        return f"{prefix}_{self._var_counter}"

    def _emit(self, op: IROp) -> IROp:
        """Emit operation to current block."""
        if self.current_block is None:
            raise RuntimeError("No current block set")
        self.current_block.add_op(op)
        if op.result:
            self.module.register_variable(op.result)
        return op

    # -------------------------------------------------------------------------
    # DATA MOVEMENT
    # -------------------------------------------------------------------------

    def encrypt(self, input_name: str) -> IROperand:
        """Emit encryption operation."""
        result = IROperand(
            name=self._fresh_var("ct"),
            operand_type="ciphertext",
            level=0,
            scale=self.module.ckks_params.scale,
        )
        self._emit(IROp(
            op_type=IROpType.ENCRYPT,
            result=result,
            operands=[IROperand(input_name, "plaintext")],
        ))
        return result

    def decrypt(self, ct: IROperand) -> IROperand:
        """Emit decryption operation."""
        result = IROperand(
            name=self._fresh_var("pt"),
            operand_type="plaintext",
        )
        self._emit(IROp(
            op_type=IROpType.DECRYPT,
            result=result,
            operands=[ct],
        ))
        return result

    def encode_plaintext(self, name: str, values: Any) -> IROperand:
        """Register pre-encoded plaintext."""
        result = IROperand(
            name=name,
            operand_type="plaintext",
            scale=self.module.ckks_params.scale,
        )
        self.module.plaintext_registry[name] = values
        self._emit(IROp(
            op_type=IROpType.ENCODE_PLAINTEXT,
            result=result,
            attributes={'values_ref': name},
        ))
        return result

    # -------------------------------------------------------------------------
    # ARITHMETIC
    # -------------------------------------------------------------------------

    def mul_plain(
        self,
        ct: IROperand,
        pt: IROperand,
        rescale: bool = True,
    ) -> IROperand:
        """Emit Ct×Pt multiplication."""
        if rescale:
            # Fused operation
            result_level = (ct.level or 0) + 1
            result = IROperand(
                name=self._fresh_var("ct"),
                operand_type="ciphertext",
                level=result_level,
                scale=self.module.ckks_params.scale,
            )
            self._emit(IROp(
                op_type=IROpType.MUL_PLAIN_RESCALE,
                result=result,
                operands=[ct, pt],
                rescale_count=1,
            ))
        else:
            result_scale = (ct.scale or 1.0) * (pt.scale or 1.0)
            result = IROperand(
                name=self._fresh_var("ct"),
                operand_type="ciphertext",
                level=ct.level,
                scale=result_scale,
            )
            self._emit(IROp(
                op_type=IROpType.MUL_PLAIN,
                result=result,
                operands=[ct, pt],
            ))
        return result

    def add(self, ct1: IROperand, ct2: IROperand) -> IROperand:
        """Emit Ct + Ct addition."""
        result = IROperand(
            name=self._fresh_var("ct"),
            operand_type="ciphertext",
            level=max(ct1.level or 0, ct2.level or 0),
            scale=ct1.scale,
        )
        self._emit(IROp(
            op_type=IROpType.ADD,
            result=result,
            operands=[ct1, ct2],
        ))
        return result

    def add_inplace(self, ct1: IROperand, ct2: IROperand) -> None:
        """Emit in-place addition: ct1 += ct2."""
        self._emit(IROp(
            op_type=IROpType.ADD_INPLACE,
            operands=[ct1, ct2],
        ))

    # -------------------------------------------------------------------------
    # ROTATION
    # -------------------------------------------------------------------------

    def rotate(self, ct: IROperand, steps: int) -> IROperand:
        """Emit rotation operation."""
        result = IROperand(
            name=self._fresh_var("ct"),
            operand_type="ciphertext",
            level=ct.level,
            scale=ct.scale,
        )
        self._emit(IROp(
            op_type=IROpType.ROTATE,
            result=result,
            operands=[ct],
            attributes={'steps': steps},
            rotation_count=1,
            keyswitch_count=1,
        ))
        return result

    def rotate_inplace(self, ct: IROperand, steps: int) -> None:
        """Emit in-place rotation."""
        self._emit(IROp(
            op_type=IROpType.ROTATE_INPLACE,
            operands=[ct],
            attributes={'steps': steps},
            rotation_count=1,
            keyswitch_count=1,
        ))

    # -------------------------------------------------------------------------
    # LEVEL MANAGEMENT
    # -------------------------------------------------------------------------

    def rescale(self, ct: IROperand) -> IROperand:
        """Emit rescale operation."""
        result = IROperand(
            name=self._fresh_var("ct"),
            operand_type="ciphertext",
            level=(ct.level or 0) + 1,
            scale=self.module.ckks_params.scale,
        )
        self._emit(IROp(
            op_type=IROpType.RESCALE,
            result=result,
            operands=[ct],
            rescale_count=1,
        ))
        return result

    def modswitch_to_level(self, ct: IROperand, target_level: int) -> IROperand:
        """Emit modswitch to specific level."""
        if ct.level == target_level:
            return ct

        result = IROperand(
            name=self._fresh_var("ct"),
            operand_type="ciphertext",
            level=target_level,
            scale=ct.scale,
        )
        switches = target_level - (ct.level or 0)
        self._emit(IROp(
            op_type=IROpType.MODSWITCH_TO_LEVEL,
            result=result,
            operands=[ct],
            attributes={'target_level': target_level, 'switches': switches},
        ))
        return result

    # -------------------------------------------------------------------------
    # ANNOTATIONS
    # -------------------------------------------------------------------------

    def comment(self, text: str) -> None:
        """Add comment to IR."""
        self._emit(IROp(
            op_type=IROpType.COMMENT,
            attributes={'text': text},
        ))

    def level_check(self, ct: IROperand, expected_level: int) -> None:
        """Assert ciphertext level for debugging."""
        self._emit(IROp(
            op_type=IROpType.LEVEL_CHECK,
            operands=[ct],
            attributes={'expected_level': expected_level},
        ))

    def sync(self) -> None:
        """Emit stream synchronization."""
        self._emit(IROp(op_type=IROpType.SYNC))


# =============================================================================
# COST PREDICTIONS
# =============================================================================

@dataclass
class CostPrediction:
    """Predicted costs for a compiled LoRA module."""
    # Per-token costs
    rotations_per_token: int
    keyswitches_per_token: int
    rescales_per_token: int
    multiplications_per_token: int
    additions_per_token: int

    # Per-layer costs (all adapters in a layer)
    rotations_per_layer: int
    keyswitches_per_layer: int

    # Memory estimates
    ciphertext_count: int
    plaintext_count: int
    estimated_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'per_token': {
                'rotations': self.rotations_per_token,
                'keyswitches': self.keyswitches_per_token,
                'rescales': self.rescales_per_token,
                'multiplications': self.multiplications_per_token,
                'additions': self.additions_per_token,
            },
            'per_layer': {
                'rotations': self.rotations_per_layer,
                'keyswitches': self.keyswitches_per_layer,
            },
            'memory': {
                'ciphertext_count': self.ciphertext_count,
                'plaintext_count': self.plaintext_count,
                'estimated_mb': self.estimated_memory_mb,
            },
        }


def predict_costs(
    config: LoRAConfig,
    layout: PackingLayout,
) -> CostPrediction:
    """
    Predict costs for LoRA computation before compilation.

    This is used for cost model validation and schedule selection.

    Args:
        config: LoRA configuration
        layout: Packing layout

    Returns:
        Predicted costs
    """
    # Rotations per matmul (from packing layout)
    rotations_per_matmul = layout.total_rotations_per_matmul

    # LoRA has two matmuls: B @ x and A @ (B @ x)
    rotations_per_adapter = rotations_per_matmul * 2

    # Number of adapters
    num_adapters = config.num_adapters

    # Rescales: one per multiplication
    rescales_per_adapter = 2  # One for each matmul

    # Multiplications: depends on blocks
    muls_per_adapter = layout.num_blocks * 2  # For each matmul

    # Additions: accumulate across blocks
    adds_per_adapter = layout.num_blocks - 1  # Tree reduction

    return CostPrediction(
        rotations_per_token=rotations_per_adapter,
        keyswitches_per_token=rotations_per_adapter,  # Each rotation needs keyswitch
        rescales_per_token=rescales_per_adapter,
        multiplications_per_token=muls_per_adapter,
        additions_per_token=adds_per_adapter,
        rotations_per_layer=rotations_per_adapter * num_adapters,
        keyswitches_per_layer=rotations_per_adapter * num_adapters,
        ciphertext_count=1 + layout.num_blocks,  # Input + intermediates
        plaintext_count=layout.num_blocks * 2,  # A and B blocks
        estimated_memory_mb=layout.total_slots_used * 8 * 10 / (1024 * 1024),  # Rough
    )
