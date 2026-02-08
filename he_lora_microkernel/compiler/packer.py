"""
SIMD Packing System for HE-LoRA Microkernel

This module implements MOAI-inspired aggressive SIMD packing strategies
for batch-first LoRA computation.

Key design principles:
  1. SIMD lanes = batch dimension (batch_size sequences in parallel)
  2. Blocked packing when hidden_size × batch_size > slot_count
  3. Automatic block size selection
  4. Minimize cross-block rotations

The packing layout is deterministically computed from (hidden_size, rank, batch_size)
and recompiled whenever these parameters change.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np

from .ckks_params import CKKSParams


class PackingStrategy(Enum):
    """Packing strategy for SIMD slots."""
    BATCH_FIRST = "batch_first"       # Slots: [seq0_val0, seq1_val0, ..., seq0_val1, seq1_val1, ...]
    CHANNEL_FIRST = "channel_first"   # Slots: [seq0_val0, seq0_val1, ..., seq1_val0, seq1_val1, ...]
    BLOCK_HYBRID = "block_hybrid"     # Hybrid for large dimensions


@dataclass(frozen=True)
class BlockSpec:
    """Specification for a single packing block."""
    block_id: int
    start_channel: int      # Start index in hidden dimension
    end_channel: int        # End index in hidden dimension
    num_channels: int       # Number of channels in this block
    slot_offset: int        # Starting slot in ciphertext
    num_slots: int          # Number of slots used

    def channel_range(self) -> range:
        """Get range of channels in this block."""
        return range(self.start_channel, self.end_channel)


@dataclass
class PackingLayout:
    """
    Complete packing layout for a LoRA computation.

    This describes how activations and weights are arranged in SIMD slots.
    The layout is fully deterministic given the input parameters.
    """
    # Configuration
    hidden_size: int
    lora_rank: int
    batch_size: int
    slot_count: int
    strategy: PackingStrategy

    # Block structure
    block_size: int
    num_blocks: int
    blocks: List[BlockSpec] = field(default_factory=list)

    # Packing info
    total_slots_used: int = 0
    slots_per_batch: int = 0
    padding_slots: int = 0

    # Rotation requirements (CRITICAL FOR MOAI)
    intra_block_rotations: int = 0      # Rotations within blocks
    cross_block_rotations: int = 0       # Rotations between blocks
    total_rotations_per_matmul: int = 0  # Total for one matrix multiply

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for artifact emission."""
        return {
            'hidden_size': self.hidden_size,
            'lora_rank': self.lora_rank,
            'batch_size': self.batch_size,
            'slot_count': self.slot_count,
            'strategy': self.strategy.value,
            'block_size': self.block_size,
            'num_blocks': self.num_blocks,
            'blocks': [
                {
                    'block_id': b.block_id,
                    'start_channel': b.start_channel,
                    'end_channel': b.end_channel,
                    'num_channels': b.num_channels,
                    'slot_offset': b.slot_offset,
                    'num_slots': b.num_slots,
                }
                for b in self.blocks
            ],
            'total_slots_used': self.total_slots_used,
            'slots_per_batch': self.slots_per_batch,
            'padding_slots': self.padding_slots,
            'intra_block_rotations': self.intra_block_rotations,
            'cross_block_rotations': self.cross_block_rotations,
            'total_rotations_per_matmul': self.total_rotations_per_matmul,
        }


@dataclass
class PackedTensor:
    """
    A tensor packed into SIMD slot layout.

    This represents how a tensor (activations or weights) is
    arranged in ciphertext slots for efficient HE computation.
    """
    # Original shape
    original_shape: Tuple[int, ...]

    # Packed data (for plaintexts, actual values; for ciphertexts, indices)
    data: np.ndarray

    # Layout reference
    layout: PackingLayout

    # Which dimension is packed
    packed_dim: int

    # Mapping from original indices to slot indices
    slot_map: Optional[Dict[Tuple[int, ...], int]] = None


# =============================================================================
# PACKING COMPUTATION
# =============================================================================

def compute_optimal_block_size(
    hidden_size: int,
    batch_size: int,
    slot_count: int,
    target_blocks: int = 8,
) -> int:
    """
    Compute optimal block size for packing.

    Goals:
      1. Fit batch_size × block_size slots per block
      2. Minimize number of blocks (fewer cross-block rotations)
      3. Keep block size as power of 2 for efficient rotations

    Args:
        hidden_size: Model hidden dimension
        batch_size: Batch size
        slot_count: Available SIMD slots
        target_blocks: Preferred number of blocks (if achievable)

    Returns:
        Optimal block size (power of 2)
    """
    # Maximum slots per block
    max_slots_per_block = slot_count  # Use full capacity (Zero Rotation critical)

    # Maximum channels per block given batch size
    max_channels_per_block = max_slots_per_block // batch_size

    # Find largest power of 2 <= max_channels_per_block
    block_size = 1
    while block_size * 2 <= max_channels_per_block:
        block_size *= 2

    # Cap at 4096 (larger models support) - enable Zero Rotation for 4096 hidden
    block_size = min(block_size, 4096)

    # Ensure at least 64 channels per block
    block_size = max(block_size, 64)

    return block_size


def compute_packing_layout(
    hidden_size: int,
    lora_rank: int,
    batch_size: int,
    params: CKKSParams,
    strategy: PackingStrategy = PackingStrategy.BATCH_FIRST,
) -> PackingLayout:
    """
    Compute complete packing layout for LoRA computation.

    This is the main entry point for the packing system. It determines:
      - Block size and count
      - Slot assignments for each block
      - Rotation requirements

    Args:
        hidden_size: Model hidden dimension (e.g., 4096)
        lora_rank: LoRA rank (e.g., 16)
        batch_size: Batch size (e.g., 8)
        params: CKKS parameters
        strategy: Packing strategy

    Returns:
        Complete PackingLayout

    Raises:
        ValueError: If layout cannot fit in slot count
    """
    slot_count = params.slot_count

    # Compute optimal block size
    block_size = compute_optimal_block_size(
        hidden_size, batch_size, slot_count
    )

    # Compute number of blocks needed
    num_blocks = math.ceil(hidden_size / block_size)

    # Compute slots needed
    slots_per_block = block_size * batch_size
    total_slots_needed = num_blocks * slots_per_block

    # Check if layout fits
    if total_slots_needed > slot_count:
        raise ValueError(
            f"Packing layout requires {total_slots_needed} slots but only "
            f"{slot_count} available. Reduce batch_size ({batch_size}) or "
            f"hidden_size ({hidden_size})."
        )

    # Build block specifications
    blocks = []
    slot_offset = 0

    for block_id in range(num_blocks):
        start_channel = block_id * block_size
        end_channel = min(start_channel + block_size, hidden_size)
        num_channels = end_channel - start_channel

        # Slots for this block
        block_slots = num_channels * batch_size

        blocks.append(BlockSpec(
            block_id=block_id,
            start_channel=start_channel,
            end_channel=end_channel,
            num_channels=num_channels,
            slot_offset=slot_offset,
            num_slots=block_slots,
        ))

        slot_offset += block_slots

    # Compute rotation requirements
    # For MOAI-style CPMM (Column-Packed Matrix Multiplication):
    #   - No rotations needed for Ct×Pt when using column packing
    #   - Cross-block accumulation requires log2(num_blocks) rotations
    intra_block_rotations = 0  # MOAI CPMM eliminates intra-block rotations
    cross_block_rotations = 0 if num_blocks == 1 else int(math.log2(num_blocks)) + 1

    # Total rotations for one matmul (A @ B @ x)
    # First matmul: B @ x -> r outputs, needs accumulation
    # Second matmul: A @ (B @ x) -> hidden_size outputs, needs accumulation
    rotations_for_B_x = int(math.log2(hidden_size // block_size)) if num_blocks > 1 else 0
    rotations_for_A_Bx = 0  # Output is smaller (rank), may not need rotation

    total_rotations = rotations_for_B_x + rotations_for_A_Bx + cross_block_rotations

    # Build layout
    layout = PackingLayout(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
        batch_size=batch_size,
        slot_count=slot_count,
        strategy=strategy,
        block_size=block_size,
        num_blocks=num_blocks,
        blocks=blocks,
        total_slots_used=slot_offset,
        slots_per_batch=slot_offset // batch_size,
        padding_slots=slot_count - slot_offset,
        intra_block_rotations=intra_block_rotations,
        cross_block_rotations=cross_block_rotations,
        total_rotations_per_matmul=total_rotations,
    )

    return layout


# =============================================================================
# ACTIVATION PACKING
# =============================================================================

def pack_activations(
    activations: np.ndarray,
    layout: PackingLayout,
) -> np.ndarray:
    """
    Pack batch activations into SIMD slot layout.

    Input shape: (batch_size, hidden_size)
    Output shape: (slot_count,) - single ciphertext

    The packing follows BATCH_FIRST strategy:
      Slot layout: [b0_c0, b1_c0, ..., b{N-1}_c0, b0_c1, b1_c1, ...]

    Args:
        activations: Input activations (batch_size, hidden_size)
        layout: Packing layout

    Returns:
        Packed activations for encryption
    """
    batch_size, hidden_size = activations.shape

    # Validate dimensions
    if batch_size != layout.batch_size:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, expected {layout.batch_size}"
        )
    if hidden_size != layout.hidden_size:
        raise ValueError(
            f"Hidden size mismatch: got {hidden_size}, expected {layout.hidden_size}"
        )

    # Initialize packed array with zeros (padding)
    packed = np.zeros(layout.slot_count, dtype=np.float64)

    # Pack each block
    for block in layout.blocks:
        for local_ch, global_ch in enumerate(block.channel_range()):
            if global_ch >= hidden_size:
                break

            # Slot positions for this channel across batch
            for b in range(batch_size):
                slot_idx = block.slot_offset + local_ch * batch_size + b
                packed[slot_idx] = activations[b, global_ch]

    return packed


def unpack_activations(
    packed: np.ndarray,
    layout: PackingLayout,
) -> np.ndarray:
    """
    Unpack SIMD slot layout back to batch activations.

    Input shape: (slot_count,)
    Output shape: (batch_size, hidden_size)

    Args:
        packed: Packed activations
        layout: Packing layout

    Returns:
        Unpacked activations
    """
    activations = np.zeros(
        (layout.batch_size, layout.hidden_size),
        dtype=np.float64
    )

    # Unpack each block
    for block in layout.blocks:
        for local_ch, global_ch in enumerate(block.channel_range()):
            if global_ch >= layout.hidden_size:
                break

            for b in range(layout.batch_size):
                slot_idx = block.slot_offset + local_ch * layout.batch_size + b
                activations[b, global_ch] = packed[slot_idx]

    return activations


# =============================================================================
# WEIGHT PACKING (FOR PLAINTEXT MULTIPLICATION)
# =============================================================================

@dataclass
class PackedLoRAWeights:
    """
    Pre-packed LoRA weights for efficient Ct×Pt multiplication.

    LoRA: Δy = A @ B @ x
    Where:
      - B: (rank, hidden_size) - down projection
      - A: (hidden_size, rank) - up projection
      - alpha is folded into A

    Weights are packed into plaintext blocks matching the activation layout.
    """
    # Original matrices
    A_original: np.ndarray  # (hidden_size, rank)
    B_original: np.ndarray  # (rank, hidden_size)

    # Pre-scaled A (alpha folded in)
    A_scaled: np.ndarray

    # Packed plaintext blocks for B (list of arrays, one per block)
    B_packed_blocks: List[np.ndarray]

    # Packed plaintext blocks for A
    A_packed_blocks: List[np.ndarray]

    # Combined AB for direct computation (optional optimization)
    AB_combined: Optional[np.ndarray] = None

    # Layout reference
    layout: PackingLayout = None


def pack_lora_weights(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
    layout: PackingLayout,
) -> PackedLoRAWeights:
    """
    Pack LoRA weight matrices for efficient HE computation.

    This pre-encodes the weights in a format that matches the
    activation packing layout, enabling rotation-free Ct×Pt multiplication.

    Args:
        A: Up-projection matrix (hidden_size, rank)
        B: Down-projection matrix (rank, hidden_size)
        alpha: LoRA alpha scaling factor
        layout: Packing layout

    Returns:
        PackedLoRAWeights ready for HE computation
    """
    hidden_size, rank = A.shape
    rank_b, hidden_size_b = B.shape

    # Validate dimensions
    if rank != rank_b:
        raise ValueError(f"Rank mismatch: A has {rank}, B has {rank_b}")
    if hidden_size != hidden_size_b:
        raise ValueError(
            f"Hidden size mismatch: A has {hidden_size}, B has {hidden_size_b}"
        )
    if hidden_size != layout.hidden_size:
        raise ValueError(
            f"Layout hidden size mismatch: {hidden_size} vs {layout.hidden_size}"
        )
    if rank != layout.lora_rank:
        raise ValueError(
            f"Layout rank mismatch: {rank} vs {layout.lora_rank}"
        )

    # Scale A with alpha/rank
    scaling = alpha / rank
    A_scaled = A * scaling

    # Pack B matrix blocks (for B @ x computation)
    B_packed_blocks = []
    for block in layout.blocks:
        # Extract columns of B for this block
        B_block = B[:, block.start_channel:block.end_channel]  # (rank, block_channels)

        # Pack for batch-first layout
        # Each column of B becomes a diagonal for Ct×Pt
        packed = np.zeros(layout.slot_count, dtype=np.float64)

        for local_ch in range(B_block.shape[1]):
            # For MOAI CPMM: replicate the column across batch slots
            for r in range(rank):
                # Place B[r, ch] at slots where x's ch values will be
                for b in range(layout.batch_size):
                    slot_idx = block.slot_offset + local_ch * layout.batch_size + b
                    if slot_idx < layout.slot_count:
                        packed[slot_idx] = B_block[r, local_ch]

        B_packed_blocks.append(packed)

    # Pack A matrix blocks (for A @ intermediate computation)
    A_packed_blocks = []
    for block in layout.blocks:
        # Extract rows of A for this block
        A_block = A_scaled[block.start_channel:block.end_channel, :]  # (block_channels, rank)

        packed = np.zeros(layout.slot_count, dtype=np.float64)

        for local_ch in range(A_block.shape[0]):
            for r in range(rank):
                for b in range(layout.batch_size):
                    slot_idx = block.slot_offset + local_ch * layout.batch_size + b
                    if slot_idx < layout.slot_count:
                        packed[slot_idx] = A_block[local_ch, r]

        A_packed_blocks.append(packed)

    # Optionally compute combined AB for direct single matmul
    # AB = A @ B has shape (hidden_size, hidden_size) - may be too large
    AB_combined = None
    if hidden_size <= 1024:  # Only compute for smaller models
        AB_combined = A_scaled @ B

    return PackedLoRAWeights(
        A_original=A,
        B_original=B,
        A_scaled=A_scaled,
        B_packed_blocks=B_packed_blocks,
        A_packed_blocks=A_packed_blocks,
        AB_combined=AB_combined,
        layout=layout,
    )


# =============================================================================
# SLOT MAPPING UTILITIES
# =============================================================================

def create_slot_map(layout: PackingLayout) -> Dict[Tuple[int, int], int]:
    """
    Create mapping from (batch_idx, channel_idx) to slot index.

    This is useful for debugging and verification.

    Args:
        layout: Packing layout

    Returns:
        Dictionary mapping (b, ch) -> slot_idx
    """
    slot_map = {}

    for block in layout.blocks:
        for local_ch, global_ch in enumerate(block.channel_range()):
            for b in range(layout.batch_size):
                slot_idx = block.slot_offset + local_ch * layout.batch_size + b
                slot_map[(b, global_ch)] = slot_idx

    return slot_map


def get_rotation_schedule_for_accumulation(
    num_elements: int,
    batch_size: int,
) -> List[int]:
    """
    Compute rotation schedule for accumulating elements.

    For MOAI-style tree reduction, we use powers of 2 rotations.

    Args:
        num_elements: Number of elements to accumulate
        batch_size: Batch size (rotation stride)

    Returns:
        List of rotation amounts for tree reduction
    """
    rotations = []
    stride = batch_size

    while stride < num_elements * batch_size:
        rotations.append(stride)
        stride *= 2

    return rotations


# =============================================================================
# PACKING VERIFICATION
# =============================================================================

def verify_packing_roundtrip(
    layout: PackingLayout,
    tolerance: float = 1e-10,
) -> bool:
    """
    Verify that pack/unpack roundtrip preserves values.

    Args:
        layout: Packing layout to verify
        tolerance: Maximum acceptable error

    Returns:
        True if verification passes
    """
    # Generate random activations
    activations = np.random.randn(layout.batch_size, layout.hidden_size)

    # Pack and unpack
    packed = pack_activations(activations, layout)
    unpacked = unpack_activations(packed, layout)

    # Check equality
    max_error = np.max(np.abs(activations - unpacked))
    return max_error < tolerance


def estimate_rotation_cost(layout: PackingLayout) -> Dict[str, int]:
    """
    Estimate rotation costs for a full LoRA computation.

    Args:
        layout: Packing layout

    Returns:
        Dictionary with rotation cost estimates
    """
    return {
        'intra_block': layout.intra_block_rotations,
        'cross_block': layout.cross_block_rotations,
        'per_matmul': layout.total_rotations_per_matmul,
        'per_lora_layer': layout.total_rotations_per_matmul * 2,  # Two matmuls
        'per_token_qkv': layout.total_rotations_per_matmul * 2 * 3,  # Q, K, V
        'per_token_qkvo': layout.total_rotations_per_matmul * 2 * 4,  # Q, K, V, O
    }
