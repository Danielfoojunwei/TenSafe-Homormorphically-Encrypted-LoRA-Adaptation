# SIMD Packing Strategy

## Overview

CKKS provides SIMD (Single Instruction, Multiple Data) slots that allow
parallel processing of multiple values in a single ciphertext. The HE-LoRA
microkernel uses **batch-first packing** to process entire batches in parallel.

## Slot Layout

### Batch-First Packing

```
Slots: [b0_c0, b1_c0, b2_c0, ..., b0_c1, b1_c1, b2_c1, ...]
        └────── channel 0 ──────┘ └────── channel 1 ──────┘
```

Where:
- `bX_cY` = batch X, channel Y
- Channels (hidden dimensions) are contiguous per batch element
- Batch elements are interleaved

### Example: batch_size=4, hidden_size=8

```
Channel 0: [b0_c0, b1_c0, b2_c0, b3_c0]  slots 0-3
Channel 1: [b0_c1, b1_c1, b2_c1, b3_c1]  slots 4-7
Channel 2: [b0_c2, b1_c2, b2_c2, b3_c2]  slots 8-11
...
Channel 7: [b0_c7, b1_c7, b2_c7, b3_c7]  slots 28-31
```

Total slots used: batch_size × hidden_size = 4 × 8 = 32

## Blocked Packing

When hidden_size × batch_size > slot_count, we use blocked packing:

```
┌─────────────────────────────────────────────────────────────┐
│                     Block 0                                  │
│  Channels [0, block_size)                                    │
│  Slots: batch_size × block_size                             │
├─────────────────────────────────────────────────────────────┤
│                     Block 1                                  │
│  Channels [block_size, 2×block_size)                        │
│  Slots: batch_size × block_size                             │
├─────────────────────────────────────────────────────────────┤
│                     ...                                      │
└─────────────────────────────────────────────────────────────┘
```

### Block Size Selection

The optimal block size is:
1. Power of 2 (for efficient rotations)
2. Maximum that fits: block_size × batch_size ≤ slot_count / 2
3. Typically 512 or 1024

```python
def compute_block_size(hidden_size, batch_size, slot_count):
    max_channels = slot_count // (2 * batch_size)
    block_size = 1
    while block_size * 2 <= max_channels:
        block_size *= 2
    return min(block_size, 1024)
```

### Number of Blocks

```
num_blocks = ceil(hidden_size / block_size)
```

## MOAI Column-Packed Matrix Multiplication

Traditional matrix-vector multiplication in HE requires rotations:

```
y = M × x

Standard approach:
  y = Σᵢ rotate(x, i) × diag(M, i)    # O(n) rotations
```

MOAI's **Column-Packed Matrix Multiplication (CPMM)** achieves rotation-free
Ct×Pt multiplication:

```
Column packing:
  M is pre-packed with columns aligned to SIMD slots
  y = x × column_packed(M)            # 0 rotations within block
```

### Why This Works for LoRA

LoRA uses Ct×Pt (ciphertext × plaintext):
- Activations (x) are encrypted
- Weights (A, B) are plaintext

Plaintext can be freely manipulated during encoding:
- Pack columns of weight matrix
- Align with activation layout
- Single Ct×Pt gives element-wise product

### Accumulation

After Ct×Pt, we need to accumulate across the rank dimension:

```
Bx = Σⱼ B[:, j] × x[j]
```

This uses tree reduction with log₂(channels) rotations.

## Slot Mapping

### Pack Function

```python
def pack_activations(activations, layout):
    """
    activations: (batch_size, hidden_size)
    returns: (slot_count,)
    """
    packed = zeros(slot_count)

    for block in layout.blocks:
        for local_ch, global_ch in enumerate(block.channel_range):
            for b in range(batch_size):
                slot_idx = block.slot_offset + local_ch * batch_size + b
                packed[slot_idx] = activations[b, global_ch]

    return packed
```

### Unpack Function

```python
def unpack_activations(packed, layout):
    """
    packed: (slot_count,)
    returns: (batch_size, hidden_size)
    """
    activations = zeros(batch_size, hidden_size)

    for block in layout.blocks:
        for local_ch, global_ch in enumerate(block.channel_range):
            for b in range(batch_size):
                slot_idx = block.slot_offset + local_ch * batch_size + b
                activations[b, global_ch] = packed[slot_idx]

    return activations
```

## Weight Packing

### B Matrix (Down Projection)

```
B: (rank, hidden_size)

For CPMM, pack each column of B to align with activation channels:
  B_packed[block][slot] = B[:, channel_for_slot]
```

### A Matrix (Up Projection)

```
A: (hidden_size, rank)

Pack rows of A to align with output channels:
  A_packed[block][slot] = A[channel_for_slot, :]
```

## Rotation Requirements

### Intra-Block Rotations

With MOAI CPMM: **0 rotations** within a block.

### Cross-Block Rotations

For accumulating across blocks:
```
rotations = log₂(num_blocks) if num_blocks > 1 else 0
```

### Example

| hidden_size | batch_size | block_size | blocks | cross-block rotations |
|-------------|------------|------------|--------|----------------------|
| 512 | 8 | 512 | 1 | 0 |
| 1024 | 8 | 512 | 2 | 1 |
| 4096 | 4 | 512 | 8 | 3 |

## Padding

### Zero Padding

Unused slots are zero-padded:
- Doesn't affect computation (0 × anything = 0)
- Slots beyond hidden_size are ignored in unpacking

### Batch Padding

For partial batches (actual_batch < batch_size):
```python
padded = zeros(batch_size, hidden_size)
padded[:actual_batch] = activations
# Process padded
result = result[:actual_batch]  # Remove padding
```

## Performance Implications

### Slot Utilization

Higher utilization = better efficiency:

```
utilization = (batch_size × hidden_size) / slot_count
```

Target: > 50% utilization

### Memory

Each ciphertext: ~N × log₂(Q) bits
For N=16384, Q=200 bits: ~400 KB per ciphertext

### Throughput

Batch-first packing achieves linear throughput scaling:
```
aggregate_throughput = batch_size × tokens_per_second
```
