# LoRA-Structure-Aware HE Optimizations: Research Analysis

## Executive Summary

This document analyzes potential **novel research contributions** by exploiting the specific structure of LoRA (Low-Rank Adaptation) for homomorphic encryption speedups. Unlike generic HE optimizations, these leverage the mathematical properties unique to LoRA.

---

## 1. The LoRA Structure Opportunity

### Standard LoRA Computation
```
Δy = (α/r) · A @ B @ x

Where:
  x ∈ ℝ^(batch × d)      # Input activations
  B ∈ ℝ^(r × d)          # Down-projection (d → r)
  A ∈ ℝ^(d × r)          # Up-projection (r → d)
  r << d                  # Rank constraint (typically r=16-64, d=4096-8192)
```

### Key Structural Properties

| Property | Implication for HE |
|----------|-------------------|
| **Low rank (r << d)** | Intermediate space is tiny (r values vs d values) |
| **Factorized** | A and B are separate, not combined |
| **Additive** | LoRA adds to base output (no interaction terms) |
| **Fixed at inference** | A, B are constants (plaintexts in Ct×Pt) |

---

## 2. Current Implementation Inefficiency

### Problem: Pre-computed AB Matrix
```python
# Current approach (from executor.py:204-208)
self._AB_combined = scaling * (A @ B)  # Shape: (d, d)
output = x @ AB_combined.T             # Standard d×d matmul
```

**Issues:**
1. Loses low-rank structure: Creates d×d matrix from two d×r matrices
2. Storage: O(d²) instead of O(2dr)
3. Computation: O(d²) multiplications instead of O(2dr)
4. Rotations: O(log(d/block_size)) instead of potentially O(log(r/block_size))

### Quantified Waste (Llama 8B: d=4096, r=16)
| Metric | Current (AB) | Optimal (A,B separate) | Ratio |
|--------|--------------|------------------------|-------|
| Matrix elements | 16.7M | 131K | **128×** |
| Multiplications/token | O(d²) = 16.7M | O(2dr) = 131K | **128×** |
| Memory | 64 MB | 0.5 MB | **128×** |

---

## 3. Proposed Optimization: Rank-Aware Packing

### Core Idea
**Pack and compute in the low-rank intermediate space, not the high-dimensional output space.**

### Current Packing (Hidden-Dimension Blocks)
```
Hidden dimension d=4096, batch=8, slots=8192

Current: Pack by hidden-dim blocks
┌─────────────────────────────────────────────────────┐
│ Block 0: channels [0:512]    × batch 8 = 4096 slots │
│ Block 1: channels [512:1024] × batch 8 = 4096 slots │
│ ... (8 blocks total)                                │
└─────────────────────────────────────────────────────┘
Rotations for accumulation: log₂(8) = 3 rotations
```

### Proposed: Pack by Rank-Dimension
```
Rank r=16, batch=8, slots=8192

Proposed: Pack entire rank dimension
┌─────────────────────────────────────────────────────┐
│ Intermediate: rank [0:16] × batch 8 = 128 slots     │
│ (fits entirely in ONE ciphertext with 8064 unused)  │
└─────────────────────────────────────────────────────┘
Rotations for rank accumulation: log₂(1) = 0 rotations!
```

### Algorithm: Two-Phase Low-Rank HE-LoRA

```
Phase 1: Down-projection (d → r)
─────────────────────────────────────────────────────
Input:  ct_x packed as [x₀, x₁, ..., x_{d-1}] × batch
Output: ct_intermediate packed as [z₀, z₁, ..., z_{r-1}] × batch

For each rank index i ∈ [0, r):
    # B[i, :] is a d-dimensional row vector
    ct_partial[i] = ct_x ⊙ encode(B[i, :])  # Ct×Pt
    ct_z[i] = rotate_and_sum(ct_partial[i]) # Accumulate d values

# Pack r results into single ciphertext
ct_intermediate = pack_rank_first([ct_z[0], ..., ct_z[r-1]])


Phase 2: Up-projection (r → d)
─────────────────────────────────────────────────────
Input:  ct_intermediate (r × batch packed)
Output: ct_output (d × batch packed)

For each output block b ∈ [0, num_blocks):
    # A[block_b, :] is block_size × r matrix
    ct_output[b] = 0
    for i in range(r):
        ct_output[b] += ct_intermediate[i] ⊙ encode(A[block_b, i])
```

### Rotation Analysis

| Phase | Current (AB combined) | Proposed (A,B separate) |
|-------|----------------------|------------------------|
| Phase 1 (d→r) | N/A (combined) | log₂(d/block) per rank = r·log₂(8) = 48 |
| Phase 2 (r→d) | N/A (combined) | 0 (broadcast, no accumulation) |
| Total | log₂(d/block)·2 = 6 | 48 + 0 = 48 |

**Wait - this is worse!** The naive two-phase is not better. We need a smarter approach.

---

## 4. Novel Optimization: Outer Product Decomposition

### Key Insight
LoRA can be written as a **sum of r rank-1 outer products**:

```
A @ B = Σᵢ aᵢ ⊗ bᵢ   (sum of r outer products)

Where:
  aᵢ = A[:, i]  ∈ ℝ^d   (i-th column of A)
  bᵢ = B[i, :] ∈ ℝ^d    (i-th row of B)
  aᵢ ⊗ bᵢ = rank-1 matrix
```

### HE Optimization via Outer Products

```
Δy = x @ (A @ B)ᵀ = x @ Σᵢ (aᵢ ⊗ bᵢ)ᵀ = Σᵢ (x · bᵢ) · aᵢ

Where:
  x · bᵢ = scalar (inner product, requires rotations to accumulate)
  (x · bᵢ) · aᵢ = scaled vector (just Ct×Pt, no rotations)
```

### Algorithm: Outer-Product HE-LoRA

```
Input: ct_x (encrypted x, d × batch)
Output: ct_delta (encrypted Δy, d × batch)

ct_delta = 0
for i in range(r):
    # Step 1: Compute scalar sᵢ = x · bᵢ (inner product)
    ct_prod = ct_x ⊙ encode(B[i, :])      # Ct×Pt, no rotation
    sᵢ = rotate_and_sum(ct_prod)          # log₂(num_blocks) rotations

    # Step 2: Broadcast sᵢ and multiply by aᵢ
    ct_broadcast = replicate(sᵢ, d)        # Replicate scalar to all slots
    ct_term = ct_broadcast ⊙ encode(A[:, i]) # Ct×Pt, no rotation

    # Step 3: Accumulate
    ct_delta = ct_delta + ct_term          # Ct+Ct, no rotation

return ct_delta
```

### Rotation Count Analysis

| Operation | Rotations | Count |
|-----------|-----------|-------|
| Inner product (rotate_and_sum) | log₂(num_blocks) | r times |
| Broadcast (replicate) | log₂(slots) | r times |
| Accumulation | 0 | - |
| **Total** | r · (log₂(blocks) + log₂(slots)) | - |

For d=4096, r=16, batch=8, slots=8192, blocks=8:
- Current: 2 × log₂(8) = 6 rotations
- Outer product: 16 × (3 + 13) = 256 rotations

**Still worse!** The broadcast is expensive.

---

## 5. Novel Optimization: Lazy Accumulation in Rank Space

### Key Insight
Don't broadcast immediately. **Accumulate in the rank dimension first, then broadcast once.**

```
Δy = Σᵢ sᵢ · aᵢ = A @ s   where s = [s₀, s₁, ..., s_{r-1}] = B @ x
```

The vector s has only r elements. If r fits in one SIMD block, we need **zero rotations** for the A @ s computation!

### Algorithm: Rank-Space Accumulation

```
Phase 1: Compute rank-space vector s = B @ x
──────────────────────────────────────────────
ct_s = new_ciphertext()  # Will hold r values

for i in range(r):
    ct_prod = ct_x ⊙ encode(B[i, :])
    sᵢ = rotate_and_sum(ct_prod)           # log₂(blocks) rotations
    insert_at_slot(ct_s, i, sᵢ)            # Pack into rank position

# ct_s now holds [s₀, s₁, ..., s_{r-1}] in first r slots


Phase 2: Expand to output space Δy = A @ s
──────────────────────────────────────────────
# A is (d × r), s is (r × 1)
# Use SIMD to compute all d outputs in parallel

ct_delta = 0
for i in range(r):
    # Extract sᵢ and broadcast to all d positions
    ct_si_broadcast = extract_and_broadcast(ct_s, i)  # log₂(d) rotations once
    ct_term = ct_si_broadcast ⊙ encode(A[:, i])       # Ct×Pt, no rotation
    ct_delta = ct_delta + ct_term
```

### But Wait - A Better Way: Column-Major A Packing

If we pack A in **column-major** order (all r values for each output position together):

```
A_packed[j] = [A[j,0], A[j,1], ..., A[j,r-1]]  for each output j

Then:
output[j] = dot(A_packed[j], s) = Σᵢ A[j,i] · sᵢ
```

This is an inner product of length r. If r ≤ block_size, this needs only **log₂(r)** rotations!

---

## 6. THE KEY OPTIMIZATION: Rank-Block Packing

### Core Idea
Pack so that the **rank dimension is contiguous in SIMD slots**, not the hidden dimension.

### Current Packing (Hidden-First)
```
Slot layout: [h₀b₀, h₀b₁, ..., h₀b₇, h₁b₀, h₁b₁, ..., h₁b₇, ...]
             |---- block 0 ----|     |---- block 1 ----|

Hidden dim is split across blocks → needs rotations to accumulate
```

### Proposed Packing (Rank-First for Intermediate)
```
Intermediate slot layout: [r₀b₀, r₁b₀, ..., r₁₅b₀, r₀b₁, r₁b₁, ..., r₁₅b₁, ...]
                          |---- batch 0, all r ----|  |---- batch 1, all r ----|

Rank dim is contiguous per batch → r values together → minimal rotations
```

### Complete Algorithm

```
Step 1: Down-projection x → s (d → r)
───────────────────────────────────────
# For each batch element, compute r-dimensional intermediate

ct_intermediates = []
for block_idx in range(num_blocks):
    # Multiply x_block by B_block
    ct_prod = ct_x_block[block_idx] ⊙ encode(B[:, block_range])
    ct_intermediates.append(ct_prod)

# Accumulate across blocks (only log₂(num_blocks) rotations)
ct_s = tree_reduce(ct_intermediates)  # Result: r × batch packed


Step 2: Up-projection s → Δy (r → d)
───────────────────────────────────────
# ct_s has shape (r × batch) packed contiguously
# For r=16, batch=8: 128 slots, fits in one block!

ct_outputs = []
for out_block_idx in range(num_output_blocks):
    # A_block is (block_size × r)
    # Use Ct×Pt for each column of A_block
    ct_out_block = 0
    for rank_idx in range(r):
        # Rotate ct_s to align rank_idx with slot 0
        ct_s_rotated = rotate(ct_s, rank_idx * batch_size)
        # Multiply by A column
        ct_term = ct_s_rotated ⊙ encode(A[out_block_range, rank_idx])
        ct_out_block += ct_term

    ct_outputs.append(ct_out_block)

# Combine output blocks
ct_delta = pack_blocks(ct_outputs)
```

### Rotation Count: Rank-First Packing

| Phase | Operation | Rotations |
|-------|-----------|-----------|
| Down (d→r) | Block accumulation | log₂(num_blocks) |
| Up (r→d) | Rank iteration | r rotations (to align each rank) |
| Up (r→d) | Output blocks | 0 (Ct×Pt only) |
| **Total** | | log₂(blocks) + r |

For d=4096, r=16, batch=8, blocks=8:
- **Current**: 2 × log₂(8) = 6 rotations
- **Rank-First**: log₂(8) + 16 = 19 rotations

**Still worse!** The r rotations dominate.

---

## 7. THE BREAKTHROUGH: Hoisted Rank Rotation

### Key Insight
The r rotations in Phase 2 are **data-independent**. They only depend on A's structure, which is known at compile time.

**Optimization: Pre-rotate A's encoding instead of rotating ct_s!**

```
# Instead of:
for rank_idx in range(r):
    ct_s_rotated = rotate(ct_s, rank_idx * batch_size)  # Runtime rotation
    ct_term = ct_s_rotated ⊙ encode(A[..., rank_idx])

# Do this:
for rank_idx in range(r):
    # Pre-rotate A's encoding at compile time (FREE - it's plaintext!)
    A_prerotated = prerotate_encoding(A[..., rank_idx], -rank_idx * batch_size)
    ct_term = ct_s ⊙ A_prerotated  # No runtime rotation!
```

### Final Rotation Count: Hoisted Rank-First

| Phase | Operation | Rotations |
|-------|-----------|-----------|
| Down (d→r) | Block accumulation | log₂(num_blocks) |
| Up (r→d) | Rank iteration | **0** (hoisted to plaintext) |
| **Total** | | **log₂(blocks)** |

For d=4096, r=16, batch=8, blocks=8:
- **Current (AB combined)**: 2 × log₂(8) = 6 rotations
- **Rank-First + Hoisting**: log₂(8) = **3 rotations** (50% reduction!)

---

## 8. Summary of Novel Contribution

### Technique: Rank-Aware Packing with Hoisted Rotation

| Aspect | Description |
|--------|-------------|
| **Key insight** | LoRA's low-rank structure means intermediate has r << d values |
| **Packing change** | Pack rank dimension contiguously, not hidden dimension |
| **Hoisting** | Pre-rotate plaintext encodings to eliminate runtime rotations |
| **Result** | 50% rotation reduction (log₂(blocks) vs 2·log₂(blocks)) |

### Novelty Claim
1. **Not in prior HE-LoRA work**: Existing work uses standard matrix packing
2. **Not in generic HE optimization**: This exploits LoRA-specific rank structure
3. **Not in LoRA literature**: LoRA papers don't consider HE constraints

### Research Contribution
> "We observe that LoRA's low-rank constraint (r << d) creates an asymmetry exploitable in HE: the intermediate representation has only r values, which can be packed contiguously. By hoisting rank-indexed rotations to compile-time plaintext manipulation, we eliminate r rotations per LoRA layer, achieving 50% rotation reduction for typical configurations."

---

## 9. Implementation Roadmap

### Phase 1: Prototype
1. Modify `packer.py` to support RANK_FIRST packing strategy
2. Add plaintext pre-rotation utility
3. Implement two-phase LoRA scheduler

### Phase 2: Optimization
1. Fuse down-projection accumulation with rescale
2. Optimize A encoding pre-rotation cache
3. Handle edge cases (r > block_size)

### Phase 3: Evaluation
1. Benchmark rotation counts vs current approach
2. Measure end-to-end latency improvement
3. Compare across model sizes (r=16, 32, 64; d=2048, 4096, 8192)

---

## 10. Additional Research Directions

### A. Batch-Rank Joint Packing
Pack (batch × rank) together for maximum SIMD utilization.

### B. Speculative Low-Rank
For r very small (r ≤ 8), compute all rank contributions in parallel using SIMD lanes.

### C. Sparse LoRA in HE
If LoRA weights are sparse/structured, exploit sparsity for fewer Ct×Pt ops.

### D. Quantized LoRA in HE
Use lower CKKS precision for LoRA delta (it's additive, so errors don't compound).

### E. Progressive Rank Expansion
For gated LoRA, skip computing higher rank components if gate is 0.
