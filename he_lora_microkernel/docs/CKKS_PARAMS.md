# CKKS Parameter Guide

## Overview

The HE-LoRA microkernel uses CKKS (Cheon-Kim-Kim-Song) homomorphic encryption
for approximate arithmetic on encrypted data. This document describes the
CKKS parameters and their tradeoffs.

## Parameter Profiles

### FAST Profile

```python
FAST = {
    'poly_modulus_degree': 16384,     # N (ring dimension)
    'coeff_modulus_bits': [60, 40, 40, 60],
    'scale_bits': 40,
}
```

**Properties:**
- Slots: 8192 (N/2)
- Max depth: 2 (number of rescales before exhaustion)
- Scale: 2⁴⁰ ≈ 1.1 × 10¹²

**Best for:**
- Low-rank LoRA (r ≤ 16)
- Latency-sensitive workloads
- Quick testing

### SAFE Profile

```python
SAFE = {
    'poly_modulus_degree': 16384,
    'coeff_modulus_bits': [60, 45, 45, 45, 60],
    'scale_bits': 45,
}
```

**Properties:**
- Slots: 8192
- Max depth: 3
- Scale: 2⁴⁵ ≈ 3.5 × 10¹³

**Best for:**
- Higher precision requirements
- Larger ranks (r up to 32)
- When noise margin is critical

## Parameter Tradeoffs

### Polynomial Modulus Degree (N)

| N | Slots | Security | Performance |
|---|-------|----------|-------------|
| 4096 | 2048 | 128-bit | Fast |
| 8192 | 4096 | 128-bit | Moderate |
| 16384 | 8192 | 128-bit | Slow |
| 32768 | 16384 | 128-bit | Very slow |

**Tradeoff:** Larger N = more slots but slower operations.

### Coefficient Modulus

The coefficient modulus is a product of primes: Q = q₀ × q₁ × ... × qₖ

- First prime (q₀): Special prime for precision
- Middle primes: Consumed by rescaling
- Last prime (qₖ): Special prime for precision

**Security constraint:** Total bits ≤ security_bound(N)

| N | Max total bits (128-bit security) |
|---|-----------------------------------|
| 4096 | 109 |
| 8192 | 218 |
| 16384 | 438 |
| 32768 | 881 |

### Scale

The scale determines encoding precision:

```
encoded_value = round(plaintext_value × scale)
```

**Tradeoff:**
- Larger scale = better precision
- Scale must fit in intermediate primes
- After rescale, scale is divided by prime size

### Depth vs Scale vs Precision

Each rescale:
1. Divides scale by prime
2. Reduces precision slightly
3. Consumes one level

For LoRA (2 multiplications):
- FAST profile: depth 2 is sufficient
- SAFE profile: depth 3 provides margin

## Depth Requirements

### LoRA Computation Depth

```
Δy = A(Bx)

Operations:
1. Encrypt x           level 0
2. Bx = Σ B_i × x_i    level 0 → mul → rescale → level 1
3. A(Bx) = Σ A_i × Bx  level 1 → mul → rescale → level 2
4. Decrypt             level 2
```

Minimum depth: 2

### Blocked Packing Depth

With blocked packing, cross-block accumulation may require:
- Level alignment via modswitch
- No additional depth (additions don't increase level)

## Security Bounds

The microkernel uses parameters that provide 128-bit security:

```
Security level = f(N, log₂(Q))

For N = 16384:
  log₂(Q) ≤ 438 bits → 128-bit security
```

## No Bootstrapping

The microkernel explicitly **does not support bootstrapping**:

- Bootstrapping is slow (seconds per operation)
- LoRA computation fits within 2-3 levels
- Schedules that exceed depth are rejected at compile time

If you see this error:
```
ValueError: Level plan exceeds depth: X > Y. Bootstrapping NOT supported.
```

Solutions:
1. Use SAFE profile (more depth)
2. Reduce LoRA rank
3. Reduce batch size

## Slot Count Planning

### Batch-First Packing

Slots are allocated as:
```
slots_needed = batch_size × ceil(hidden_size / block_size) × block_size
```

### Example Calculations

| hidden_size | batch_size | block_size | slots_needed | Fits in 8192? |
|-------------|------------|------------|--------------|---------------|
| 512 | 8 | 512 | 4096 | ✓ |
| 1024 | 8 | 512 | 8192 | ✓ |
| 4096 | 8 | 512 | 32768 | ✗ |
| 4096 | 4 | 512 | 16384 | ✗ |
| 4096 | 1 | 512 | 4096 | ✓ |

For large hidden_size, reduce batch_size or use larger N.

## Profile Selection Guide

```python
def select_profile(hidden_size, rank, batch_size, precision):
    if precision == 'high':
        return SAFE

    if rank > 16:
        return SAFE  # Need extra depth margin

    slots_needed = estimate_slots(hidden_size, batch_size)
    if slots_needed > 8192:
        raise ValueError("Reduce batch_size or hidden_size")

    return FAST  # Default for most cases
```

## Advanced: Custom Profiles

For advanced users, custom profiles can be defined:

```python
custom_params = CKKSParams(
    poly_modulus_degree=32768,  # More slots
    coeff_modulus_bits=(60, 50, 50, 50, 50, 60),  # More depth
    scale_bits=50,
    profile=CKKSProfile.SAFE,
)
```

Ensure total bits stay within security bounds.
