# Rotation Minimization Policy

## Why Rotations Dominate HE Cost

In CKKS homomorphic encryption, **rotations are the most expensive operation**:

| Operation | Relative Cost | Key Required |
|-----------|---------------|--------------|
| Add | 1× | No |
| Ct×Pt Mul | 10× | No |
| Rescale | 5× | No |
| **Rotate** | **100×** | **Yes (Galois)** |
| Keyswitch | 100× | Yes |

Each rotation requires:
1. Key switching (expensive NTT operations)
2. Galois key access (memory bandwidth)
3. Large polynomial multiplications

## MOAI Alignment

This microkernel is inspired by MOAI (Matrix Operations for Approximate
computation with Integrity), which pioneered rotation-minimal HE algorithms.

### MOAI Key Insights

1. **Column Packing**: Eliminates rotations for Ct×Pt matrix-vector products
2. **Halevi-Shoup Diagonals**: O(√n) rotations for n×n matrices
3. **Consistent Packing**: No format conversions between layers

### Application to LoRA

LoRA's structure is naturally suited for MOAI optimization:
- Small rank (r << hidden_size)
- Ct×Pt regime (no Ct×Ct multiplication)
- Batch parallelism via SIMD

## Rotation Budget

### Default Budgets

```python
ROTATION_BUDGET = {
    'max_rotations_per_token': 16,      # R_max
    'max_rotations_per_layer': 64,      # Layer budget
    'max_rotations_qkv': 48,            # QKV target (3 adapters)
    'max_rotations_qkvo': 64,           # QKVO target (4 adapters)
}

KEYSWITCH_BUDGET = {
    'max_keyswitches_per_token': 16,    # K_max = R_max (1:1)
    'max_keyswitches_per_layer': 64,
}

RESCALE_BUDGET = {
    'max_rescales_per_token': 8,        # S_max
}
```

### CI Enforcement

CI **MUST FAIL** if any budget is exceeded:

```python
def ci_check(actual, budget):
    if actual > budget:
        raise CIFailure(f"Budget exceeded: {actual} > {budget}")
```

This ensures rotation counts don't regress.

## Rotation Sources

### 1. Cross-Block Accumulation

When hidden_size requires multiple blocks:

```
Rotations = log₂(num_blocks) if num_blocks > 1 else 0
```

| Blocks | Rotations |
|--------|-----------|
| 1 | 0 |
| 2 | 1 |
| 4 | 2 |
| 8 | 3 |

### 2. Tree Reduction

For summing across channels in a block:

```
# Traditional: O(n) rotations
result = sum(rotate(ct, i) for i in range(n))

# Tree reduction: O(log n) rotations
while n > 1:
    ct = ct + rotate(ct, n//2)
    n = n // 2
```

### 3. Final Output Collection

If output needs different packing than input: additional rotations.

**Mitigation**: Keep consistent packing throughout computation.

## Rotation Minimization Strategies

### Strategy 1: Maximize Block Size

Larger blocks = fewer cross-block rotations.

```python
# Prefer fewer, larger blocks
block_size = max_power_of_2_that_fits(slot_count // batch_size)
```

### Strategy 2: Batch-First Packing

Process entire batch in single ciphertext:
- All batch elements computed together
- Rotations amortized across batch

### Strategy 3: Pre-Combined Weights

For small hidden_size, pre-compute AB:
```
Δy = AB × x    # 1 Ct×Pt instead of 2
```

Reduces depth and operations (but increases plaintext size).

### Strategy 4: Lazy Accumulation

Delay cross-block accumulation:
```
# Instead of accumulating after each block:
results = [process_block(b) for b in blocks]
final = tree_reduce(results)  # Single accumulation pass
```

## Galois Key Requirements

Each rotation step requires a specific Galois key:

```python
# Required keys for rotation schedule
galois_keys = {abs(step) for step in rotation_schedule}
```

Generating unnecessary keys wastes:
- Key generation time
- GPU memory
- Key transfer bandwidth

### Minimal Key Sets

| Computation | Required Keys |
|-------------|---------------|
| 1 block | {} |
| 2 blocks | {block_size × batch_size} |
| 4 blocks | {bs, 2×bs} |
| 8 blocks | {bs, 2×bs, 4×bs} |

## Monitoring and Alerts

### Runtime Monitoring

```python
class RotationMonitor:
    def record_rotation(self):
        self.token_rotations += 1
        if self.token_rotations > self.budget:
            self.alert("BUDGET EXCEEDED")

    def end_token(self):
        # Check budget compliance
        assert self.token_rotations <= R_max
        self.history.append(self.token_rotations)
        self.token_rotations = 0
```

### Regression Detection

```python
def check_regression(current, baseline):
    if current > baseline * 1.1:  # 10% tolerance
        raise RegressionAlert(
            f"Rotation count increased: {baseline} -> {current}"
        )
```

## Profiling Rotation Costs

### Microbenchmark Rotation

```bash
python -m he_lora_microkernel.bench.bench_micro --benchmark rotate
```

### Analyze Token-Level Rotations

```python
from he_lora_microkernel.runtime import TelemetryCollector

collector = TelemetryCollector()
# ... run inference ...
averages = collector.get_token_averages()
print(f"Avg rotations/token: {averages['avg_rotations_per_token']}")
```

## Future Optimizations

### Rotation-Free Architectures

Research directions for zero-rotation HE:
1. Purely diagonal weight matrices
2. Element-wise operations only
3. Custom LoRA structures

### Hardware Acceleration

GPU-specific optimizations:
1. Batched key switching
2. Fused rotate-add kernels
3. Key caching strategies
