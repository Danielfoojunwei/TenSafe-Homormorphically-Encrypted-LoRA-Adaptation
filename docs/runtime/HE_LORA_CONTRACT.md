# HE/LoRA Runtime Contract

**Date**: 2026-01-30
**Status**: DOCUMENTED

## Overview

The N2HE (Neural Network Homomorphic Encryption) runtime provides encrypted LoRA inference. This document specifies the numerical contracts, quantization strategy, and limitations.

## Runtime Modes

### Production Mode (NOT YET IMPLEMENTED)
- Requires actual HE library (e.g., SEAL, OpenFHE)
- Provides cryptographic security guarantees
- Currently NOT available in this release

### Toy Mode (Development/Testing)
- Enabled by: `TENSAFE_TOY_HE=1` environment variable
- **WARNING**: NOT cryptographically secure
- Useful for:
  - API testing
  - Shape validation
  - Integration testing
  - Performance profiling (approximate)

## Quantization Specification

### Float-to-Integer Quantization

The ToyN2HEScheme uses the following quantization:

```python
# Parameters
SCALE = 2^20  # 1,048,576
MODULUS_Q = 2^40  # For LWE ciphertext space

# Quantization (float -> int)
def quantize(x: float) -> int:
    return int(round(x * SCALE)) % MODULUS_Q

# Dequantization (int -> float)
def dequantize(x: int) -> float:
    if x > MODULUS_Q // 2:
        x -= MODULUS_Q
    return x / SCALE
```

### Error Bounds

| Operation | Expected Error | Notes |
|-----------|----------------|-------|
| Encrypt/Decrypt roundtrip | ±1e-5 | Quantization only |
| Addition | ±1e-5 | Linear accumulation |
| Multiplication | ±1e-4 | Quadratic growth |
| MatMul (N ops) | ±N×1e-4 | Accumulated error |

### Supported Ranges

```python
# Input range for reliable results
MIN_VALUE = -1000.0
MAX_VALUE = 1000.0

# Values outside this range may overflow
```

## Shape Preservation Contract

### Input/Output Shapes

All operations preserve batch and sequence dimensions:

```python
# Forward pass shape contract
input_shape:  (B, S, H)  # Batch, Sequence, Hidden
output_shape: (B, S, O)  # Batch, Sequence, Output

# Where:
# - B is preserved
# - S is preserved
# - H -> O depends on weight dimensions
```

### LoRA Delta Computation

```python
# LoRA forward: y = x @ W + scale * (x @ A @ B)
# Where:
#   x: (B, S, H)
#   W: (H, O) - frozen base weights (not encrypted)
#   A: (H, R) - LoRA down projection
#   B: (R, O) - LoRA up projection
#   scale: alpha / rank

# Encrypted computation:
#   encrypted_x @ encrypted_A @ encrypted_B
#   Result shape: (B, S, O)
```

## Determinism Guarantees

### With Fixed Seed

```python
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# Same inputs + same seed = same outputs
result1 = runtime.forward(input_ids)
result2 = runtime.forward(input_ids)
assert np.allclose(result1, result2)
```

### Sources of Non-Determinism

1. **Noise in real HE** - Intentional for security
2. **Float precision** - Platform-dependent
3. **Parallel execution** - Order-dependent accumulation

The toy mode adds NO noise unless explicitly configured.

## API Contract

### EncryptedLoRARuntime

```python
class EncryptedLoRARuntime:
    def __init__(self, config: LoRAConfig):
        """
        Initialize runtime.

        Raises:
            ToyModeNotEnabledError: If TENSAFE_TOY_HE not set
        """

    def register_adapter(
        self,
        adapter_id: str,
        lora_a: np.ndarray,  # Shape: (hidden, rank)
        lora_b: np.ndarray,  # Shape: (rank, output)
    ) -> None:
        """
        Register LoRA adapter weights.

        Raises:
            ValueError: If shapes don't match config
        """

    def forward(
        self,
        adapter_id: str,
        activations: np.ndarray,  # Shape: (batch, seq, hidden)
    ) -> np.ndarray:  # Shape: (batch, seq, output)
        """
        Compute encrypted LoRA delta.

        Raises:
            KeyError: If adapter_id not registered
            ValueError: If activation shape invalid
        """
```

### Key Bundle API

```python
class HEKeyBundle:
    def export_public_key(self) -> Dict[str, Any]:
        """Export for client distribution."""

    def export_evaluation_key(self) -> Dict[str, Any]:
        """Export for server-side computation."""

    def to_manifest_claims(self) -> Dict[str, Any]:
        """Generate TGSP manifest claims."""
```

## Test Coverage

### Shape Preservation Tests

```bash
pytest tests/n2he/test_n2he_adapter.py -k "shape" -v
```

### Correctness Tests

```bash
TENSAFE_TOY_HE=1 pytest tests/n2he/ -v
```

### Determinism Tests

```bash
TENSAFE_TOY_HE=1 pytest tests/n2he/test_n2he_core.py -k "determinism" -v
```

## Performance Benchmarks

### Running Benchmarks

```bash
TENSAFE_TOY_HE=1 python -c "
from tensorguard.n2he.benchmark import N2HEBenchmarkRunner
runner = N2HEBenchmarkRunner()
results = runner.run_all_benchmarks()
runner.print_report(results)
"
```

### Expected Performance (Toy Mode)

| Operation | Size | Latency (ms) | Throughput |
|-----------|------|--------------|------------|
| Encrypt | 768-dim | ~0.1 | ~10K ops/s |
| Add | 768-dim | ~0.05 | ~20K ops/s |
| Multiply | 768-dim | ~0.2 | ~5K ops/s |
| MatMul | 768x64 | ~1.0 | ~1K ops/s |

Note: Real HE implementations will be 100-1000x slower.

## Limitations

### What IS Protected (in production mode)
- Activations during computation
- LoRA deltas during inference
- Key material

### What is NOT Protected
- Model architecture (public)
- Base model weights (frozen, public)
- Output logits (returned to client)
- LoRA adapter dimensions (visible in manifest)

### Known Limitations

1. **Toy mode is NOT secure** - For testing only
2. **No bootstrapping** - Limited multiplication depth
3. **Fixed precision** - May not suit all use cases
4. **Batch processing** - Required for efficiency

## Migration Path

When production HE is available:

1. Replace `ToyN2HEScheme` with real implementation
2. Remove `TENSAFE_TOY_HE` requirement
3. Update performance expectations
4. Enable noise parameters
