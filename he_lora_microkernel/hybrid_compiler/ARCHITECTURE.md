# Hybrid CKKS-TFHE Compiler Architecture

## Overview

The Hybrid CKKS-TFHE Compiler enables **conditional/gated LoRA inference** by combining two homomorphic encryption schemes:

- **CKKS**: Approximate arithmetic for linear algebra (MatMul, Add, Mul)
- **TFHE**: Exact arithmetic for discrete control flow (gates, thresholds)

This design leverages the strengths of each scheme while maintaining end-to-end encrypted computation.

## Core Design Principles

### 1. Scheme Separation

```
CKKS Domain                    TFHE Domain
─────────────                  ────────────
• Linear algebra only          • Discrete logic only
• No activations               • Programmable bootstrapping
• No conditionals              • Exact LUT evaluation
• MOAI column packing          • Limited to scalars/tiny vectors
```

### 2. Explicit Bridge Operations

**All scheme transitions go through mandatory bridge operations:**

```
CKKS → CKKSQuantizeToInt → CKKSToTFHE → TFHE
TFHE → TFHEToCKKS → CKKS
```

This ensures:
- Type safety at compile time
- Explicit quantization for precision control
- No accidental scheme mixing

### 3. Bootstrap Budget

TFHE programmable bootstrapping is expensive (~10-50ms per operation). The compiler enforces:

- **Maximum 2 bootstraps per layer** (configurable)
- Cost-aware scheduling to minimize bootstrap count
- Warnings when budget is exceeded

## Architecture Components

```
hybrid_compiler/
├── ir/                    # Intermediate Representation
│   ├── types.py          # Scheme-aware type system
│   ├── nodes.py          # IR operation nodes
│   └── validation.py     # Static validation rules
├── scheduler/            # Execution Planning
│   ├── cost_model.py     # Operation cost estimates
│   └── scheduler.py      # Phase-based scheduling
├── bridge/               # Scheme Conversion
│   ├── bridge.py         # CKKS↔TFHE conversion
│   └── quantizer.py      # Quantization helpers
├── tfhe_lut/             # LUT Library
│   └── lut_library.py    # Precomputed LUTs
├── gated_lora/           # Gated LoRA Implementation
│   ├── compiler.py       # Configuration → IR
│   └── executor.py       # IR execution
└── tests/                # Test Suite
    ├── test_ir.py        # IR validation tests
    ├── test_gated_lora.py # Functional correctness
    └── test_precision.py  # Precision benchmarks
```

## IR Type System

### Schemes

```python
class Scheme(Enum):
    CKKS = auto()  # Approximate arithmetic
    TFHE = auto()  # Exact discrete arithmetic
```

### Value Types

| Type | Scheme | Description |
|------|--------|-------------|
| `REAL_APPROX` | CKKS | Approximate real numbers |
| `BIT` | TFHE | Single bit {0, 1} |
| `INT_4` | TFHE | 4-bit integer |
| `INT_8` | TFHE | 8-bit integer |
| `INT_16` | TFHE | 16-bit integer |

### Constraints

- **TFHE vectors limited to ≤16 elements** (configurable via `Shape.MAX_TFHE_ELEMENTS`)
- **CKKS values require `REAL_APPROX` type**
- **TFHE values require discrete types**

## Gated LoRA Implementation

### Mathematical Form

```
y = Wx + g(x) · Δ(x)

where:
  Δ(x) = B(Ax)           # LoRA delta (CKKS)
  g(x) = step(wᵍᵀx + bᵍ) # Gate (TFHE)
```

### Execution Phases

```
Phase 1: CKKS_LORA_DELTA
  └─ x → [PackMOAI] → A·x → [Rescale] → B·(A·x) → [Rescale] → δ

Phase 2: CKKS_GATE_PRE
  └─ x → wᵍ·x + bᵍ → [Rescale] → z

Phase 3: BRIDGE_TO_TFHE
  └─ z → [QuantizeToInt] → [CKKSToTFHE] → z_tfhe

Phase 4: TFHE_GATE_EVAL
  └─ z_tfhe → [LUT: step] → g_tfhe  (1 bootstrap)

Phase 5: BRIDGE_TO_CKKS
  └─ g_tfhe → [TFHEToCKKS] → g_ckks

Phase 6: CKKS_APPLY_GATE
  └─ g_ckks · δ → [Rescale] → gated_δ

Phase 7: CKKS_FINAL_ADD
  └─ Wx + gated_δ → y
```

## Cost Model

| Operation | Estimated Latency | Notes |
|-----------|-------------------|-------|
| CKKS MatMul | 0.1 ms | With MOAI column packing |
| CKKS Add | 0.01 ms | |
| CKKS Rescale | 0.02 ms | |
| CKKS Rotate | 0.5 ms | Avoided with MOAI |
| TFHE LUT | 10.0 ms | Includes bootstrap |
| Bridge CKKS→TFHE | 1.0 ms | Requires re-encryption |
| Bridge TFHE→CKKS | 1.0 ms | Requires re-encryption |

## LUT Library

Precomputed lookup tables for TFHE evaluation:

| LUT | Function | Output |
|-----|----------|--------|
| `step` | step(x) = x ≥ 0 ? 1 : 0 | BIT |
| `sign` | sign(x) = x > 0 ? +1 : (x < 0 ? -1 : 0) | INT |
| `relu` | relu(x) = max(0, x) | INT |
| `clip` | clip(x, lo, hi) = max(lo, min(hi, x)) | INT |
| `argmax_2` | argmax(a, b) = a > b ? 0 : 1 | BIT |

All LUTs are **evaluated exactly** via TFHE programmable bootstrapping.

## Validation Rules

The compiler enforces these invariants at compile time:

1. **No direct scheme mixing**: CKKS ops cannot take TFHE inputs directly
2. **Mandatory bridges**: All scheme switches use explicit bridge ops
3. **TFHE size limits**: TFHE operations limited to small vectors
4. **Bootstrap budget**: Maximum bootstraps per program enforced
5. **Type consistency**: Value types match scheme requirements

## Example: Compiling Gated LoRA

```python
from he_lora_microkernel.hybrid_compiler import compile_gated_lora

# Compile configuration
program, plan = compile_gated_lora(
    hidden_size=4096,
    lora_rank=16,
    gate_type="step",
    quantization_bits=8,
)

# Validate
from he_lora_microkernel.hybrid_compiler.ir import validate_program
result = validate_program(program)
assert result.valid

# Execute (simulated)
from he_lora_microkernel.hybrid_compiler.gated_lora import GatedLoRAExecutor
executor = GatedLoRAExecutor(program, plan)
output = executor.execute_simulated(x, base_output, weights)
```

## Integration with N2HE

The `he_lora_microkernel.n2he` module provides:

- **FasterNTT**: CPU fallback for NTT operations
- **LUT Activation Engine**: GPU-accelerated LUT evaluation
- **Backend Abstraction**: GPU/CPU backend selection

```python
from he_lora_microkernel.n2he import (
    create_n2he_backend,
    N2HEBackendType,
)

# Create backend (auto-selects GPU if available)
backend = create_n2he_backend()

# Or explicitly request CPU fallback
backend = create_n2he_backend(N2HEBackendType.CPU_FASTERNTT)
```

## Security Considerations

1. **End-to-end encryption**: No plaintext exposed during computation
2. **Bridge re-encryption**: Scheme switches use re-encryption (client interaction required in production)
3. **No timing leaks**: Constant-time LUT evaluation via bootstrapping
4. **Key separation**: CKKS and TFHE use separate key hierarchies

## Performance Guidelines

### Minimize TFHE Operations

- Use TFHE only for discrete control (gates, thresholds)
- Avoid TFHE on large vectors
- Batch multiple gate evaluations if possible

### Leverage MOAI Column Packing

- Eliminates rotations in CKKS MatMul
- Use `use_moai_packing=True` (default)

### Control Quantization Precision

- 8-bit quantization balances precision vs latency
- Higher bits (10-12) for critical paths
- Lower bits (4-6) for gates where exact threshold matters

## Benchmark Results (Empirical)

The following benchmarks were collected on 2026-02-02 using simulation mode.

### Gated LoRA Performance

| Hidden Size | LoRA Rank | Mean Latency | P95 Latency | Throughput | Max Error |
|-------------|-----------|--------------|-------------|------------|-----------|
| 512 | 8 | 67.4 μs | 87.1 μs | 14,842 ops/sec | 1.36e-02 |
| 512 | 16 | 70.5 μs | 89.5 μs | 14,186 ops/sec | 2.00e-07 |
| 512 | 32 | 88.3 μs | 148.0 μs | 11,327 ops/sec | 3.82e-02 |
| 1024 | 8 | 69.4 μs | 83.4 μs | 14,406 ops/sec | 1.26e-07 |
| 1024 | 16 | 75.3 μs | 89.0 μs | 13,287 ops/sec | 4.59e-02 |
| 1024 | 32 | 96.2 μs | 129.4 μs | 10,394 ops/sec | 6.42e-02 |

**Operation Profile:**
- CKKS Operations: 10
- TFHE LUT Evaluations: 1
- Bridge Operations: 3
- Bootstraps per operation: 1
- Multiplicative Depth: 4
- Gate ON Rate: ~50% (balanced random inputs)

### Comparison with Linear LoRA

| Metric | Linear LoRA | Gated LoRA | Ratio |
|--------|-------------|------------|-------|
| Avg Latency (sim) | 637.6 μs | 77.8 μs | 8.2x |
| Multiplicative Depth | 2 | 4 | 2x |
| Rotations | 0 | 0 | - |
| Bootstraps | 0 | 1 | - |

> **Note:** In simulation mode, gated LoRA appears faster because TFHE bootstrap is simulated. In production with real cryptographic backends, TFHE bootstrap adds ~10-50ms per operation.

### Production Latency Estimates

| Operation | Simulation | Production (Est.) |
|-----------|------------|-------------------|
| Linear LoRA (CKKS) | 0.6 ms | 7-14 ms |
| Gated LoRA (Hybrid) | 0.08 ms | 20-70 ms |
| TFHE Bootstrap | ~0 μs | 10-50 ms |

**Recommendation:** Use linear LoRA for latency-critical paths. Reserve gated LoRA for applications requiring conditional adaptation.

## Future Directions

1. **Multi-party computation**: Threshold schemes for distributed inference
2. **Hardware acceleration**: FPGA/ASIC backends for bootstrapping
3. **Adaptive gating**: Learning gate configurations during training
4. **Pipeline parallelism**: Overlapping CKKS and TFHE computation
