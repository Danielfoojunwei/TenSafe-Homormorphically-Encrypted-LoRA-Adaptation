# TenSafe HE-LoRA Benchmark Results

**Generated:** 2026-02-02
**Version:** 1.0.0
**Benchmark Mode:** Simulation (CKKS + TFHE hybrid)

## Executive Summary

This document presents canonical benchmark results for TenSafe's HE-LoRA inference system, comparing:

1. **Linear LoRA**: CKKS encryption with MOAI column packing (rotation-free)
2. **Gated LoRA**: Hybrid CKKS-TFHE with programmable bootstrapping for conditional adaptation

### Key Findings

| Metric | Linear LoRA | Gated LoRA | Notes |
|--------|-------------|------------|-------|
| Avg Latency (sim) | 637.6 μs | 77.8 μs | Per-token HE computation |
| Throughput (sim) | 1,569 ops/sec | 12,847 ops/sec | Simulation mode |
| Max Error | 0.19 | 0.06 | Relative to plaintext |
| Multiplicative Depth | 2 | 4 | CKKS levels consumed |
| Rotations | **0** | 0 | MOAI eliminates rotations |
| TFHE Bootstraps | 0 | 1 | Per-operation |

> **Note:** Simulation mode measures computational overhead without actual cryptographic operations. In production with real CKKS/TFHE backends, TFHE bootstrapping adds ~10-50ms per operation, making linear LoRA significantly faster for latency-critical applications.

## Detailed Results

### Linear LoRA (CKKS + MOAI)

The MOAI column packing optimization eliminates all rotations from CKKS matrix multiplication, achieving constant-time ciphertext-plaintext multiplication regardless of matrix dimensions.

| Hidden Size | LoRA Rank | Mean (μs) | P95 (μs) | Ops/sec | Max Error |
|-------------|-----------|-----------|----------|---------|-----------|
| 512 | 8 | 722.0 | 839.3 | 1,385 | 6.64e-02 |
| 512 | 16 | 411.1 | 664.0 | 2,432 | 9.09e-02 |
| 512 | 32 | 362.2 | 402.3 | 2,761 | 1.15e-01 |
| 1024 | 8 | 729.3 | 1001.3 | 1,371 | 1.13e-01 |
| 1024 | 16 | 823.6 | 1101.4 | 1,214 | 1.39e-01 |
| 1024 | 32 | 777.0 | 1034.5 | 1,287 | 1.87e-01 |

**HE Operation Profile:**
- CKKS Multiplications: 2 (one per matmul)
- CKKS Additions: 1 (final add)
- Rotations: **0** (MOAI eliminates all rotations)
- Rescales: 2 (after each multiplication)
- Multiplicative Depth: 2

### Gated LoRA (Hybrid CKKS-TFHE)

Gated LoRA uses TFHE programmable bootstrapping for discrete gate evaluation, enabling conditional adaptation: `y = Wx + g(x) * Δ(x)`.

| Hidden Size | LoRA Rank | Mean (μs) | P95 (μs) | Ops/sec | Bootstraps | Gate ON Rate |
|-------------|-----------|-----------|----------|---------|------------|--------------|
| 512 | 8 | 67.4 | 87.1 | 14,842 | 1 | 46% |
| 512 | 16 | 70.5 | 89.5 | 14,186 | 1 | 51% |
| 512 | 32 | 88.3 | 148.0 | 11,327 | 1 | 53% |
| 1024 | 8 | 69.4 | 83.4 | 14,406 | 1 | 48% |
| 1024 | 16 | 75.3 | 89.0 | 13,287 | 1 | 63% |
| 1024 | 32 | 96.2 | 129.4 | 10,394 | 1 | 55% |

**HE Operation Profile:**
- CKKS Operations: 10 (LoRA computation + gate pre-activation)
- TFHE LUT Evaluations: 1 (step function)
- Bridge Operations: 3 (quantize + CKKS→TFHE + TFHE→CKKS)
- Bootstraps: 1 per operation
- Multiplicative Depth: 4

### Precision Analysis

| Configuration | Linear Max Error | Gated Max Error | Notes |
|---------------|------------------|-----------------|-------|
| h=512, r=8 | 6.64e-02 | 1.36e-02 | |
| h=512, r=16 | 9.09e-02 | 2.00e-07 | Gated more precise |
| h=512, r=32 | 1.15e-01 | 3.82e-02 | |
| h=1024, r=8 | 1.13e-01 | 1.26e-07 | Gated more precise |
| h=1024, r=16 | 1.39e-01 | 4.59e-02 | |
| h=1024, r=32 | 1.87e-01 | 6.42e-02 | |

**Observations:**
- Gated LoRA achieves lower error in some configurations due to the discrete quantization providing exact gate evaluation
- Linear LoRA error increases with matrix dimensions due to accumulated CKKS approximation error
- TFHE gates are **exact** on discrete plaintexts (no approximation in LUT evaluation)

## Production Estimates

In production with real cryptographic backends:

### Linear LoRA (CKKS + GPU)
- Encryption: ~1-2 ms
- HE Computation: ~5-10 ms per token
- Decryption: ~1-2 ms
- **Total: ~7-14 ms per token**

### Gated LoRA (CKKS + TFHE)
- CKKS Computation: ~5-10 ms
- Bridge (CKKS→TFHE): ~1-5 ms
- TFHE Bootstrap: **~10-50 ms** (dominant cost)
- Bridge (TFHE→CKKS): ~1-5 ms
- **Total: ~20-70 ms per token**

> The TFHE programmable bootstrapping is the dominant cost in gated LoRA. Use gated adapters only when conditional adaptation is required.

## Recommendations

### Use Linear LoRA When:
- Latency is critical (real-time inference)
- Simple LoRA adaptation is sufficient
- Maximum throughput is required

### Use Gated LoRA When:
- Conditional adaptation is needed (e.g., task-specific routing)
- Discrete control flow is required
- Exact gate evaluation is necessary

## Benchmark Configuration

```json
{
  "hidden_sizes": [512, 1024],
  "lora_ranks": [8, 16, 32],
  "num_iterations": 100,
  "warmup_iterations": 10,
  "seed": 42
}
```

## Reproduction

To reproduce these benchmarks:

```bash
# Run canonical benchmark
python scripts/run_canonical_benchmark.py \
  --hidden-sizes 512 1024 \
  --lora-ranks 8 16 32 \
  --iterations 100 \
  --output benchmark_results.json
```

## References

- [HE-LoRA Microkernel Architecture](../../he_lora_microkernel/docs/ARCHITECTURE.md)
- [Hybrid CKKS-TFHE Compiler](../../he_lora_microkernel/hybrid_compiler/ARCHITECTURE.md)
- [MOAI Column Packing](../../he_lora_microkernel/docs/PACKING.md)
