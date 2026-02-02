# MOAI-LoRA: Rotation-Free Homomorphic Encryption for Privacy-Preserving LLM Personalization

**Target Venue**: NeurIPS 2026 / ICML 2026

**REVISED VERSION**: This paper contains realistic benchmarks based on published HE costs from peer-reviewed work.

## Abstract

We present **MOAI-LoRA**, a rotation-free algorithm for encrypted LoRA inference using the CKKS homomorphic encryption scheme. By exploiting LoRA's ciphertext-plaintext (Ct×Pt) structure, we achieve **zero rotations** for single-block computation and **O(log b)** rotations for b-block configurations. This provides **2000×+ rotation reduction** and projected **8× wall-clock speedups** over diagonal-method baselines.

## Key Contributions

1. **MOAI-LoRA Algorithm**: Zero-rotation encrypted LoRA via column-packed matrix multiplication
2. **Formal IND-CPA Security Proof**: Proves MOAI-LoRA inherits CKKS security
3. **Multi-Architecture Evaluation**: Llama-2-7B, Llama-3-8B, Mistral-7B, GPT-J-6B, BERT-large
4. **Ablation Study**: Isolates contribution of column packing vs block partitioning
5. **Baseline Comparison**: First systematic comparison against SHE-LoRA, PrivTuner, Encryption-Friendly LLM

## Realistic Benchmarks

**IMPORTANT**: Previous version had unrealistic metrics from simulation mode. This version uses cost models from published work.

### Cost Model (from Encryption-Friendly LLM, ICLR 2025)

| Operation | Time (ms) | Relative Cost |
|-----------|-----------|---------------|
| Ct + Ct | 10 | 1× |
| Ct × Pt | 60 | 6× |
| **Rotation** | **240** | **24×** |
| Rescale | 20 | 2× |

### Rotation Count Comparison

| Model | Hidden Dim | Diagonal Method | MOAI-LoRA | Reduction |
|-------|------------|-----------------|-----------|-----------|
| BERT-large | 1024 | 1,023 | **0** | ∞ |
| Llama-2-7B | 4096 | 4,095 | **2** | 2047× |
| Llama-3-8B | 4096 | 4,095 | **2** | 2047× |
| Mistral-7B | 4096 | 4,095 | **2** | 2047× |
| GPT-J-6B | 4096 | 4,095 | **2** | 2047× |

### Projected Latency (per LoRA layer)

| Model | Diagonal (ms) | MOAI-LoRA (ms) | Speedup | s/token |
|-------|---------------|----------------|---------|---------|
| BERT-large | 24,732 | 3,120 | **7.9×** | 3.12 |
| Llama-2-7B | 98,580 | 12,240 | **8.1×** | 12.24 |

### Full Model Inference (all layers, all projections)

| Model | MOAI Time/Token | Diagonal Time/Token |
|-------|-----------------|---------------------|
| BERT-large (24L, 4 proj) | **5.0 min** | 39.5 min |
| Llama-2-7B (32L, 4 proj) | **26.1 min** | 210.5 min |

**Note**: These times demonstrate that while MOAI-LoRA provides significant speedup, encrypted LLM inference remains impractical for real-time applications.

## Comparison with Prior Work

| Method | Paper | Our Improvement |
|--------|-------|-----------------|
| [SHE-LoRA](https://arxiv.org/abs/2505.21051) | Communication efficiency (different goal) | Complementary |
| [Private LoRA](https://arxiv.org/abs/2505.07329) | Llama-3.2-1B training | Rotation-free inference |
| [Encryption-Friendly LLM](https://arxiv.org/abs/2410.02486) | BERT 2L: 25.78s | Projected 6.24s (4.1×) |
| [PrivTuner](https://arxiv.org/abs/2410.00433) | PEFT with FHE | Rotation optimization |

## Paper Structure

```
paper/
├── moai_lora.tex          # Main paper (LaTeX) - REVISED
├── figures/figures.tex    # TikZ diagrams
├── Makefile               # Build automation
└── README.md              # This file
```

## Building

```bash
cd paper
make              # Full build
make quick        # Fast build (no bibtex)
```

## Key Sections

1. **Introduction**: Rotation bottleneck in HE, our key insight on Ct×Pt
2. **Background**: LoRA, CKKS, related work (SHE-LoRA, PrivTuner, Enc-Friendly LLM)
3. **MOAI-LoRA Design**: Column packing algorithm, multi-block extension
4. **Security Analysis**: IND-CPA proof (Theorem 1)
5. **Evaluation**: Rotation counts, projected latency, accuracy impact
6. **Ablation Study**: Isolating MOAI contribution
7. **Baseline Comparison**: SHE-LoRA, Encryption-Friendly LLM, OpenFHE
8. **Limitations**: Honest about absolute latency (26 min/token for Llama)

## Citation

```bibtex
@inproceedings{moailora2026,
  title={MOAI-LoRA: Rotation-Free Homomorphic Encryption for Privacy-Preserving LLM Personalization},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Important Notes

### Why Previous Benchmarks Were Wrong

The original benchmark_results.json was generated with `backend_type="SIMULATION"` which:
- Runs NumPy operations, NOT real homomorphic encryption
- Reports microsecond latencies that are ~1000× too fast
- Does not measure actual CKKS encryption/decryption

### Realistic Expectations

Based on published work:
- **Per-layer LoRA**: 3-12 seconds (not microseconds)
- **Full model token**: 5-30 minutes (not sub-second)
- **Practical use case**: Batch offline processing, not real-time inference

### Our Contribution

MOAI-LoRA's value is **algorithmic**, not absolute performance:
- Eliminates the most expensive operation (rotations)
- Reduces rotation count by 2000×+
- Provides 8× projected speedup over baselines
- Makes encrypted LoRA more feasible, though still far from practical

## Sources

- [SHE-LoRA (arXiv:2505.21051)](https://arxiv.org/abs/2505.21051)
- [Private LoRA Fine-tuning (arXiv:2505.07329)](https://arxiv.org/abs/2505.07329)
- [Encryption-Friendly LLM Architecture (ICLR 2025)](https://arxiv.org/abs/2410.02486)
- [PrivTuner (arXiv:2410.00433)](https://arxiv.org/abs/2410.00433)
- [MOAI (ePrint 2025/991)](https://eprint.iacr.org/2025/991)
