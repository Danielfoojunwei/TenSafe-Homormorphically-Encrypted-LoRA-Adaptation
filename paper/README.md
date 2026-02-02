# MOAI-LoRA: Rotation-Free Homomorphic Encryption for Privacy-Preserving LLM Personalization

**Target Venue**: NeurIPS 2026 / ICML 2026

## Abstract

Low-Rank Adaptation (LoRA) has emerged as the dominant method for personalizing Large Language Models (LLMs), but deploying personalized adapters raises significant privacy concerns. We present **MOAI-LoRA**, a novel system for privacy-preserving LoRA inference using homomorphic encryption (HE). Our key insight is that LoRA's low-rank structure uniquely enables *rotation-free* encrypted computation, achieving **25× speedup** over rotation-based baselines with **411μs latency** for rank-16 LoRA.

## Key Contributions

1. **MOAI-LoRA Algorithm**: Zero-rotation encrypted LoRA via column-packed matrix multiplication
2. **Depth-Optimal Implementation**: Only multiplicative depth 2 required in CKKS
3. **Hybrid CKKS-TFHE Extension**: Encrypted gating for conditional LoRA architectures
4. **Production System**: GPU-accelerated microkernel integrated with vLLM (2,432 ops/sec)

## Paper Structure

```
paper/
├── moai_lora.tex          # Main paper (LaTeX)
├── figures/
│   └── figures.tex        # TikZ figures (standalone)
├── Makefile               # Build automation
└── README.md              # This file
```

## Building the Paper

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full latexmk

# macOS with Homebrew
brew install --cask mactex
```

### Compile

```bash
# Full build
make

# Quick build (no bibtex)
make quick

# Clean auxiliary files
make clean

# Build figures only
make figures
```

### Output

- `moai_lora.pdf` - Main paper
- `figures/*.pdf` - Individual figures

## Benchmark Data

All benchmark data is sourced from the TenSafe repository:

| Configuration | Latency (μs) | Throughput (ops/s) | Rotations |
|---------------|-------------|-------------------|-----------|
| h=512, r=8 | 722 | 1,385 | **0** |
| h=512, r=16 | 411 | 2,432 | **0** |
| h=512, r=32 | 362 | 2,761 | **0** |
| h=1024, r=8 | 729 | 1,371 | 1 |
| h=1024, r=16 | 824 | 1,214 | 1 |
| h=1024, r=32 | 777 | 1,287 | 1 |

### Comparison with Baselines

| Method | Latency (h=512, r=16) | Speedup |
|--------|----------------------|---------|
| Diagonal-HE | 10,275 μs | 1× |
| **MOAI-LoRA** | **411 μs** | **25×** |
| Unencrypted | 32 μs | 320× |

## Technical Highlights

### Why Zero Rotations?

Traditional HE matrix multiplication uses the diagonal method requiring O(n) rotations:
```
y = Σᵢ diag_i(M) ⊙ Rot_i(x)   # O(n) rotations
```

MOAI-LoRA exploits the Ct×Pt (ciphertext × plaintext) regime:
- Activations x are encrypted (user data)
- Weights A, B are plaintext (server owns them)
- Plaintext weights can be arbitrarily rearranged during encoding
- Column packing aligns weights with SIMD slot layout → **0 rotations**

### CKKS Parameters

**FAST Profile** (sufficient for LoRA):
- Polynomial degree: N = 16384 (8192 slots)
- Coefficient modulus: [60, 40, 40, 60] bits
- Scale: 2^40
- Max depth: 2
- Security: 128-bit

### Error Analysis

| (h, r) | Max Error | Mean Error | Impact on Accuracy |
|--------|-----------|------------|-------------------|
| (512, 8) | 0.0664 | 0.0426 | <0.1% |
| (512, 16) | 0.0909 | 0.0579 | <0.2% |
| (1024, 16) | 0.1391 | 0.0863 | <0.2% |

Errors are absorbed by LLM robustness; downstream task accuracy degrades by <0.3%.

## Citation

```bibtex
@inproceedings{moailora2026,
  title={MOAI-LoRA: Rotation-Free Homomorphic Encryption for Privacy-Preserving LLM Personalization},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Related Documentation

- [HE-LoRA Microkernel Architecture](../he_lora_microkernel/docs/ARCHITECTURE.md)
- [MOAI Packing Specification](../he_lora_microkernel/docs/PACKING.md)
- [Benchmark Results](../benchmark_results.json)
- [Hybrid Compiler Design](../he_lora_microkernel/hybrid_compiler/ARCHITECTURE.md)

## Reproducibility

All code is available in the TenSafe repository:

```bash
# Run benchmarks
python -m he_lora_microkernel.benchmark.run_benchmarks

# View results
cat benchmark_results.json | jq '.linear_lora'
```

## Submission Checklist

- [ ] Double-blind: Remove author names and affiliations
- [ ] Page limit: 9 pages (NeurIPS) / 8 pages (ICML)
- [ ] Supplementary: Code release plan, additional experiments
- [ ] Ethics statement: Privacy implications discussed
- [ ] Reproducibility: Benchmark scripts included

## License

Paper content: CC BY-NC-SA 4.0
Code: Apache 2.0 (see main repository)
