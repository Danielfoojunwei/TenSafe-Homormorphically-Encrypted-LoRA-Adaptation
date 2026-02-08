# Performance Optimizations Guide

**Version**: 4.1.0
**Last Updated**: 2026-02-08

TenSafe v4.1 includes several breakthrough optimizations to minimize the latency and memory overhead of privacy-preserving machine learning.

## Zero-Rotation (MOAI)

The most significant optimization in v4.1 is the **Zero-Rotation (MOAI)** engine. 

### The Problem
Traditional CKKS encryption requires expensive "rotations" to align data for matrix multiplications. These rotations are mathematically complex and create a significant bottleneck in HE-LoRA.

### The MOAI Solution
The MOAI microkernel uses a specialized SIMD packing and encoding scheme that ensures weights and hidden states are perfectly aligned *without* requiring rotations.

- **Latency Reduction**: ~8x-10x improvement for large matrix products.
- **Hardware Acceleration**: Zero-Rotation enables highly efficient execution on NPUs and specialized HE accelerators.

---

## Speculative HE Batching

Inference on encrypted data is inherently slower than plaintext. TenSafe uses **Speculative Batching** to maximize throughput.

1. **Draft Step**: A small, fast plaintext model predicts the next N tokens.
2. **Verification Step**: The large model + HE-LoRA adapter verifies all tokens in parallel using a single HE forward pass.
3. **Acceptance**: Tokens matching the HE verification are accepted; the others are corrected.

This typically increases throughput by **2x-3x** depending on the task difficulty.

---

## Liger Kernels (Triton)

For training, TenSafe integrates **Liger Kernels**—a suite of Triton-based, high-performance kernels for LLM components.

- **Memory Efficiency**: Up to 40% reduction in peak VRAM usage.
- **Speed**: 20% faster than standard PyTorch/Eager execution for Llama-3 architectures.
- **Combined with HE**: Allows for larger effective batch sizes or higher LoRA ranks on smaller GPUs.

### How to Enable
```python
config = TenSafeRayConfig(
    use_liger_kernels=True,
    gradient_checkpointing=True
)
```

---

## Post-Quantum Cryptography (PQC) Optimizations

TenSafe uses a "Hybrid Architecture" for package verification:

- **Verification**: Uses **Dilithium3** for signatures.
- **Speedup**: Signatures are checked in a separate threadpool parallel to model loading, ensuring zero-latency impact on engine startup.
- **KEM**: Uses **Kyber768** for secure key exchange during worker registration in Ray.

## Related Documentation

- [vLLM Guide](vllm-integration.md) - Using speculative batching in production.
- [Ray Train Guide](ray-train.md) - Distributed throughput optimizations.
