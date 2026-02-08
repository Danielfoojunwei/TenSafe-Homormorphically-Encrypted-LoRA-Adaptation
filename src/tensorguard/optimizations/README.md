# TenSafe Performance Optimizations

This component hosts hardware-specific and algorithm-specific optimizations to reduce the overhead of privacy-preserving machine learning.

## Component Overview

- **Liger Kernels**: Integration of Triton-based Liger kernels for memory-efficient and faster training of large language models.
- **Microkernel Optimizations**: Performance tuning for the HE-LoRA microkernel, specifically targeting Zero-Rotation execution.
- **Speculative Batching**: Logic for optimizing encrypted inference throughput via speculative execution and adaptive batching.

## Key Optimizations

1. **Zero-Rotation (MOAI)**: Enforcement of the MOAI contract in HE-LoRA, eliminating expensive rotary operations in the encrypted domain.
2. **Gradient Checkpointing + Liger**: Simultaneous use of gradient checkpointing and memory-optimized Triton kernels to fit larger LoRA ranks into standard GPU memory.
3. **PQC Hybrid Signatures**: Optimized implementation of Dilithium3 and Ed25519 hybrid signatures for high-speed package verification.

## Supported Architectures

Currently optimized for:
- Llama-3 (8B, 70B)
- Mistral-v0.3
- Phi-3

## Dependencies

- `triton`: For Liger kernel execution.
- `he_lora_microkernel`: Interface for Zero-Rotation hardware acceleration.
