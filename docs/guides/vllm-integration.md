# vLLM Integration Guide

**Version**: 4.1.0
**Last Updated**: 2026-02-08

This guide covers integrating TenSafe's privacy-preserving features with `vLLM` for high-throughput inference using encrypted LoRA adapters.

## Overview

TenSafe's vLLM integration provides:
- **HE-LoRA Forward Hooks**: Inject encrypted LoRA computation into the vLLM PagedAttention pipeline.
- **Zero-Rotation (MOAI)**: Enforcement of the MOAI contract for high-speed encrypted inference.
- **OpenAI-Compatible API**: Drop-in replacement server for standard OpenAI clients.
- **Evidence Fabric**: Hardware attestation (TEE) for the inference host.

## Quick Start

### Basic Engine Setup

```python
from tensorguard.backends.vllm import TenSafeAsyncEngine, TenSafeVLLMConfig

# Configure the engine for HE-LoRA
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B-Instruct",
    tssp_package_path="/path/to/adapter.tssp",
    enable_he_lora=True,
    he_scheme="ckks"
)

# Initialize engine
engine = TenSafeAsyncEngine(config)

# Generate stream
async for token in engine.generate_stream("Explain Homomorphic Encryption:"):
    print(token, end="", flush=True)
```

### OpenAI-Compatible Server

TenSafe provides a direct CLI and Python entry point for serving:

```bash
python -m tensorguard.backends.vllm.server \
    --model meta-llama/Llama-3-8B \
    --he-lora-path my_adapter.tssp \
    --port 8000
```

Then use the standard OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="tensafe-test")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## v4.1 Feature: Zero-Rotation (MOAI)

In version 4.1, TenSafe enforces the **Zero-Rotation (MOAI)** contract for all HE-LoRA operations.

### What is MOAI?
Rotary operations in standard Homomorphic Encryption (like CKKS) are computationally expensive and hard to parallelize. The MOAI (Microkernel Optimized AI) engine eliminates these rotations by using a specialized encoding scheme.

- **Benefit**: Up to 10x reduction in encrypted inference latency.
- **Requirement**: HE-LoRA adapters must be trained with MOAI-compatiblity (default in v4.1).

---

## v4.1 Feature: Evidence Fabric (TEE)

The vLLM engine now integrates with the **Evidence Fabric** for production deployments.

- **`require_attestation: True`**: When enabled, the engine will not load the TSSP master key unless a valid TEE hardware quote has been verified by the tenant's attestation service.
- **Hardware Support**: Compatible with Intel TDX, AMD SEV-SNP, and AWS Nitro Enclaves.

---

## Performance Tuning

1. **Speculative Batching**: Enable `speculative_model` to use a smaller, faster model to draft tokens, which are then verified by the large model + HE-LoRA.
2. **GPU Memory Utilization**: vLLM reserves memory for KV cache. Adjust `gpu_memory_utilization` (default 0.9) to balance throughput vs. model size.

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - vLLM Hook mechanism details.
- [Ray Train Guide](ray-train.md) - Training the adapters used here.
- [Observability Guide](observability.md) - Monitoring token-per-second and HE latency.
