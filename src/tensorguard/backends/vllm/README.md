# TenSafe vLLM Backend

The TenSafe vLLM backend provides high-throughput, privacy-preserving inference support by integrating `vllm` with homomorphically encrypted LoRA (HE-LoRA) adapters.

## Component Overview

- **`TenSafeAsyncEngine`**: A wrapper around the `AsyncLLMEngine` that manages model loading, HE-LoRA hook registration, and TSSP package verification.
- **`HELoRAHookManager`**: Handles the lifecycle of PyTorch forward hooks that inject encrypted LoRA computation into the model's linear layers.
- **`OpenAI Router`**: A FastAPI-based router that provides OpenAI-compatible endpoints for completions and chat.

## Key Features

1. **Zero-Rotation (MOAI)**: Built-in support for MOAI-enforced Zero-Rotation HE-LoRA, eliminating rotation overhead in encrypted computation.
2. **Evidence Fabric**: Integration with the TEE Evidence Fabric for verified attestation of the inference environment.
3. **TSSP Verification**: Automatic signature and integrity verification for all loaded `.tssp` adapter packages.
4. **Metrics Correlation**: OpenTelemetry instrumentation for tracking HE operation count and latency alongside standard throughput metrics.

## Configuration

The backend is configured via `TenSafeVLLMConfig`:

```python
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B",
    enable_he_lora=True,
    he_lora_adapter_path="path/to/adapter.tssp",
    he_scheme="ckks",
    require_attestation=True
)
```

## Internal Dependencies

- `tensorguard.n2he`: Native HE backend for encrypted operations.
- `he_lora_microkernel`: High-performance execution core for HE-LoRA modules.
- `tensorguard.tgsp`: Secure package loading and verification.
