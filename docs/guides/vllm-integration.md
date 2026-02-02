# vLLM Integration Guide

**Version**: 4.0.0
**Last Updated**: 2026-02-02

This guide covers integrating TenSafe's privacy-preserving features with vLLM for high-throughput inference.

## Overview

TenSafe's vLLM integration provides:
- **HE-LoRA Hooks**: Apply encrypted LoRA transformations during inference
- **OpenAI-Compatible API**: Drop-in replacement for standard vLLM serving
- **Privacy-Preserving Inference**: Model weights remain encrypted
- **High Throughput**: PagedAttention and continuous batching

## Prerequisites

```bash
# Install TenSafe with vLLM support
pip install tensafe[vllm]>=4.0.0

# Or install dependencies separately
pip install vllm>=0.4.0 torch>=2.0
```

## Quick Start

### Basic Engine Setup

```python
from tensorguard.backends.vllm import TenSafeAsyncEngine, TenSafeVLLMConfig

# Configure the engine
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=1,  # GPUs per model instance
    gpu_memory_utilization=0.90,
    max_model_len=4096,

    # HE-LoRA settings
    enable_he_lora=True,
    he_lora_adapter_path="/path/to/encrypted_lora.tssp",
    he_lora_rank=16,
    he_lora_alpha=32,
)

# Initialize engine
engine = TenSafeAsyncEngine(config)
await engine.initialize()

# Generate text
request_id = "req-001"
prompt = "Explain privacy-preserving machine learning:"

async for output in engine.generate(prompt, request_id):
    print(output.outputs[0].text, end="", flush=True)
```

### OpenAI-Compatible Server

```python
from tensorguard.backends.vllm import TenSafeVLLMServer, TenSafeVLLMConfig

# Configure
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_he_lora=True,
    he_lora_adapter_path="/path/to/encrypted_lora.tssp",
)

# Create server
server = TenSafeVLLMServer(config)

# Run (compatible with uvicorn)
import uvicorn
uvicorn.run(server.app, host="0.0.0.0", port=8000)
```

Then use standard OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Can add auth if configured
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is homomorphic encryption?"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Configuration Reference

### TenSafeVLLMConfig

```python
@dataclass
class TenSafeVLLMConfig:
    # Model settings
    model: str                          # HuggingFace model ID or path
    tokenizer: Optional[str] = None     # Tokenizer (defaults to model)

    # Hardware
    tensor_parallel_size: int = 1       # GPUs for tensor parallelism
    gpu_memory_utilization: float = 0.9 # GPU memory fraction
    max_model_len: int = 4096           # Maximum sequence length

    # HE-LoRA
    enable_he_lora: bool = False        # Enable encrypted LoRA
    he_lora_adapter_path: str = ""      # Path to TSSP adapter
    he_lora_rank: int = 8               # LoRA rank
    he_lora_alpha: float = 16.0         # LoRA scaling
    he_lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Serving
    max_num_seqs: int = 256             # Max concurrent sequences
    block_size: int = 16                # KV cache block size

    # Privacy
    verify_tssp_signature: bool = True  # Verify adapter signature
    require_attestation: bool = False   # Require TPM attestation
```

## HE-LoRA Architecture

### How It Works

1. **Base Model Loading**: Standard HuggingFace model loaded into vLLM
2. **Hook Registration**: Custom forward hooks attached to target modules
3. **Encrypted Computation**: LoRA transformation computed on encrypted weights
4. **Output Combination**: Base output + scaled encrypted LoRA output

```
Input → Base Model → Base Output ─────────────────────────┐
                                                          │
                     ┌─────────────────────────────────┐  │
                     │        HE-LoRA Hook             │  │
                     │  ┌─────────┐     ┌─────────┐   │  │
Input ──────────────▶│  │ A (enc) │ ──▶ │ B (enc) │   │  │
                     │  └─────────┘     └─────────┘   │  │
                     │         HE Computation          │  │
                     └─────────────────────────────────┘  │
                                  │                       │
                                  ▼                       │
                           Scaled Output ◀────────────────┘
                                  │
                                  ▼
                           Final Output
```

### Target Modules

Common target modules for different architectures:

| Architecture | Recommended Targets |
|--------------|---------------------|
| Llama | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| GPT-2 | `c_attn`, `c_proj` |
| Mistral | `q_proj`, `v_proj`, `k_proj`, `o_proj` |
| Phi | `q_proj`, `v_proj`, `dense` |

### Loading Encrypted Adapters

```python
from tensorguard.backends.vllm import TenSafeAsyncEngine
from tensorguard.tgsp import TSSPService

# Verify and load TSSP package
tssp = TSSPService()
package = tssp.load_package("/path/to/adapter.tssp")
verification = tssp.verify_package(package)

if not verification.valid:
    raise ValueError(f"Invalid TSSP package: {verification.reason}")

# Engine automatically loads and registers hooks
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B",
    enable_he_lora=True,
    he_lora_adapter_path="/path/to/adapter.tssp",
)

engine = TenSafeAsyncEngine(config)
```

## Multi-GPU Inference

### Tensor Parallelism

```python
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,  # Split across 4 GPUs
    gpu_memory_utilization=0.95,
)
```

### Pipeline Parallelism (Coming Soon)

```python
# Future support
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-70B",
    pipeline_parallel_size=2,
    tensor_parallel_size=2,
    # Total: 4 GPUs (2 pipeline stages × 2 tensor parallel)
)
```

## API Endpoints

### Completions

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "prompt": "Privacy-preserving ML is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is differential privacy?"}
    ],
    "stream": true
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
# {"status": "healthy", "model": "meta-llama/Llama-3-8B-Instruct"}

curl http://localhost:8000/v1/models
# {"data": [{"id": "meta-llama/Llama-3-8B-Instruct", ...}]}
```

## Monitoring Integration

### OpenTelemetry Metrics

```python
from tensorguard.backends.vllm import TenSafeVLLMServer
from tensorguard.observability import setup_observability

# Setup observability first
setup_observability(
    service_name="tensafe-vllm",
    otlp_endpoint="http://otel-collector:4317",
)

# Server automatically exports metrics
server = TenSafeVLLMServer(config)
```

Exported metrics:
- `tensafe_vllm_requests_total` - Total requests by status
- `tensafe_vllm_tokens_generated` - Tokens generated
- `tensafe_vllm_latency_seconds` - Request latency histogram
- `tensafe_vllm_he_operations_total` - HE operations count
- `tensafe_vllm_kv_cache_utilization` - KV cache usage

## Performance Tuning

### Memory Optimization

```python
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B",

    # Increase GPU memory usage
    gpu_memory_utilization=0.95,

    # Smaller block size for better packing
    block_size=8,

    # Limit concurrent sequences
    max_num_seqs=128,
)
```

### Throughput Optimization

```python
config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B",

    # Increase concurrent sequences
    max_num_seqs=512,

    # Larger block size
    block_size=32,

    # Enable speculative decoding (experimental)
    # speculative_model="meta-llama/Llama-3-1B",
)
```

### HE-LoRA Performance

The HE-LoRA hooks add computational overhead. To minimize impact:

1. **Target fewer modules**: Only apply to `q_proj`, `v_proj`
2. **Use smaller rank**: Rank 8 instead of 16
3. **Batch HE operations**: Enable operation batching

```python
config = TenSafeVLLMConfig(
    enable_he_lora=True,
    he_lora_rank=8,  # Smaller rank
    he_lora_target_modules=["q_proj", "v_proj"],  # Fewer modules
)
```

## Troubleshooting

### Common Issues

**Out of Memory**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce `gpu_memory_utilization` or `max_num_seqs`

**TSSP Verification Failed**
```
ValueError: Invalid TSSP package: signature verification failed
```
Solution: Ensure adapter was signed with correct keys

**HE-LoRA Shape Mismatch**
```
RuntimeError: shape mismatch for LoRA adapter
```
Solution: Verify adapter was trained for the correct base model

### Debug Mode

```python
import logging
logging.getLogger("tensorguard.backends.vllm").setLevel(logging.DEBUG)

config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B",
    enable_he_lora=True,
    # Additional debug info logged
)
```

## Security Considerations

1. **TSSP Verification**: Always verify adapter signatures before loading
2. **Attestation**: Enable TPM attestation in production
3. **Network**: Use TLS for API endpoints
4. **Access Control**: Implement authentication for production deployments

```python
config = TenSafeVLLMConfig(
    verify_tssp_signature=True,  # Always verify
    require_attestation=True,     # TPM attestation
)
```

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [ray-train.md](ray-train.md) - Distributed training for adapters
- [observability.md](observability.md) - Monitoring setup
