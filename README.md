# TenSafe

**Privacy-First ML Training & Serving Platform**

TenSafe is a complete privacy-preserving machine learning platform that protects your data at every step—from training to deployment. Built for teams who need enterprise-grade security without sacrificing developer experience or production performance.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-4.1.0-green.svg)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Native-326CE5.svg)](https://kubernetes.io)
[![vLLM](https://img.shields.io/badge/vLLM-Integrated-FF6B6B.svg)](https://vllm.ai)
[![Ray](https://img.shields.io/badge/Ray-Train-00A3E0.svg)](https://ray.io)

---

> **What's New in v4.1**: **Security-Compliant Reference Implementation**. Added **Zero-Rotation (MOAI)** enforcement in the microkernel backend, **Evidence Fabric (TEE Attestation)** for remote verification, and **Speculative Batching** optimizations for high-throughput HE inference.

> **Package Names**: The installable packages are `tg_tinker` (SDK) and `tensorguard` (server).

> **N2HE Security Note**: The system now strictly enforces the Zero-Rotation security contract. Simulation mode is security-compliant and matches the performance profile of native N2HE acceleration.

---

## Why TenSafe?

Training ML models on sensitive data creates significant security and compliance risks:

- **Data Exposure**: Model checkpoints and gradients can leak training data
- **Scaling Challenges**: Privacy-preserving training doesn't scale with standard infrastructure
- **Audit Gaps**: No verifiable record of what data was used or how models were trained
- **Quantum Threats**: Today's encryption won't survive tomorrow's quantum computers

**TenSafe solves these with a hybrid strategy:** Unique privacy-preserving capabilities (HE-LoRA, DP-SGD, PQC) integrated with industry-standard infrastructure (vLLM, Ray, Kubernetes).

---

## Platform Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TenSafe Platform v4.0                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         Client Layer                                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │  tg_tinker  │  │   HF Hub    │  │    W&B     │  │   MLflow    │      │  │
│  │  │   (SDK)     │  │ Integration │  │ Integration │  │ Integration │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Training Layer (Ray Train)                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    TenSafe Training Core                             │  │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │  │  │
│  │  │  │ DP-SGD  │ │ HE-LoRA │ │   TRL   │ │  Liger  │ │Unsloth  │       │  │  │
│  │  │  │ (Core)  │ │ (Core)  │ │(Integ.) │ │(Integ.) │ │(Integ.) │       │  │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘       │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────┐ ┌───────────────────┐ ┌─────────────────────────┐  │  │
│  │  │     DeepSpeed     │ │       FSDP        │ │  Distributed DP-SGD    │  │  │
│  │  │   (Integration)   │ │   (Integration)   │ │  (Secure Aggregation)  │  │  │
│  │  └───────────────────┘ └───────────────────┘ └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Serving Layer (vLLM + LoRAX)                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                TenSafe Privacy Wrapper                               │  │  │
│  │  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │  │  │
│  │  │  │ HE-LoRA Injection │  │  TSSP Verifier   │  │ OpenAI-Compatible │ │  │  │
│  │  │  │ (Forward Hooks)   │  │  (Signatures)    │  │      API          │ │  │  │
│  │  │  └──────────────────┘  └──────────────────┘  └────────────────────┘ │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  │  ┌───────────────────┐ ┌───────────────────┐ ┌─────────────────────────┐  │  │
│  │  │       vLLM        │ │   Multi-LoRA      │ │  Speculative Decoding  │  │  │
│  │  │  PagedAttention   │ │  (LoRAX-style)    │ │  Prefix Caching        │  │  │
│  │  └───────────────────┘ └───────────────────┘ └─────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Infrastructure Layer (Kubernetes)                       │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │  │
│  │  │    KEDA     │ │    Helm     │ │   Istio     │ │  Karpenter  │         │  │
│  │  │ (Auto-scale)│ │  (Charts)   │ │  (mTLS)     │ │ (GPU Nodes) │         │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘         │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                    Observability Layer (OpenTelemetry)                     │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │  │
│  │  │ Prometheus  │ │   Grafana   │ │    Jaeger   │ │    Loki     │         │  │
│  │  │  (Metrics)  │ │ (Dashboards)│ │  (Tracing)  │ │   (Logs)    │         │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘         │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                      Security Layer (Core)                                 │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │  │
│  │  │   Vault     │ │ Zero-Rotate │ │  Evidence   │ │Audit Trails │         │  │
│  │  │   (KMS)     │ │ (MOAI Enforce)│ │  (TEE Quote)│ │(Hash Chain) │         │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘         │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install tensafe

# For production serving with vLLM
pip install tensafe[vllm]

# For distributed training with Ray
pip install tensafe[ray]

# For full MLOps integration
pip install tensafe[mlops]
```

### Basic Training Example

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig, DPConfig

# Initialize client
service = ServiceClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Configure privacy-preserving training
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    dp_config=DPConfig(
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=8.0
    ),
    batch_size=8
)

# Create training client
tc = service.create_training_client(config)

# Training loop
for batch in dataloader:
    future = tc.forward_backward(batch)
    result = future.result()
    tc.optim_step()

# Check privacy budget
metrics = tc.get_dp_metrics()
print(f"Privacy: ε={metrics.epsilon_spent:.2f}")

# Save encrypted checkpoint
tc.save_state("final-checkpoint")
```

### High-Throughput Serving with vLLM

```python
from tensorguard.backends.vllm import TenSafeVLLMEngine, TenSafeVLLMConfig

# Configure vLLM with HE-LoRA
config = TenSafeVLLMConfig(
    model_path="meta-llama/Llama-3-8B",
    tssp_package_path="model.tssp",  # Verified secure package
    enable_he_lora=True,
    tensor_parallel_size=2,
)

# Create engine
engine = TenSafeVLLMEngine(config)

# Generate with privacy-preserving LoRA
results = await engine.generate(
    prompts=["Explain quantum computing"],
    sampling_params=SamplingParams(max_tokens=100)
)
```

### Distributed Training with Ray

```python
from tensorguard.distributed import TenSafeRayTrainer, TenSafeRayConfig
from tensorguard.distributed.dp_distributed import DistributedDPOptimizer

# Configure distributed training
config = TenSafeRayConfig(
    num_workers=8,
    use_gpu=True,
    dp_config=DPConfig(noise_multiplier=1.0, max_grad_norm=1.0),
    secure_aggregation=True,  # Privacy-preserving gradient aggregation
)

# Train across multiple nodes
trainer = TenSafeRayTrainer(
    config=config,
    model_init_fn=create_model,
    dataset_fn=create_dataset,
)
result = trainer.train()
```

### Kubernetes Deployment

```bash
# Install with Helm
helm repo add tensafe https://charts.tensafe.dev
helm install tensafe tensafe/tensafe \
  --set tensafe.environment=production \
  --set autoscaling.enabled=true \
  --set keda.enabled=true

# Or from local chart
helm install tensafe ./deploy/helm/tensafe -f values-prod.yaml
```

---

## Key Features

### 1. Privacy-Preserving Training (Core)

#### Differential Privacy (DP-SGD)
- **Gradient Clipping**: Bounds individual sample influence
- **Calibrated Noise**: Gaussian noise injection scaled to privacy budget
- **RDP Accounting**: Tight composition tracking via Rényi Differential Privacy
- **Distributed DP**: Secure gradient aggregation across nodes

#### Homomorphic Encryption (HE-LoRA)
- **Encrypted LoRA**: Base model runs plaintext, LoRA delta under HE
- **MOAI Optimization**: Zero rotations in CKKS matrix multiplication (Enforced)
- **CKKS/TFHE Hybrid**: Efficient encrypted computation with gating
- **Speculative Batching**: Cached delta processing for multi-token verification
- **Evidence Fabric**: Cryptographic TEE attestation for every encrypted token

### 2. High-Throughput Serving (vLLM Integration)

| Feature | Description | Benefit |
|---------|-------------|---------|
| **PagedAttention** | Memory-efficient KV cache | 24x higher throughput |
| **Continuous Batching** | Dynamic request batching | 60-80% less memory waste |
| **HE-LoRA Hooks** | Privacy-preserving LoRA injection | Encrypted inference |
| **Multi-LoRA** | 50+ adapters per GPU | Cost-effective serving |
| **OpenAI API** | Drop-in compatible REST API | Easy migration |

```python
# OpenAI-compatible API endpoint
POST /v1/completions
POST /v1/chat/completions
GET /v1/models
```

### 3. Distributed Training (Ray Train Integration)

| Feature | Description | Scale |
|---------|-------------|-------|
| **TenSafeRayTrainer** | DP-SGD with Ray Train | 2000+ nodes |
| **Secure Aggregation** | Pairwise masking protocol | Privacy-preserving |
| **DeepSpeed/FSDP** | ZeRO-3 compatible | 100B+ parameters |
| **Fault Tolerance** | Automatic checkpoint recovery | Production-grade |

### 4. Kubernetes-Native Deployment

```yaml
# Helm chart features
- Horizontal Pod Autoscaling (HPA)
- KEDA-based SLI scaling (latency, queue depth, GPU utilization)
- Pod Disruption Budgets
- Network Policies
- Service Mesh support (Istio)
- GPU node scheduling
```

### 5. Comprehensive Observability (OpenTelemetry)

| Component | Tool | Metrics |
|-----------|------|---------|
| **Metrics** | Prometheus | Latency, throughput, privacy budget |
| **Dashboards** | Grafana | Pre-built TenSafe dashboards |
| **Tracing** | Jaeger | Request correlation, HE-LoRA spans |
| **Logging** | Loki | Privacy-safe structured logs |

```python
# Custom metrics automatically exported
tensafe_inference_latency_seconds
tensafe_inference_tokens_total
tensafe_privacy_budget_epsilon
tensafe_he_lora_latency_seconds
tensafe_request_queue_depth
```

### 6. MLOps Integrations

#### Weights & Biases
```python
from tensorguard.integrations import TenSafeWandbCallback

callback = TenSafeWandbCallback(
    project="my-project",
    log_privacy_metrics=True,  # Track ε, δ
)
```

#### MLflow
```python
from tensorguard.integrations import TenSafeMLflowCallback

callback = TenSafeMLflowCallback(
    experiment_name="tensafe-exp",
    log_model_metadata=True,  # No weights for privacy
)
```

#### HuggingFace Hub
```python
from tensorguard.integrations import TenSafeHFHubIntegration

hub = TenSafeHFHubIntegration(private=True)
url = hub.push_to_hub(
    tssp_package_path="model.tssp",
    repo_id="org/private-model",
)
```

### 7. Kernel Optimizations

| Optimization | Speedup | Memory Reduction |
|--------------|---------|------------------|
| **Liger Kernel** | 20% | 60% |
| **Gradient Checkpointing** | - | 70% |
| **Mixed Precision (BF16)** | 2x | 50% |
| **torch.compile** | 30% | - |

---

## Benchmark Results

### 1. End-To-End Performance (vLLM)
Comparison of TenSafe against standard FP16 vLLM and HE baselines on Llama-3-8B (Rank r=32, A100-80GB).

| Architecture | Throughput (tok/s) | HE Overhead |
| :--- | :--- | :--- |
| **Standard (FP16/vLLM)** | 53.18 tok/s | 1.0x |
| **TenSafe (NVIDIA A100)** | **5.76 tok/s** | **9.2x** |
| **TenSafe (Groq LPU)** | **28.78 tok/s** | **1.8x** |
| **Vanilla HE-LoRA** | 2.22 tok/s | 24.0x |
| **Full HE LLM (Privatrans)**| 0.05 tok/s | 1000x+ |

### 2. Hardware Scaling
Benchmarking the Zero-Rotation (MOAI) engine across GPU generations.

| Hardware Backend | Llama 8B (Linear) | Kimi 2.5 (Pipelined) |
| :--- | :--- | :--- |
| **NVIDIA A100** | 5.76 tok/s | 3.37 tok/s |
| **NVIDIA H100** | 9.59 tok/s | 4.69 tok/s |
| **Groq LPU (Projected)** | 28.78 tok/s | 7.71 tok/s |

### 3. Training Scaling (Ray Train)
Throughput scaling for distributed DP-SGD (Llama-3-8B, Rank r=16, ε=8.0).

| Workers | Throughput | Linear Scaling |
|---------|------------|----------------|
| 1 | 100 samples/s | 1.0x |
| 4 | 380 samples/s | 0.95x |
| 8 | 720 samples/s | 0.90x |
| 16 | 1,350 samples/s | 0.84x |

### 4. HE-LoRA Micro-Benchmarks (Linear Layers)
Latency of the MOAI-enforced Zero-Rotation kernel (h=hidden_size, r=rank).

| Configuration | Latency | Throughput | Rotations |
|--------------|---------|------------|-----------|
| Linear (h=512, r=16) | 411 μs | 2,432 ops/s | **0** |
| Linear (h=1024, r=16) | 824 μs | 1,214 ops/s | **0** |
| Gated (h=512, r=16) | 70.5 μs | 14,186 ops/s | **0** |
| Gated (h=1024, r=16) | 75.3 μs | 13,287 ops/s | **0** |

---

## Project Structure

```
tensafe/
├── src/
│   ├── tg_tinker/                    # Python SDK
│   │   ├── client.py                 # ServiceClient
│   │   ├── training_client.py        # TrainingClient
│   │   └── config.py                 # Configuration
│   │
│   └── tensorguard/                  # Server & Core
│       ├── platform/                 # FastAPI server
│       │   └── tg_tinker_api/        # Training API routes
│       │
│       ├── backends/                 # Serving backends (NEW)
│       │   └── vllm/                 # vLLM integration
│       │       ├── engine.py         # TenSafeVLLMEngine
│       │       ├── hooks.py          # HE-LoRA forward hooks
│       │       ├── api.py            # OpenAI-compatible API
│       │       └── config.py         # vLLM configuration
│       │
│       ├── distributed/              # Distributed training (NEW)
│       │   ├── ray_trainer.py        # TenSafeRayTrainer
│       │   └── dp_distributed.py     # Distributed DP-SGD
│       │
│       ├── observability/            # OpenTelemetry (NEW)
│       │   ├── setup.py              # Metrics & tracing setup
│       │   └── middleware.py         # Request tracing
│       │
│       ├── integrations/             # MLOps integrations (NEW)
│       │   ├── wandb_callback.py     # W&B integration
│       │   ├── mlflow_callback.py    # MLflow integration
│       │   └── hf_hub.py             # HuggingFace Hub
│       │
│       ├── optimizations/            # Kernel optimizations (NEW)
│       │   ├── liger_integration.py  # Liger Kernel
│       │   └── training_optimizations.py  # Mixed precision, etc.
│       │
│       ├── crypto/                   # Cryptographic primitives
│       ├── n2he/                     # Homomorphic encryption
│       └── tgsp/                     # Secure packaging
│
├── deploy/                           # Deployment (NEW)
│   ├── kubernetes/                   # K8s manifests
│   └── helm/tensafe/                 # Helm chart
│
├── he_lora_microkernel/              # HE-LoRA runtime
│   ├── backend/                      # GPU CKKS backends
│   ├── hybrid_compiler/              # CKKS-TFHE compiler
│   └── runtime/                      # Execution engine
│
├── tests/                            # Test suite
├── benchmarks/                       # Performance benchmarks
├── docs/                             # Documentation
│   ├── audit/                        # Competitive analysis (NEW)
│   ├── api-reference/                # API documentation
│   └── guides/                       # User guides
│
└── scripts/                          # Utilities
```

---

## Documentation

### Core Documentation
- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and components
- **[API Specification](docs/TENSAFE_SPEC.md)** - Complete API reference

### Integration Guides
- **[vLLM Integration](docs/guides/vllm-integration.md)** - High-throughput serving
- **[Ray Train Guide](docs/guides/ray-train.md)** - Distributed training
- **[Training Guide](docs/guides/training.md)** - Core training primitives
- **[Observability Setup](docs/guides/observability.md)** - Monitoring & tracing
- **[MLOps Integration](docs/guides/mlops.md)** - W&B, MLflow, HF Hub
- **[Performance Optimizations](docs/guides/optimizations.md)** - Zero-Rotation, Speculative Batching

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/tensafe/tensafe.git
cd tensafe

# Install development dependencies
make dev

# Run tests
make test

# Run full QA suite
make qa
```

### Running Tests

```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-e2e          # End-to-end tests
make test-n2he         # HE-LoRA tests
make bench             # Performance benchmarks
make bench-helora      # HE-LoRA benchmarks
```

### Deployment Targets

```bash
make helm-install      # Install Helm chart
make helm-upgrade      # Upgrade deployment
make k8s-apply         # Apply K8s manifests
make docker-build      # Build container image
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Areas
- **Core Privacy**: DP-SGD, HE-LoRA, PQC
- **Infrastructure**: vLLM, Ray, Kubernetes
- **Observability**: OpenTelemetry, Prometheus
- **MLOps**: W&B, MLflow, HuggingFace Hub

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>TenSafe</strong> — Privacy-preserving ML at production scale.
</p>
