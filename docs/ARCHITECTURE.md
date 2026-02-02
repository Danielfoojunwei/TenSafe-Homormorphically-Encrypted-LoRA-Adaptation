# TenSafe Architecture

**Version**: 4.0.0
**Last Updated**: 2026-02-02

## Overview

TenSafe is a unified privacy-first ML platform that integrates core subsystems with enterprise-grade production components:

### Core Subsystems
1. **TenSafe Training API** - Privacy-preserving model fine-tuning
2. **TSSP Secure Packaging** - Cryptographically protected model distribution
3. **Platform Control Plane** - Fleet management and policy enforcement
4. **Edge Agent** - Secure deployment and attestation

### Production Integration Layer (v4.0)
5. **vLLM Backend** - High-throughput inference with HE-LoRA support
6. **Ray Train Distributed** - Scalable multi-node training with secure aggregation
7. **Observability Stack** - OpenTelemetry-native monitoring
8. **MLOps Integrations** - W&B, MLflow, HuggingFace Hub
9. **Kubernetes Deployment** - Helm charts with KEDA auto-scaling

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    TenSafe Platform v4.0                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│   ┌────────────────────────────────────────────────────────────────────────────────────────┐    │
│   │                              Client Layer (tensafe SDK)                                  │    │
│   │  ┌─────────────┐  ┌──────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │    │
│   │  │ServiceClient│──▶│ TrainingClient   │──▶│  FutureHandle  │  │  MLOps Callbacks     │  │    │
│   │  └─────────────┘  │ • forward_backward│  │  • status()    │  │  • W&B Integration   │  │    │
│   │                   │ • optim_step      │  │  • result()    │  │  • MLflow Tracking   │  │    │
│   │                   │ • sample          │  │  • cancel()    │  │  • HF Hub Push/Pull  │  │    │
│   │                   │ • save_state      │  └────────────────┘  └──────────────────────┘  │    │
│   │                   └──────────────────┘                                                  │    │
│   └────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                                         │
│                                        ▼ HTTPS/TLS 1.3                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────────────┐    │
│   │                          Server Layer (tensafe.platform)                                │    │
│   │                                                                                         │    │
│   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐ │    │
│   │  │ TenSafe API   │  │ Platform API  │  │  TSSP API     │  │   Observability Layer   │ │    │
│   │  │ /v1/training  │  │ /api/v1/      │  │ /api/tssp/    │  │   ┌─────────────────┐   │ │    │
│   │  │              │  │ attestation   │  │ upload        │  │   │ OpenTelemetry   │   │ │    │
│   │  │  ┌─────────┐ │  └───────────────┘  └───────────────┘  │   │ ├─ Traces       │   │ │    │
│   │  │  │ vLLM    │ │                                         │   │ ├─ Metrics      │   │ │    │
│   │  │  │ Backend │ │                                         │   │ └─ Logs         │   │ │    │
│   │  │  └─────────┘ │                                         │   └─────────────────┘   │ │    │
│   │  └───────────────┘                                        └─────────────────────────┘ │    │
│   │          │                  │                  │                 │                    │    │
│   │          ▼                  ▼                  ▼                 ▼                    │    │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │    │
│   │  │                           Core Services Layer                                    │  │    │
│   │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐   │  │    │
│   │  │  │ Job Queue│  │DP Engine │  │Key Mgmt  │  │Audit Log │  │ Ray Train       │   │  │    │
│   │  │  │ (Async)  │  │(RDP/PRV) │  │(KEK/DEK) │  │(Hash-    │  │ ├─ Distributed  │   │  │    │
│   │  │  │          │  │          │  │          │  │ chain)   │  │ ├─ DP-SGD       │   │  │    │
│   │  │  │          │  │          │  │          │  │          │  │ └─ Secure Agg   │   │  │    │
│   │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └─────────────────┘   │  │    │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │    │
│   │                                    │                                                   │    │
│   │                                    ▼                                                   │    │
│   │  ┌─────────────────────────────────────────────────────────────────────────────────┐  │    │
│   │  │                              Storage Layer                                       │  │    │
│   │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │  │    │
│   │  │  │Encrypted     │  │  Database    │  │ TSSP Package │  │  Artifact Registry │  │  │    │
│   │  │  │Artifact Store│  │  (SQLite/    │  │   Registry   │  │  (W&B/MLflow/HF)   │  │  │    │
│   │  │  │(AES-256-GCM) │  │   Postgres)  │  │              │  │                    │  │  │    │
│   │  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────────┘  │  │    │
│   │  └─────────────────────────────────────────────────────────────────────────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                                         │
│                                        ▼ TSSP Package                                           │
│   ┌────────────────────────────────────────────────────────────────────────────────────────┐    │
│   │                          Edge Layer (tensafe.agent)                                     │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │    │
│   │  │  Identity   │  │ Attestation │  │    TSSP     │  │   Runtime   │  │  vLLM       │  │    │
│   │  │  Manager    │  │   Verifier  │  │   Loader    │  │  (TensorRT) │  │  Inference  │  │    │
│   │  │  (mTLS)     │  │   (TPM)     │  │             │  │             │  │  Engine     │  │    │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. TenSafe Training API

The TenSafe subsystem provides privacy-first model fine-tuning with a Tinker-compatible API.

#### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TenSafe Training Flow                                  │
└──────────────────────────────────────────────────────────────────────────┘

  Client                     Server                    Storage
    │                          │                          │
    │  create_training_client  │                          │
    │─────────────────────────▶│                          │
    │                          │  Initialize DP Accountant│
    │                          │  Initialize Key Manager  │
    │  TrainingClient          │                          │
    │◀─────────────────────────│                          │
    │                          │                          │
    │  forward_backward(batch) │                          │
    │─────────────────────────▶│                          │
    │                          │  Queue Job               │
    │  FutureHandle            │                          │
    │◀─────────────────────────│                          │
    │                          │                          │
    │                          │  Worker: Compute         │
    │                          │  ├─ Forward pass         │
    │                          │  ├─ Gradient computation │
    │                          │  ├─ Gradient clipping    │
    │                          │  └─ Log to audit chain   │
    │                          │                          │
    │  future.result()         │                          │
    │─────────────────────────▶│                          │
    │  ForwardBackwardResult   │                          │
    │◀─────────────────────────│                          │
    │                          │                          │
    │  optim_step()            │                          │
    │─────────────────────────▶│                          │
    │                          │  Worker: Update          │
    │                          │  ├─ Add DP noise         │
    │                          │  ├─ Update accountant    │
    │                          │  └─ Apply gradients      │
    │                          │                          │
    │  save_state()            │                          │
    │─────────────────────────▶│                          │
    │                          │  Serialize state         │
    │                          │  Encrypt with DEK        │
    │                          │─────────────────────────▶│
    │                          │                          │  Store artifact
    │  SaveStateResult         │                          │
    │◀─────────────────────────│                          │
```

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ServiceClient` | `src/tensafe/client.py` | Main entry point, manages HTTP sessions |
| `TrainingClient` | `src/tensafe/training_client.py` | Training primitives interface |
| `FutureHandle` | `src/tensafe/futures.py` | Async operation management |
| `DPTrainer` | `src/tensafe/platform/tensafe_api/dp.py` | DP-SGD implementation |
| `RDPAccountant` | `src/tensafe/platform/tensafe_api/dp.py` | Privacy budget tracking |
| `EncryptedArtifactStore` | `src/tensafe/platform/tensafe_api/storage.py` | Per-tenant encrypted storage |
| `AuditLogger` | `src/tensafe/platform/tensafe_api/audit.py` | Hash-chained audit trail |

---

### 2. vLLM Backend Integration (v4.0)

High-throughput inference engine with HE-LoRA support via custom forward hooks.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      vLLM HE-LoRA Inference Flow                          │
└──────────────────────────────────────────────────────────────────────────┘

  Request                vLLM Engine               HE-LoRA Hooks
    │                        │                          │
    │  OpenAI-compatible    │                          │
    │  /v1/completions      │                          │
    │──────────────────────▶│                          │
    │                        │                          │
    │                        │  Token Processing        │
    │                        │  ├─ Tokenize input      │
    │                        │  └─ KV Cache lookup     │
    │                        │                          │
    │                        │  Forward Pass            │
    │                        │─────────────────────────▶│
    │                        │                          │  HE-LoRA Transform
    │                        │                          │  ├─ Encrypted A matrix
    │                        │                          │  ├─ Encrypted B matrix
    │                        │                          │  └─ HE computation
    │                        │  Modified Activations    │
    │                        │◀─────────────────────────│
    │                        │                          │
    │                        │  PagedAttention          │
    │                        │  ├─ Memory efficient    │
    │                        │  └─ Continuous batching │
    │                        │                          │
    │  Streamed tokens       │                          │
    │◀──────────────────────│                          │
```

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `HELoRAHook` | `src/tensorguard/backends/vllm/hooks.py` | Forward hooks for HE-LoRA |
| `TenSafeAsyncEngine` | `src/tensorguard/backends/vllm/engine.py` | Extended vLLM engine |
| `TenSafeVLLMConfig` | `src/tensorguard/backends/vllm/config.py` | Configuration dataclass |
| `TenSafeVLLMServer` | `src/tensorguard/backends/vllm/api.py` | OpenAI-compatible REST API |

#### Integration Points

```python
from tensorguard.backends.vllm import TenSafeAsyncEngine, TenSafeVLLMConfig

config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B",
    tensor_parallel_size=4,
    enable_he_lora=True,
    he_lora_rank=16,
)

engine = TenSafeAsyncEngine(config)
await engine.initialize()
```

---

### 3. Ray Train Distributed (v4.0)

Scalable multi-node training with differential privacy and secure gradient aggregation.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   Ray Train Distributed DP-SGD                            │
└──────────────────────────────────────────────────────────────────────────┘

    Ray Head                  Workers (N)            Secure Aggregator
       │                          │                        │
       │  ray.init(cluster)      │                        │
       │─────────────────────────▶│                        │
       │                          │                        │
       │  TenSafeTrainer.fit()   │                        │
       │─────────────────────────▶│                        │
       │                          │                        │
       │                          │  Local Batch Train     │
       │                          │  ├─ Forward pass       │
       │                          │  ├─ Per-sample clip    │
       │                          │  └─ Local gradients    │
       │                          │                        │
       │                          │  Masked Gradients      │
       │                          │───────────────────────▶│
       │                          │                        │  Pairwise masking
       │                          │                        │  g_masked = g + PRG(seed_ij)
       │                          │                        │
       │                          │  Aggregated Result     │
       │                          │◀───────────────────────│  Masks cancel:
       │                          │                        │  Σg_masked = Σg
       │                          │                        │
       │                          │  Add DP Noise          │
       │                          │  g_noisy = g + N(0,σ²) │
       │                          │                        │
       │                          │  Apply to Model        │
       │                          │  θ ← θ - η·g_noisy     │
       │                          │                        │
```

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `TenSafeTrainer` | `src/tensorguard/distributed/ray_trainer.py` | Ray Train wrapper |
| `TenSafeTrainingLoop` | `src/tensorguard/distributed/ray_trainer.py` | Per-worker training |
| `DistributedDPOptimizer` | `src/tensorguard/distributed/dp_distributed.py` | Distributed DP-SGD |
| `SecureGradientAggregator` | `src/tensorguard/distributed/dp_distributed.py` | Pairwise masking protocol |
| `DistributedRDPAccountant` | `src/tensorguard/distributed/dp_distributed.py` | Cross-worker privacy accounting |

#### Secure Aggregation Protocol

```
Worker i ←──────────────────────────────────────────────────────▶ Worker j
          │                                                        │
          │  Establish pairwise seed: seed_ij = DH(sk_i, pk_j)     │
          │                                                        │
          │  Generate masks:                                       │
          │    mask_ij = PRG(seed_ij)                              │
          │    mask_ji = -PRG(seed_ij)  (negative for j)           │
          │                                                        │
          │  Add masks to local gradient:                          │
          │    g_masked_i = g_i + Σ_j mask_ij                      │
          │                                                        │
          │  After aggregation:                                    │
          │    Σ_i g_masked_i = Σ_i g_i + Σ_i Σ_j mask_ij         │
          │                   = Σ_i g_i + 0  (masks cancel)        │
          │                   = Σ_i g_i                             │
```

---

### 4. Observability Stack (v4.0)

OpenTelemetry-native monitoring with privacy-aware tracing.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Observability Architecture                             │
└──────────────────────────────────────────────────────────────────────────┘

  TenSafe Services          OTEL Collector           Backends
        │                        │                      │
        │  Traces                │                      │
        │  ├─ Training spans    │                      │
        │  ├─ Inference spans   │                      │
        │  └─ Crypto ops        │                      │
        │───────────────────────▶│                      │
        │                        │──────────────────────▶│  Jaeger/Tempo
        │                        │                      │
        │  Metrics               │                      │
        │  ├─ DP epsilon spent  │                      │
        │  ├─ Training loss     │                      │
        │  ├─ Inference latency │                      │
        │  └─ HE operations/sec │                      │
        │───────────────────────▶│                      │
        │                        │──────────────────────▶│  Prometheus
        │                        │                      │
        │  Logs                  │                      │
        │  ├─ Structured JSON   │                      │
        │  └─ Redacted secrets  │                      │
        │───────────────────────▶│                      │
        │                        │──────────────────────▶│  Loki/ELK
```

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `setup_observability` | `src/tensorguard/observability/setup.py` | OTEL initialization |
| `TenSafeTracingMiddleware` | `src/tensorguard/observability/middleware.py` | Request tracing |
| `create_span_decorator` | `src/tensorguard/observability/middleware.py` | Function tracing |

#### TenSafe-Specific Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `tensafe_dp_epsilon_spent` | Gauge | Privacy budget consumed |
| `tensafe_training_loss` | Histogram | Training loss distribution |
| `tensafe_inference_latency_seconds` | Histogram | Inference latency (P50/P95/P99) |
| `tensafe_he_operations_total` | Counter | HE operations count |
| `tensafe_gradient_norm` | Histogram | Gradient norm after clipping |

#### Privacy-Aware Tracing

```python
# Sensitive fields automatically redacted
SENSITIVE_PATTERNS = [
    "password", "token", "api_key", "secret", "credential", "authorization"
]

# Safe to trace
span.set_attribute("http.method", "POST")
span.set_attribute("dp.epsilon_spent", 0.5)

# Automatically redacted
span.set_attribute("http.header.authorization", "[REDACTED]")
```

---

### 5. MLOps Integrations (v4.0)

Enterprise experiment tracking and model registry integration.

#### Weights & Biases Integration

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    W&B Integration Flow                                   │
└──────────────────────────────────────────────────────────────────────────┘

  TenSafe Trainer          W&B Callback              W&B Cloud
        │                       │                        │
        │  on_train_begin()    │                        │
        │─────────────────────▶│                        │
        │                       │  wandb.init()          │
        │                       │───────────────────────▶│
        │                       │                        │  Create run
        │                       │                        │
        │  on_step()           │                        │
        │─────────────────────▶│                        │
        │  {loss, grad_norm,   │  wandb.log()           │
        │   dp_epsilon}        │───────────────────────▶│
        │                       │                        │
        │  on_train_end()      │                        │
        │─────────────────────▶│                        │
        │                       │  Log privacy report   │
        │                       │  Finish run           │
        │                       │───────────────────────▶│
```

#### MLflow Integration

| Feature | Component | Description |
|---------|-----------|-------------|
| Experiment Tracking | `MLflowCallback` | Metrics, params, artifacts |
| Model Registry | `log_model()` | TSSP package registration |
| DP Certificate | `dp_certificate.json` | Privacy guarantee artifact |

#### HuggingFace Hub Integration

| Feature | Component | Description |
|---------|-----------|-------------|
| Model Push | `push_to_hub()` | TSSP-verified upload |
| Model Pull | `pull_from_hub()` | Verified download |
| Model Card | Auto-generated | Privacy-aware documentation |

---

### 6. Kubernetes Deployment (v4.0)

Production-ready Helm charts with KEDA auto-scaling.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Kubernetes Deployment Architecture                     │
└──────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │   Ingress       │
                         │   (TLS)         │
                         └────────┬────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
    │  API Pod 1    │    │  API Pod 2    │    │  API Pod N    │
    │  ┌─────────┐  │    │  ┌─────────┐  │    │  ┌─────────┐  │
    │  │tensafe  │  │    │  │tensafe  │  │    │  │tensafe  │  │
    │  │server   │  │    │  │server   │  │    │  │server   │  │
    │  │+ OTEL   │  │    │  │+ OTEL   │  │    │  │+ OTEL   │  │
    │  └─────────┘  │    │  └─────────┘  │    │  └─────────┘  │
    └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
            │                     │                    │
            └──────────┬──────────┴────────────┬───────┘
                       │                       │
                       ▼                       ▼
              ┌───────────────┐        ┌───────────────┐
              │  PostgreSQL   │        │    Redis      │
              │  (Primary +   │        │   (Cluster)   │
              │   Replicas)   │        │               │
              └───────────────┘        └───────────────┘
                       │
                       ▼
              ┌───────────────┐
              │   Vault       │
              │  (Secrets)    │
              └───────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                            KEDA Auto-scaling                              │
└──────────────────────────────────────────────────────────────────────────┘

    Prometheus ──────▶ KEDA ScaledObject ──────▶ HPA ──────▶ Deployment
        │                     │
        │  Metrics:           │  Triggers:
        │  • P95 latency      │  • P95 latency > 100ms → scale up
        │  • Queue depth      │  • Queue depth > 100 → scale up
        │  • GPU utilization  │  • GPU util < 30% → scale down
```

#### Helm Chart Structure

```
deploy/helm/tensafe/
├── Chart.yaml
├── values.yaml              # Configuration
├── templates/
│   ├── deployment.yaml      # Main deployment
│   ├── service.yaml         # ClusterIP/LoadBalancer
│   ├── ingress.yaml         # TLS termination
│   ├── configmap.yaml       # Non-secret config
│   ├── secrets.yaml         # Secret references
│   ├── keda-scaledobject.yaml  # Auto-scaling rules
│   ├── hpa.yaml             # Fallback HPA
│   └── serviceaccount.yaml  # RBAC
└── charts/                  # Dependencies
    ├── postgresql/
    ├── redis/
    └── keda/
```

#### Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replicaCount` | 2 | Initial replicas |
| `autoscaling.enabled` | true | Enable KEDA |
| `autoscaling.minReplicas` | 2 | Minimum pods |
| `autoscaling.maxReplicas` | 20 | Maximum pods |
| `resources.limits.nvidia.com/gpu` | 1 | GPU per pod |
| `vault.enabled` | true | HashiCorp Vault |

---

### 7. TSSP Secure Packaging

TSSP provides cryptographically protected model distribution with post-quantum signatures.

#### Package Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TSSP Package Lifecycle                                 │
└──────────────────────────────────────────────────────────────────────────┘

  Training                    Packaging                   Edge
    │                            │                          │
    │  TenSafe checkpoint        │                          │
    │───────────────────────────▶│                          │
    │                            │                          │
    │                            │  Create manifest         │
    │                            │  ├─ Hash all files       │
    │                            │  ├─ Add evidence.json    │
    │                            │  └─ Add dp_cert.json     │
    │                            │                          │
    │                            │  Sign manifest           │
    │                            │  ├─ Ed25519 signature    │
    │                            │  └─ Dilithium3 signature │
    │                            │                          │
    │                            │  Encrypt weights         │
    │                            │  ├─ Generate DEK         │
    │                            │  ├─ Wrap for recipients  │
    │                            │  └─ AES-256-GCM encrypt  │
    │                            │                          │
    │                            │  Package as .tssp        │
    │                            │───────────────────────────▶
    │                            │                          │
    │                            │                          │  Verify signature
    │                            │                          │  ├─ Ed25519 ✓
    │                            │                          │  └─ Dilithium3 ✓
    │                            │                          │
    │                            │                          │  Verify integrity
    │                            │                          │  └─ SHA-256 hashes ✓
    │                            │                          │
    │                            │                          │  Check policy
    │                            │                          │  └─ OPA/Rego eval ✓
    │                            │                          │
    │                            │                          │  Load
    │                            │                          │  └─ GPU memory
```

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `TSSPService` | `src/tensafe/tssp/service.py` | Package creation and verification |
| `sign_hybrid` | `src/tensafe/crypto/sig.py` | Hybrid Ed25519+Dilithium3 signatures |
| `generate_hybrid_keypair` | `src/tensafe/crypto/kem.py` | X25519+Kyber768 key generation |

---

### 8. Security Architecture

#### Encryption at Rest

```
┌─────────────────────────────────────────────────────────────────┐
│                    Encryption Architecture                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  HashiCorp      │
                    │  Vault / AWS KMS│
                    │  (KEK Storage)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │      KEK        │
                    │  (Master Key)   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ DEK-T1    │      │ DEK-T2    │      │ DEK-T3    │
    │(Tenant 1) │      │(Tenant 2) │      │(Tenant 3) │
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ Artifacts │      │ Artifacts │      │ Artifacts │
    │ (AES-GCM) │      │ (AES-GCM) │      │ (AES-GCM) │
    └───────────┘      └───────────┘      └───────────┘
```

#### Cryptographic Algorithms

| Purpose | Algorithm | Key Size | Notes |
|---------|-----------|----------|-------|
| Artifact Encryption | AES-256-GCM | 256-bit | Per-artifact nonce |
| Key Wrapping | AES-256-KWP | 256-bit | NIST SP 800-38F |
| Classical Signatures | Ed25519 | 256-bit | EdDSA |
| PQ Signatures | Dilithium3 | ~2.5KB | NIST Level 3 |
| Classical KEM | X25519 | 256-bit | ECDH |
| PQ KEM | Kyber768 | ~1KB | NIST Level 3 |
| Hashing | SHA-256 | 256-bit | Integrity |
| Password | Argon2id | 256-bit | OWASP params |

---

### 9. Differential Privacy Architecture

#### DP-SGD Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DP-SGD Training Flow                          │
└─────────────────────────────────────────────────────────────────┘

  Batch Data              DP Engine                  Privacy Accountant
      │                      │                              │
      │  input_ids          │                              │
      │  attention_mask     │                              │
      │  labels             │                              │
      │─────────────────────▶│                              │
      │                      │                              │
      │                      │  Forward pass                │
      │                      │  ├─ loss = f(x, θ)          │
      │                      │                              │
      │                      │  Backward pass               │
      │                      │  ├─ ∇θ = ∂loss/∂θ           │
      │                      │                              │
      │                      │  Per-sample gradient clipping│
      │                      │  ├─ g̃ᵢ = gᵢ / max(1, ‖gᵢ‖/C)│
      │                      │                              │
      │                      │  Aggregate + Noise           │
      │                      │  ├─ g = Σg̃ᵢ + N(0, σ²C²I)   │
      │                      │                              │
      │                      │  Track privacy spend         │
      │                      │─────────────────────────────▶│
      │                      │                              │
      │                      │                              │  Compute RDP
      │                      │                              │  ε(α) = α/(2σ²)
      │                      │                              │
      │                      │                              │  Convert to (ε,δ)-DP
      │                      │  (ε_spent, δ)                │
      │                      │◀─────────────────────────────│
      │                      │                              │
      │                      │  Update θ ← θ - η·g         │
      │                      │                              │
```

#### Privacy Accountants

| Accountant | Method | Use Case |
|------------|--------|----------|
| RDP | Rényi Differential Privacy | Default, tight composition |
| Moments | Moments accountant | Legacy compatibility |
| PRV | Privacy Random Variable | Advanced composition |
| Distributed RDP | Cross-worker composition | Ray Train distributed |

---

### 10. Training Optimizations (v4.0)

Performance optimizations for privacy-preserving training.

#### Optimization Stack

| Optimization | Component | Benefit |
|--------------|-----------|---------|
| Mixed Precision | `enable_mixed_precision()` | 2x memory reduction |
| Gradient Checkpointing | `apply_gradient_checkpointing()` | 3-4x memory reduction |
| Fused Kernels | `LigerIntegration` | 20% throughput increase |
| torch.compile | `TenSafeOptimizedTrainer` | 15% inference speedup |
| Efficient DataLoader | `create_optimized_dataloader()` | Better GPU utilization |

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `TrainingOptimizationConfig` | `src/tensorguard/optimizations/training_optimizations.py` | Optimization config |
| `TenSafeOptimizedTrainer` | `src/tensorguard/optimizations/training_optimizations.py` | Optimized training loop |
| `LigerIntegration` | `src/tensorguard/optimizations/liger_integration.py` | Fused kernel injection |

---

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Module Dependency Graph (v4.0)                │
└─────────────────────────────────────────────────────────────────┘

                    ┌───────────────┐
                    │    tensafe    │
                    │     (SDK)     │
                    └───────┬───────┘
                            │ imports
    ┌───────────────────────┼────────────────────────┐
    │                       │                        │
    ▼                       ▼                        ▼
┌─────────────┐   ┌────────────────────┐   ┌────────────────────┐
│  MLOps      │   │  tensafe.platform  │   │    Distributed     │
│ Integrations│   │  ┌──────────────┐  │   │   ┌────────────┐   │
│ ├─ W&B     │   │  │ tensafe_api  │  │   │   │ Ray Train  │   │
│ ├─ MLflow  │   │  └──────┬───────┘  │   │   │ DP Distrib │   │
│ └─ HF Hub  │   │         │          │   │   └────────────┘   │
└─────────────┘   └─────────┼──────────┘   └────────────────────┘
                            │ imports
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    crypto     │   │   identity    │   │     tssp      │
│  sig, kem,    │   │ keys, acme,   │   │ service,      │
│  pqc          │   │ scheduler     │   │ format        │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │ imports
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   backends    │   │ observability │   │ optimizations │
│   ├─ vLLM    │   │ ├─ OTEL      │   │ ├─ Liger      │
│   └─ hooks   │   │ └─ Middleware│   │ └─ Training   │
└───────────────┘   └───────────────┘   └───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     core      │
                    │ client, crypto│
                    │ adapters      │
                    └───────────────┘
```

---

## Deployment Topologies

### Single-Node Development

```
┌─────────────────────────────────────┐
│           Development Host          │
│  ┌─────────────────────────────┐   │
│  │       tensafe server        │   │
│  │  ├─ TenSafe API (:8000)     │   │
│  │  ├─ Platform API            │   │
│  │  └─ SQLite DB               │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │       tensafe client        │   │
│  │  (Python SDK)               │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Production Kubernetes (v4.0)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Production Kubernetes Deployment                    │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   Ingress       │
                    │   Controller    │
                    │   (TLS Term)    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   API Pod 1   │    │   API Pod 2   │    │   API Pod N   │
│  tensafe +    │    │  tensafe +    │    │  tensafe +    │
│  OTEL agent   │    │  OTEL agent   │    │  OTEL agent   │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┬────────────────────┐
        │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  PostgreSQL   │    │    Redis      │    │  HashiCorp    │    │    KEDA       │
│  (HA Cluster) │    │  (Cluster)    │    │    Vault      │    │  ScaledObject │
└───────────────┘    └───────────────┘    └───────────────┘    └───────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Prometheus  │    │    Jaeger     │    │   Grafana     │
│   (Metrics)   │    │   (Traces)    │    │ (Dashboards)  │
└───────────────┘    └───────────────┘    └───────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │   S3/GCS      │
                    │ Artifact Store│
                    └───────────────┘
```

### Ray Cluster (v4.0)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Ray Cluster Deployment                           │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   Ray Head      │
                    │   ├─ GCS       │
                    │   └─ Dashboard │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Ray Worker 1  │    │ Ray Worker 2  │    │ Ray Worker N  │
│ ├─ GPU x4    │    │ ├─ GPU x4    │    │ ├─ GPU x4    │
│ ├─ TenSafe   │    │ ├─ TenSafe   │    │ ├─ TenSafe   │
│ └─ DP-SGD    │    │ └─ DP-SGD    │    │ └─ DP-SGD    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │Secure Gradient│
                    │  Aggregator   │
                    │(Pairwise Mask)│
                    └───────────────┘
```

---

## Configuration Reference

### Environment Variables

| Variable | Component | Required | Default |
|----------|-----------|----------|---------|
| `TS_ENVIRONMENT` | All | Yes (prod) | `development` |
| `TS_SECRET_KEY` | Platform | Yes (prod) | - |
| `TS_KEY_MASTER` | Identity | Yes (prod) | - |
| `DATABASE_URL` | Platform | Yes (prod) | `sqlite:///./tensafe.db` |
| `TENSAFE_API_KEY` | SDK | Yes | - |
| `TENSAFE_BASE_URL` | SDK | No | `https://api.tensafe.io` |
| `TS_PQC_REQUIRED` | Crypto | No | `false` |
| `TS_DETERMINISTIC` | Training | No | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Observability | No | - |
| `WANDB_API_KEY` | W&B | No | - |
| `MLFLOW_TRACKING_URI` | MLflow | No | - |
| `HF_TOKEN` | HuggingFace | No | - |
| `RAY_ADDRESS` | Ray | No | `auto` |

---

## Related Documentation

- [TENSAFE_SPEC.md](TENSAFE_SPEC.md) - TenSafe API specification
- [TSSP_SPEC.md](TSSP_SPEC.md) - Secure packaging format
- [MATURITY.md](MATURITY.md) - Feature maturity matrix
- [SECURITY.md](../SECURITY.md) - Security policy
- [guides/vllm-integration.md](guides/vllm-integration.md) - vLLM integration guide
- [guides/ray-train.md](guides/ray-train.md) - Ray Train distributed guide
- [guides/kubernetes.md](guides/kubernetes.md) - Kubernetes deployment guide
- [guides/observability.md](guides/observability.md) - Observability setup guide
- [guides/mlops.md](guides/mlops.md) - MLOps integrations guide
