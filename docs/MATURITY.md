# TenSafe Feature Maturity Matrix

**Version**: 4.0.0
**Date**: 2026-02-02

## Maturity Levels

| Level | Description | Production Use |
|-------|-------------|----------------|
| **Stable** | Battle-tested, full test coverage, documented | Yes |
| **Beta** | Feature complete, needs more testing | With caution |
| **Alpha** | Working but API may change | No |
| **Experimental** | Proof of concept only | Never |
| **Toy** | Testing/simulation only, insecure | Never |

## Feature Matrix

### Core SDK (`tg_tinker`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| ServiceClient | **Stable** | HTTP client with retries, auth |
| TrainingClient | **Stable** | Async training operations |
| FutureHandle | **Stable** | Async result polling |
| LoRA Config | **Stable** | Configuration validated |
| DP Config | **Stable** | Validated against Opacus |

### Platform Server (`tensorguard.platform`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| FastAPI Routes | **Stable** | Full OpenAPI spec |
| Health Endpoints | **Stable** | K8s ready |
| Security Headers | **Stable** | OWASP compliant |
| SQLite Backend | **Beta** | Dev only, not for production |
| PostgreSQL Backend | **Beta** | Production ready with HA |
| Authentication | **Alpha** | API key only, no OAuth/OIDC |

### Cryptography (`tensorguard.crypto`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| AES-256-GCM Encryption | **Stable** | Uses cryptography library |
| ChaCha20-Poly1305 | **Stable** | Uses cryptography library |
| Ed25519 Signatures | **Stable** | Uses cryptography library |
| X25519 ECDH | **Stable** | Uses cryptography library |
| ML-KEM-768 (Kyber) | **Beta** | Requires liboqs |
| ML-DSA-65 (Dilithium) | **Beta** | Requires liboqs |
| Hybrid PQC Signatures | **Beta** | Classical + PQC |

### TGSP Packaging (`tensorguard.tgsp`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| Package Creation | **Stable** | Manifest + payload |
| Package Verification | **Stable** | Signature verification |
| Multi-recipient | **Beta** | HPKE-based |
| Streaming Decrypt | **Beta** | Large file support |

### Homomorphic Encryption (`tensorguard.n2he`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| ToyN2HEScheme | **Toy** | NO SECURITY, testing only |
| NativeN2HEScheme | **Alpha** | Requires C++ library |
| Key Management | **Beta** | HEKeyManager |
| Ciphertext Serialization | **Stable** | Binary, JSON, Base64 |
| Encrypted LoRA Runtime | **Alpha** | Toy mode for testing |
| Private Inference | **Experimental** | Proof of concept |
| MOAI Rotation Optimization | **Beta** | 25x speedup |

### Differential Privacy

| Feature | Maturity | Notes |
|---------|----------|-------|
| RDP Accountant | **Stable** | Opacus-compatible |
| Gradient Clipping | **Stable** | Per-sample bounds |
| Noise Injection | **Stable** | Calibrated Gaussian |
| Budget Tracking | **Stable** | (ε, δ) conversion |
| Distributed DP-SGD | **Beta** | Ray Train integration |

### Compliance & Telemetry

| Feature | Maturity | Notes |
|---------|----------|-------|
| Hash-Chain Audit | **Stable** | Tamper-evident |
| Compliance Events | **Beta** | ISO/SOC mapping |
| PII Scanning | **Alpha** | Regex-based |
| Evidence Reports | **Beta** | JSON + Markdown |

---

## v4.0 Integration Features

### vLLM Backend (`tensorguard.backends.vllm`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| TenSafeAsyncEngine | **Beta** | vLLM 0.4.0+ compatible |
| HE-LoRA Hooks | **Alpha** | Custom forward hooks |
| PagedAttention Support | **Stable** | Native vLLM feature |
| Tensor Parallelism | **Beta** | Multi-GPU inference |
| OpenAI-Compatible API | **Stable** | /v1/completions, /v1/chat |
| Streaming Responses | **Stable** | SSE protocol |
| Speculative Decoding | **Experimental** | Draft model support |

### Ray Train Distributed (`tensorguard.distributed`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| TenSafeTrainer | **Beta** | Ray Train 2.9+ |
| Distributed DP-SGD | **Beta** | Cross-worker privacy |
| Secure Gradient Aggregation | **Alpha** | Pairwise masking protocol |
| DistributedRDPAccountant | **Beta** | Coordinated budget tracking |
| Checkpoint Callback | **Stable** | TSSP-compatible saves |
| Fault Tolerance | **Beta** | Worker recovery |
| Multi-node Training | **Beta** | Tested up to 8 nodes |

### Observability (`tensorguard.observability`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| OpenTelemetry Setup | **Stable** | OTEL SDK 1.20+ |
| Tracing Middleware | **Stable** | FastAPI/Starlette |
| Privacy Metrics | **Beta** | DP epsilon, gradients |
| Sensitive Data Redaction | **Stable** | Auto-redact secrets |
| Prometheus Exporter | **Stable** | Native format |
| Jaeger Integration | **Stable** | Trace export |
| Span Decorators | **Stable** | Function tracing |

### MLOps Integrations (`tensorguard.integrations`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| **Weights & Biases** | | |
| TenSafeWandbCallback | **Stable** | W&B SDK compatible |
| Privacy Metrics Logging | **Stable** | ε, δ tracking |
| Gradient Histograms | **Beta** | Distribution tracking |
| Model Artifact Logging | **Beta** | TSSP packages |
| | | |
| **MLflow** | | |
| TenSafeMLflowCallback | **Stable** | MLflow 2.0+ |
| Experiment Tracking | **Stable** | Metrics, params |
| Model Registry | **Beta** | TSSP integration |
| DP Certificate Logging | **Stable** | Artifact storage |
| | | |
| **HuggingFace Hub** | | |
| TenSafeHFHubIntegration | **Beta** | HF Hub API |
| TSSP Push/Pull | **Beta** | Verified uploads |
| Auto Model Card | **Beta** | Privacy-aware docs |
| Private Repos | **Stable** | Token auth |

### Kubernetes Deployment (`deploy/`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| Helm Chart | **Beta** | Helm 3.0+ |
| KEDA Auto-scaling | **Beta** | SLI-based triggers |
| PostgreSQL Subchart | **Stable** | Bitnami chart |
| Redis Subchart | **Stable** | Bitnami chart |
| Vault Integration | **Alpha** | Secret injection |
| GPU Scheduling | **Beta** | NVIDIA device plugin |
| Ingress TLS | **Stable** | cert-manager ready |
| Network Policies | **Alpha** | Zero-trust networking |

### Training Optimizations (`tensorguard.optimizations`)

| Feature | Maturity | Notes |
|---------|----------|-------|
| Mixed Precision (BF16) | **Stable** | PyTorch native |
| Mixed Precision (FP16) | **Stable** | GradScaler support |
| Gradient Checkpointing | **Stable** | Memory optimization |
| Liger Kernel Integration | **Beta** | Fused operations |
| torch.compile | **Alpha** | PyTorch 2.0+ |
| Efficient DataLoader | **Stable** | Pin memory, prefetch |
| Channels Last Format | **Beta** | Memory layout opt |

---

## Important Warnings

### ToyN2HEScheme

**WARNING**: The `ToyN2HEScheme` provides NO cryptographic security. It is a pure-Python simulation for:
- Testing API contracts
- Benchmarking performance characteristics
- Development without native library

To use toy mode, you MUST set:
```bash
export TENSAFE_TOY_HE=1
```

Production deployments MUST use `NativeN2HEScheme` with the N2HE C++ library.

### SQLite Backend

The default SQLite database is NOT suitable for production:
- No connection pooling
- Single-threaded writes
- No replication

Set `DATABASE_URL` to PostgreSQL for production.

### Post-Quantum Cryptography

PQC features (ML-KEM, ML-DSA) require `liboqs` native library:
```bash
pip install liboqs-python
```

Without liboqs, PQC operations will fail with clear errors.

### vLLM Backend

The vLLM backend requires:
```bash
pip install vllm>=0.4.0
```

HE-LoRA hooks are in **Alpha** and may impact inference performance. Benchmark thoroughly before production use.

### Ray Train Distributed

Secure gradient aggregation is in **Alpha**. For production distributed training with strong privacy guarantees, consider:
- Using a trusted aggregator
- Implementing full MPC protocol
- Adding verification steps

---

## Package Names

| README Name | Actual Package | Description |
|-------------|----------------|-------------|
| `tensafe` | `tg_tinker` | Python SDK |
| `tensorguard` | `tensorguard` | Server + security layer |

The README uses "tensafe" as a product name, but the installable packages are:
- `tg_tinker`: Client SDK
- `tensorguard`: Server components

```python
# Correct imports
from tg_tinker import ServiceClient, TrainingConfig
from tensorguard.n2he import ToyN2HEScheme
from tensorguard.crypto import sign_ed25519
from tensorguard.backends.vllm import TenSafeAsyncEngine
from tensorguard.distributed import TenSafeTrainer
from tensorguard.integrations import TenSafeWandbCallback
```

---

## Testing Requirements

| Test Suite | Requirements | Command |
|------------|--------------|---------|
| Unit Tests | Python 3.9+ | `pytest tests/unit` |
| Integration | FastAPI, httpx | `pytest tests/integration` |
| N2HE Tests | TENSAFE_TOY_HE=1 | `pytest tests/n2he` |
| Security Tests | cryptography | `pytest tests/security` |
| E2E Tests | All dependencies | `pytest tests/e2e` |
| vLLM Tests | vllm, torch | `pytest tests/integration/test_vllm*.py` |
| Ray Tests | ray[train] | `pytest tests/integration/test_ray*.py` |
| OTEL Tests | opentelemetry | `pytest tests/integration/test_observability*.py` |

---

## Upgrade Path

### From 3.x to 4.x

1. **New dependencies**: Install vLLM, Ray, OpenTelemetry as needed
2. **Configuration changes**: New environment variables for integrations
3. **API additions**: New modules under `tensorguard.backends`, `tensorguard.distributed`, etc.
4. **Backward compatible**: All 3.x code continues to work

```bash
# Upgrade command
pip install tensafe[all]>=4.0.0

# Or selective features
pip install tensafe[vllm]>=4.0.0
pip install tensafe[ray]>=4.0.0
pip install tensafe[observability]>=4.0.0
pip install tensafe[mlops]>=4.0.0
```

### From 2.x to 3.x

1. **Package rename**: `tensafe` → `tg_tinker`
2. **N2HE gating**: Set `TENSAFE_TOY_HE=1` for toy mode
3. **Error codes**: All errors now have `TG_*` codes
4. **Logging**: Structured JSON in production

---

## Future Roadmap

| Feature | Target Version | Status |
|---------|---------------|--------|
| Native N2HE Integration | 4.1 | In progress |
| OAuth/OIDC Authentication | 4.2 | Planned |
| Full MPC Gradient Aggregation | 4.2 | Planned |
| HSM Key Storage | 5.0 | Planned |
| TensorRT-LLM Backend | 4.1 | Planned |
| Speculative Decoding GA | 4.1 | Planned |
| Multi-cloud Deployment | 4.2 | Planned |
