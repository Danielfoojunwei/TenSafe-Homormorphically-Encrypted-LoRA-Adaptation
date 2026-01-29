# TensorGuardFlow

**TensorGuardFlow** is a unified, privacy-first ML platform for secure model training, packaging, and deployment. It combines post-quantum cryptographic protection, differential privacy training, and fleet-wide governance into one cohesive system.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-278%20passed-green.svg)]()

---

## Overview

TensorGuardFlow provides an end-to-end privacy-preserving ML pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TensorGuardFlow Platform                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  TG-Tinker   │───▶│    TGSP      │───▶│   Platform   │───▶│   Edge     │ │
│  │  Training    │    │   Packaging  │    │   Control    │    │   Agent    │ │
│  │  API         │    │              │    │   Plane      │    │            │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│        │                   │                   │                    │        │
│        ▼                   ▼                   ▼                    ▼        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Unified Security Layer                                ││
│  │  • AES-256-GCM encryption at rest   • Hash-chained audit logging        ││
│  │  • Post-quantum signatures          • Per-tenant key isolation          ││
│  │  • Differential privacy (DP-SGD)    • mTLS identity management          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Capabilities

| Component | Description |
|-----------|-------------|
| **TG-Tinker Training API** | Privacy-first Python SDK for fine-tuning LLMs with async execution, DP-SGD, and encrypted checkpoints |
| **TGSP Secure Packaging** | Cryptographically protected model packaging with hybrid PQC signatures (Ed25519 + Dilithium3) |
| **Platform Control Plane** | FastAPI server for fleet management, telemetry, policy enforcement, and artifact registry |
| **Edge Agent** | Unified daemon for identity management, attestation, and secure model deployment |
| **Privacy Engine** | Differential privacy accounting (RDP, Moments, PRV) with per-batch gradient clipping |

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/tensortrace.git
cd tensortrace
pip install -e ".[all]"
```

### Start the Platform Server

```bash
tensorguard server --host 0.0.0.0 --port 8000
```

### Train a Model with Privacy Guarantees (TG-Tinker)

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig, DPConfig

# Initialize the service
service = ServiceClient(
    base_url="http://localhost:8000",
    api_key="tg-key-xxxx"
)

# Configure privacy-first training
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    dp_config=DPConfig(
        enabled=True,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=8.0
    )
)

# Create training client
tc = service.create_training_client(config)

# Training loop with async execution
for batch in dataloader:
    fb_future = tc.forward_backward(batch)
    opt_future = tc.optim_step()

    result = fb_future.result()
    opt_result = opt_future.result()

    print(f"Loss: {result.loss}, Epsilon: {opt_result.dp_metrics.total_epsilon}")

# Save encrypted checkpoint
checkpoint = tc.save_state()
print(f"Artifact ID: {checkpoint.artifact_id}")
```

### Package and Deploy

```bash
# Create secure package from trained model
tensorguard pkg create \
  --out model.tgsp \
  --producer-signing-key keys/producer.priv \
  --payload "adapter:weights:adapter.bin" \
  --policy policy.yaml \
  --recipient "fleet1:keys/fleet1.pub"

# Verify and deploy
tensorguard pkg verify --in model.tgsp
```

---

## Architecture

```
tensortrace/
├── src/
│   ├── tg_tinker/                     # Training SDK (Client-side)
│   │   ├── client.py                  # ServiceClient - main entry point
│   │   ├── training_client.py         # TrainingClient - training primitives
│   │   ├── futures.py                 # FutureHandle - async operations
│   │   ├── schemas.py                 # Pydantic models
│   │   └── exceptions.py              # Error handling
│   │
│   └── tensorguard/                   # Platform (Server-side)
│       ├── platform/
│       │   ├── tg_tinker_api/         # Training API server
│       │   │   ├── routes.py          # FastAPI endpoints
│       │   │   ├── storage.py         # Encrypted artifact storage
│       │   │   ├── audit.py           # Hash-chained audit logging
│       │   │   └── dp.py              # Differential privacy
│       │   └── api/                   # Platform APIs
│       │
│       ├── tgsp/                      # Secure packaging
│       ├── crypto/                    # Cryptographic primitives
│       │   ├── sig.py                 # Hybrid signatures (Ed25519 + Dilithium3)
│       │   └── kem.py                 # Key encapsulation (X25519 + Kyber768)
│       │
│       ├── identity/                  # Identity management
│       │   ├── keys/                  # Key providers (File, HSM, KMS)
│       │   └── acme/                  # Certificate automation
│       │
│       ├── privacy/                   # Privacy components
│       ├── agent/                     # Edge daemon
│       └── core/                      # Core SDK
│
├── tests/                             # Test suite
├── docs/                              # Documentation
└── examples/                          # Example scripts
```

---

## TG-Tinker Training API

TG-Tinker provides a privacy-first alternative to Thinking Machines' Tinker API with:

### Training Primitives

| Primitive | Description | Returns | Async |
|-----------|-------------|---------|-------|
| `forward_backward(batch)` | Compute forward pass and gradients | `FutureHandle` | Yes |
| `optim_step()` | Apply optimizer update with optional DP noise | `FutureHandle` | Yes |
| `sample(prompt, **kwargs)` | Generate samples from current model | `SampleResult` | No |
| `save_state()` | Save encrypted checkpoint | `SaveStateResult` | No |
| `load_state(artifact_id)` | Load checkpoint from artifact | `LoadStateResult` | No |

### Privacy Features

- **Encryption at Rest**: All artifacts encrypted with AES-256-GCM using per-tenant keys
- **Hash-Chained Audit**: Tamper-evident logging of all operations
- **Differential Privacy**: Built-in DP-SGD with RDP/Moments/PRV accountants
- **Batch Privacy**: Training batches held in memory only, hashes logged

### CLI Commands

```bash
# Create training client
tg-tinker create-client --model meta-llama/Llama-3-8B --dp --lora-rank 16

# Check status
tg-tinker status --client-id tc-xxx

# View audit logs
tg-tinker audit-logs --client-id tc-xxx
```

See [docs/PRIVACY_TINKER_SPEC.md](docs/PRIVACY_TINKER_SPEC.md) for full specification.

---

## TGSP Secure Packaging

TGSP (TensorGuard Secure Package) provides cryptographically protected model packaging:

### Features

- **Hybrid Signatures**: Ed25519 + Dilithium3 (post-quantum resistant)
- **Hybrid Encryption**: X25519 + Kyber768 key encapsulation
- **HPKE v0.3**: Modern hybrid public key encryption
- **Policy Enforcement**: Embedded OPA/Rego policy manifests
- **Evidence Chain**: Training telemetry and privacy budget records

### Package Structure

```
model.tgsp
├── manifest.json          # Integrity hashes
├── manifest.sig           # Detached hybrid signature
├── policy.rego            # OPA gatekeeper policy
├── weights.enc            # Encrypted model weights
├── optimization.json      # Hardware acceleration metadata
└── evidence.json          # Training conditions audit
```

### Usage

```python
from tensorguard.tgsp.service import TGSPService
from tensorguard.crypto.sig import generate_hybrid_sig_keypair

# Generate keys
sig_pub, sig_priv = generate_hybrid_sig_keypair()

# Create secure package
service = TGSPService()
package = service.create(
    payloads=[("adapter", "weights", "model.bin")],
    signing_key=sig_priv,
    recipients=[("fleet1", kem_pub)],
    policy=policy_data
)
```

See [docs/TGSP_SPEC.md](docs/TGSP_SPEC.md) for full specification.

---

## Security Model

### Cryptographic Primitives

| Component | Algorithm | Security Level |
|-----------|-----------|----------------|
| Symmetric Encryption | AES-256-GCM | 256-bit |
| Key Encapsulation | X25519 + Kyber768 | 128-bit post-quantum |
| Digital Signatures | Ed25519 + Dilithium3 | 128-bit post-quantum |
| Password Hashing | Argon2id | OWASP recommended |
| Gradient Encryption | LWE-based N2HE | 128-bit (research) |

### Key Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │           Key Hierarchy             │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
              ┌─────▼─────┐                    ┌──────▼──────┐
              │    KEK    │                    │   Signing   │
              │  (Vault)  │                    │     Key     │
              └─────┬─────┘                    └─────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
  ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
  │  DEK-T1   │ │DEK-T2 │ │  DEK-Tn   │
  │ (Tenant1) │ │       │ │           │
  └───────────┘ └───────┘ └───────────┘
```

### Production Requirements

| Variable | Purpose | Required |
|----------|---------|----------|
| `TG_ENVIRONMENT` | Environment mode | Yes (`production`) |
| `TG_SECRET_KEY` | JWT signing key | Yes |
| `TG_KEY_MASTER` | Key encryption master key | Yes |
| `DATABASE_URL` | Database connection | Yes |
| `TG_PQC_REQUIRED` | Enforce PQC signatures | Recommended |

See [SECURITY.md](SECURITY.md) and [HARDENING_REPORT.md](HARDENING_REPORT.md) for details.

---

## Platform API

### Training API Endpoints (TG-Tinker)

| Endpoint | Description |
|----------|-------------|
| `POST /v1/training_clients` | Create training client |
| `POST /v1/training_clients/{id}/forward_backward` | Queue forward-backward pass |
| `POST /v1/training_clients/{id}/optim_step` | Queue optimizer step |
| `POST /v1/training_clients/{id}/sample` | Generate samples |
| `POST /v1/training_clients/{id}/save_state` | Save encrypted checkpoint |
| `GET /v1/futures/{id}` | Get async operation status |
| `GET /v1/audit_logs` | Retrieve audit logs |

### Platform API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/attestation/verify` | Verify device attestation |
| `POST /api/v1/tgsp/key-release` | Conditional key release |
| `GET /api/v1/enablement/stats` | Fleet enablement statistics |
| `POST /api/community/tgsp/upload` | Upload TGSP package |
| `POST /api/v1/telemetry/ingest` | Ingest telemetry data |

---

## Integrations

### Robotics Middleware

- **ROS 2**: Native `rclpy` node for camera/joint state subscription
- **VDA5050**: AMR fleet management protocol support
- **Formant.io**: Cloud observability connector

### Simulation

- **NVIDIA Isaac Lab**: Sim-to-real connector with domain randomization

### Deployment Targets

- **NVIDIA Jetson**: TensorRT optimization, TrustZone TEE support
- **Kubernetes**: Helm charts with HPA scaling

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for details.

---

## CLI Reference

```
tensorguard --help

Commands:
  server   Start the Control Plane Server
  agent    Start the Unified Edge Daemon
  pkg      TGSP Package Management (create, verify, decrypt)
  bench    Run TensorGuard Benchmarks
  peft     PEFT Studio & Orchestration
  tinker   TG-Tinker Training Commands
  ingest   Ingest rosbag2 or MCAP files

tg-tinker --help

Commands:
  create-client    Create a new training client
  status           Check training client status
  forward-backward Submit forward-backward operation
  optim-step       Submit optimizer step
  save-state       Save training state
  audit-logs       View audit logs
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run by category
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/security/ -v                # Security tests
pytest -m tg_tinker -v                   # TG-Tinker tests only

# Test coverage
pytest tests/ --cov=tensorguard --cov=tg_tinker --cov-report=html
```

Current status: **278 tests passed**

---

## Documentation

| Document | Description |
|----------|-------------|
| [PRIVACY_TINKER_SPEC.md](docs/PRIVACY_TINKER_SPEC.md) | TG-Tinker API specification |
| [TGSP_SPEC.md](docs/TGSP_SPEC.md) | Secure packaging format |
| [SECURITY.md](SECURITY.md) | Security policy and guidelines |
| [HARDENING_REPORT.md](HARDENING_REPORT.md) | Production hardening report |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Deployment guide |
| [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) | Integration ecosystem |

---

## Installation Options

```bash
# Basic installation
pip install -e .

# With TG-Tinker dependencies (recommended)
pip install -e ".[tinker]"

# With benchmarking tools
pip install -e ".[bench]"

# With federated learning
pip install -e ".[fl]"

# With post-quantum cryptography (requires liboqs)
pip install -e ".[pqc]"

# All optional dependencies
pip install -e ".[all]"
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Run linting (`ruff check src/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **DTC @ NTU** (Digital Trust Centre, Nanyang Technological University) - N2HE research
- **HintSight Technology** - Homomorphic encryption optimization
- **Flower Labs** - Federated learning framework

---

**Version**: 3.0.0
**Maintainer**: TensorGuard Team
