# TensorTrace

**TensorTrace** (formerly TensorGuardFlow) is a post-quantum secure MLOps platform for privacy-preserving federated learning on robotic fleets. It provides cryptographic protection for model weights, secure packaging, and fleet-wide governance.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-186%20passed-green.svg)]()

---

## Overview

TensorTrace provides:

- **TGSP (TensorGuard Secure Package)**: Cryptographically protected model packaging with hybrid PQC signatures
- **N2HE Encryption**: LWE-based homomorphic encryption for gradient aggregation
- **Platform API**: FastAPI control plane for fleet management and telemetry
- **Agent Daemon**: Unified edge agent for identity, attestation, and ML operations
- **PEFT Integration**: Parameter-efficient fine-tuning workflows for VLA models
- **Benchmarking Harness**: Performance measurement for API latency, throughput, and resource consumption

---

## Architecture

```
tensortrace/
├── src/tensorguard/
│   ├── agent/              # Edge daemon (identity, network, ML)
│   ├── bench/              # Benchmark suites (micro, privacy, compliance)
│   ├── core/               # Core SDK (client, adapters, crypto, pipeline)
│   ├── crypto/             # Cryptographic primitives (PQC, signatures)
│   ├── identity/           # Identity management (keys, certificates)
│   ├── integrations/       # External integrations (PEFT, VDA5050, RMF)
│   ├── moai/               # MOAI FHE optimization
│   ├── platform/           # Control plane (FastAPI, database, API)
│   ├── privacy/            # Differential privacy components
│   ├── schemas/            # Pydantic schemas
│   ├── server/             # Aggregation server
│   ├── serving/            # Model serving backends
│   ├── tgsp/               # Secure package format
│   └── utils/              # Utilities (config, production gates)
├── benchmarks/             # Performance benchmarking harness
├── tests/                  # Test suite (unit, integration, security, e2e)
└── docs/                   # Documentation and research summaries
```

---

## Installation

### Requirements

- Python 3.9+
- SQLite (bundled) or PostgreSQL
- Optional: liboqs for post-quantum cryptography

### Basic Installation

```bash
git clone https://github.com/your-org/tensortrace.git
cd tensortrace
pip install -e .
```

### With Optional Dependencies

```bash
# Include benchmarking tools
pip install -e ".[bench]"

# Include federated learning (Flower + TenSEAL)
pip install -e ".[fl]"

# Include post-quantum cryptography (requires liboqs native library)
pip install -e ".[pqc]"

# Install all optional dependencies
pip install -e ".[all]"
```

---

## Quick Start

### Start the Platform Server

```bash
tensorguard server --host 0.0.0.0 --port 8000
```

Access the API documentation at `http://localhost:8000/docs`.

### Start the Edge Agent

```bash
tensorguard agent
```

### Create a Secure Package (TGSP)

```bash
tensorguard pkg create \
  --out model.tgsp \
  --producer-signing-key keys/producer.priv \
  --payload "adapter:weights:adapter.bin" \
  --policy policy.yaml \
  --recipient "user1:keys/user1.pub"
```

### Verify and Decrypt

```bash
# Verify package integrity
tensorguard pkg verify --in model.tgsp

# Decrypt for authorized recipient
tensorguard pkg decrypt \
  --in model.tgsp \
  --recipient-id user1 \
  --recipient-private-key keys/user1.priv \
  --outdir ./extracted
```

### Run PEFT Workflow

```bash
tensorguard peft run config.json
```

### Run Benchmarks

```bash
tensorguard bench micro    # Crypto micro-benchmarks
tensorguard bench privacy  # Privacy mechanism benchmarks
```

---

## Core Components

### 1. TGSP (TensorGuard Secure Package)

TGSP is a binary format for securely packaging ML model artifacts with:

- **Hybrid Signatures**: Ed25519 + Dilithium3 (post-quantum)
- **Hybrid Encryption**: X25519 + Kyber768 key encapsulation
- **HPKE v0.3**: Modern hybrid public key encryption
- **Policy Enforcement**: Embedded policy manifests

```python
from tensorguard.tgsp.service import TGSPService
from tensorguard.crypto.sig import generate_hybrid_sig_keypair
from tensorguard.crypto.kem import generate_hybrid_keypair

# Generate keys
sig_pub, sig_priv = generate_hybrid_sig_keypair()
kem_pub, kem_priv = generate_hybrid_keypair()

# Create secure package
service = TGSPService()
package = service.create(
    payloads=[("adapter", "weights", "model.bin")],
    signing_key=sig_priv,
    recipients=[("fleet1", kem_pub)],
    policy=policy_data
)
```

### 2. N2HE Encryption

N2HE (Near-Native Homomorphic Encryption) provides LWE-based encryption for gradient aggregation:

```python
from tensorguard.core.crypto import N2HEEncryptor, N2HEContext

# Initialize context
ctx = N2HEContext(security_level=128)
ctx.generate_keys()

# Encrypt gradients
encryptor = N2HEEncryptor(security_level=128)
ciphertext = encryptor.encrypt(gradient_bytes)

# Homomorphic addition (server-side)
summed = encryptor.add(ciphertext1, ciphertext2)
```

> **Note**: N2HE is a research prototype. See `src/tensorguard/core/crypto.py` for implementation details and security considerations.

### 3. Platform API

FastAPI-based control plane with:

- **Fleet Management**: Device registration, attestation
- **Telemetry Ingestion**: OTLP-compatible metrics/traces
- **Policy Engine**: Configurable policy packs
- **TGSP Registry**: Package upload/download
- **Enablement Jobs**: Fleet-wide rollout orchestration

Key endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/attestation/verify` | Verify device attestation |
| `POST /api/v1/tgsp/key-release` | Conditional key release |
| `GET /api/v1/enablement/stats` | Fleet enablement statistics |
| `POST /api/community/tgsp/upload` | Upload TGSP package |
| `POST /api/v1/telemetry/ingest` | Ingest telemetry data |

### 4. Edge Agent

Unified daemon providing:

- **Identity Management**: Certificate lifecycle, CSR generation
- **Machine Identity Guard (MIG)**: Automated certificate renewal
- **Network Defense**: Rate limiting, connection management
- **ML Pipeline**: Local training coordination

```python
from tensorguard.core.client import create_client
from tensorguard.core.adapters import MoEAdapter

# Create client with MoE adapter
client = create_client(
    security_level=128,
    cid="robot_alpha",
    key_path="keys/fleet.key"
)
client.set_adapter(MoEAdapter())

# Process training round
for demo in demonstrations:
    client.add_demonstration(demo)

update_package = client.process_round()
```

### 5. MoE Adapters

Mixture-of-Experts routing for VLA models:

```python
from tensorguard.core.adapters import MoEAdapter

adapter = MoEAdapter()

# Get expert weights for instruction
weights = adapter.get_expert_gate_weights("Pick up the blue block")
# Returns: {'visual_primary': 0.42, 'language_semantic': 0.25, ...}
```

Expert categories:
- `visual_primary`: Object recognition, shapes
- `visual_aux`: Colors, textures
- `language_semantic`: Instruction parsing
- `manipulation_grasp`: Grasping actions
- `manipulation_place`: Placement actions
- `navigation_spatial`: Spatial reasoning
- `fastening_screwing`: Assembly tasks

---

## Benchmarking

### Benchmark Harness

The benchmarking harness measures empirical performance:

```bash
# Full benchmark suite
make bench-full

# Quick benchmark (5 iterations)
make bench-quick

# Generate report
make bench-report

# Run regression tests
make bench-regression
```

### Measured Metrics

| Category | Metrics |
|----------|---------|
| **HTTP API** | Latency (p50, p95, p99), throughput (req/s), error rate |
| **Telemetry** | Ingestion rate (events/s), batch processing time |
| **Resources** | CPU utilization, memory usage, disk I/O |
| **Crypto** | Encryption/decryption latency, key generation time |

### Academic Benchmark Comparison

Based on published research benchmarks:

| Benchmark | Reference | Measured |
|-----------|-----------|----------|
| PlantD (Telemetry) | 10K-100K events/s | Target: 50K events/s |
| OpenTelemetry Collector | 25K spans/s | Target: 20K spans/s |
| SProBench API Latency | p99 < 100ms | Target: p99 < 50ms |

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TG_ENVIRONMENT` | `development` or `production` | `development` |
| `TG_SECRET_KEY` | JWT signing key (required in production) | - |
| `DATABASE_URL` | Database connection string | `sqlite:///./tensorguard.db` |
| `TG_SIMULATION` | Enable simulation mode | `false` |
| `TG_ALLOW_TPM_SIMULATOR` | Allow TPM simulator in production | `false` |

### Production Gates

TensorTrace enforces security gates in production:

- TPM simulator is blocked (use real TPM or set `TG_ALLOW_TPM_SIMULATOR=true`)
- Secret key is required (`TG_SECRET_KEY` must be set)
- Insecure key providers are blocked (use HSM or Vault)

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Security tests
pytest tests/security/ -v

# Integration tests
pytest tests/integration/ -v

# Skip tests requiring optional dependencies
pytest tests/ -v -m "not requires_liboqs"
```

### Test Coverage

```bash
pytest tests/ --cov=tensorguard --cov-report=html
```

Current test status: **186 passed, 17 skipped** (optional dependencies)

---

## Security Considerations

### Cryptographic Components

| Component | Algorithm | Security Level |
|-----------|-----------|----------------|
| Symmetric Encryption | AES-256-GCM | 256-bit |
| Key Encapsulation | X25519 + Kyber768 | 128-bit PQ |
| Digital Signatures | Ed25519 + Dilithium3 | 128-bit PQ |
| Password Hashing | Argon2id | OWASP recommended |
| N2HE Encryption | LWE (n=1024, q=2^32) | 128-bit (research) |

### Production Hardening

1. **Key Management**: Use HSM or cloud KMS (AWS KMS, Azure Key Vault, GCP Cloud KMS)
2. **TLS Everywhere**: All API endpoints require HTTPS in production
3. **Rate Limiting**: Built-in rate limiting on all endpoints
4. **Audit Logging**: Comprehensive audit trail for security events
5. **Secret Rotation**: Automated key rotation support

### Research vs. Production

Some components are research prototypes:

- **N2HE**: Experimental homomorphic encryption (not audited for production)
- **Privacy Budgeting**: Differential privacy accountant (verify parameters)

---

## CLI Reference

```
tensorguard --help

Commands:
  agent    Start the Unified Edge Daemon
  server   Start the Control Plane Server
  pkg      TGSP Package Management (create, verify, decrypt)
  bench    Run TensorGuard Benchmarks
  peft     PEFT Studio & Orchestration
  ingest   Ingest rosbag2 or MCAP files
```

---

## Dependencies

### Core (Required)

- FastAPI, Uvicorn (API server)
- SQLModel, Pydantic (data modeling)
- Cryptography (crypto primitives)
- NumPy (numerical operations)

### Optional

- **bench**: scipy, scikit-learn, xgboost, psutil
- **fl**: flwr (Flower), tenseal (FHE)
- **pqc**: liboqs-python (post-quantum crypto)
- **acme**: josepy (certificate management)

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

## References

### Research Papers

1. Zhang, L. et al. (2025). *MOAI: Module-Optimising Architecture for Non-Interactive Secure Transformer Inference.* IACR ePrint.
2. Lam, K.Y. et al. (2024). *Efficient FHE-based Privacy-Enhanced Neural Network for Trustworthy AI-as-a-Service.* IEEE TDSC.
3. Kim, M. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv:2406.09246.

### Specifications

- [HPKE RFC 9180](https://www.rfc-editor.org/rfc/rfc9180.html)
- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [OpenTelemetry Protocol (OTLP)](https://opentelemetry.io/docs/specs/otlp/)

---

**Version**: 2.3.0
**Maintainer**: TensorGuard Team
