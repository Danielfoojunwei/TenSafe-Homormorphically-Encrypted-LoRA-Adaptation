# TenSafe Module Map

**Updated:** 2026-01-30

This document describes the actual module structure of the TenSafe codebase.

## Package Structure

```
src/
├── tensorguard/                    # Core TenSafe library
│   ├── __init__.py                 # Package exports
│   ├── crypto/                     # Cryptographic primitives
│   │   ├── __init__.py
│   │   ├── kem.py                  # Key Encapsulation Mechanism (AES-256)
│   │   ├── sig.py                  # Digital signatures (Ed25519)
│   │   ├── payload.py              # Encrypted payload handling
│   │   └── pqc/                    # Post-quantum cryptography
│   │       ├── __init__.py
│   │       ├── agility.py          # Crypto agility abstraction
│   │       ├── kyber.py            # ML-KEM-768 (Kyber) KEM
│   │       └── dilithium.py        # ML-DSA (Dilithium) signatures
│   ├── edge/                       # Edge deployment
│   │   ├── __init__.py
│   │   └── tgsp_client.py          # TGSP client for edge devices
│   ├── evidence/                   # Audit and compliance
│   │   ├── __init__.py
│   │   ├── canonical.py            # Canonical serialization
│   │   └── store.py                # Evidence store
│   ├── n2he/                       # Homomorphic encryption (N2HE)
│   │   ├── __init__.py
│   │   ├── core.py                 # N2HE core (LWE/RLWE/CKKS)
│   │   ├── adapter.py              # Encrypted LoRA adapter runtime
│   │   ├── inference.py            # Private inference mode
│   │   ├── keys.py                 # HE key management
│   │   ├── serialization.py        # Ciphertext serialization
│   │   ├── benchmark.py            # HE benchmarking utilities
│   │   └── _native.py              # Native library bindings (optional)
│   ├── platform/                   # Server platform
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry
│   │   ├── auth.py                 # Authentication
│   │   ├── database.py             # Database models
│   │   ├── dependencies.py         # FastAPI dependencies
│   │   └── tg_tinker_api/          # TG-Tinker API implementation
│   │       ├── __init__.py
│   │       ├── routes.py           # API routes
│   │       ├── models.py           # API models
│   │       ├── audit.py            # Audit logging
│   │       ├── dp.py               # Differential privacy
│   │       ├── storage.py          # Artifact storage
│   │       ├── queue.py            # Job queue
│   │       ├── worker.py           # Background workers
│   │       ├── cli.py              # CLI interface
│   │       └── tgsp_bridge.py      # TGSP integration
│   ├── telemetry/                  # Observability
│   │   ├── __init__.py
│   │   └── compliance_events.py    # Compliance event logging
│   └── tgsp/                       # TensorGuard Secure Package
│       ├── __init__.py
│       ├── cli.py                  # TGSP CLI (build/open)
│       ├── service.py              # TGSP service layer
│       ├── manifest.py             # Package manifest
│       ├── container.py            # Container format
│       ├── format.py               # Wire format
│       ├── crypto.py               # TGSP cryptography
│       ├── hpke_v03.py             # HPKE v0.3 support
│       ├── policy.py               # Policy enforcement
│       ├── spec.py                 # TGSP specification
│       └── tar_deterministic.py    # Deterministic tar creation
│
└── tg_tinker/                      # TG-Tinker SDK
    ├── __init__.py                 # SDK exports
    ├── client.py                   # ServiceClient
    ├── training_client.py          # TrainingClient
    ├── config.py                   # SDK configuration
    ├── schemas.py                  # Pydantic schemas
    ├── futures.py                  # FutureHandle for async ops
    └── exceptions.py               # SDK exceptions
```

## Module Descriptions

### tensorguard.crypto

Classical and post-quantum cryptographic primitives.

| Module | Purpose |
|--------|---------|
| `kem.py` | AES-256-GCM key encapsulation with HKDF |
| `sig.py` | Ed25519 digital signatures |
| `payload.py` | Encrypted payload container |
| `pqc/agility.py` | Abstract interface for PQC algorithms |
| `pqc/kyber.py` | ML-KEM-768 (NIST FIPS 203) via liboqs |
| `pqc/dilithium.py` | ML-DSA-65 (NIST FIPS 204) via liboqs |

**Dependencies:** `cryptography`, `liboqs-python` (optional for PQC)

### tensorguard.n2he

Neural Network Homomorphic Encryption integration.

| Module | Purpose |
|--------|---------|
| `core.py` | N2HE schemes (LWE, RLWE, CKKS) and context |
| `adapter.py` | Encrypted LoRA adapter runtime |
| `inference.py` | Private inference mode |
| `keys.py` | HE key generation and management |
| `serialization.py` | Ciphertext serialization formats |
| `benchmark.py` | Performance benchmarking |

**Status:** Simulation mode - not production HE. See `_native.py` for planned native bindings.

### tensorguard.tgsp

TensorGuard Secure Package format for encrypted model distribution.

| Module | Purpose |
|--------|---------|
| `cli.py` | Build and open TGSP packages |
| `service.py` | High-level TGSP API |
| `manifest.py` | Package manifest with privacy claims |
| `container.py` | Package container format |
| `crypto.py` | Package encryption |
| `hpke_v03.py` | Hybrid Public Key Encryption |
| `policy.py` | Access policy enforcement |

### tensorguard.evidence

Audit trail and compliance evidence.

| Module | Purpose |
|--------|---------|
| `canonical.py` | Deterministic serialization for hashing |
| `store.py` | Hash-chained evidence store |

### tensorguard.platform

Server platform for TG-Tinker API.

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI application |
| `auth.py` | JWT authentication |
| `database.py` | SQLModel database |
| `tg_tinker_api/*` | API implementation |

**Dependencies:** `sqlmodel`, `fastapi`, `uvicorn`

### tg_tinker

Python SDK for TenSafe/TG-Tinker API.

| Module | Purpose |
|--------|---------|
| `client.py` | ServiceClient - main entry point |
| `training_client.py` | TrainingClient - training operations |
| `schemas.py` | Request/response schemas |
| `futures.py` | Async operation handles |
| `exceptions.py` | SDK exceptions |

## Import Graph

```
tg_tinker (SDK)
    └── httpx

tensorguard.platform
    ├── tensorguard.tgsp
    └── tensorguard.crypto

tensorguard.tgsp
    ├── tensorguard.evidence
    └── tensorguard.crypto

tensorguard.n2he
    └── numpy

tensorguard.crypto.pqc
    └── liboqs (optional)
```

## Optional Dependencies

| Extra | Packages | Required For |
|-------|----------|--------------|
| `[pqc]` | liboqs-python | Post-quantum cryptography |
| `[server]` | sqlmodel, alembic | Platform server |
| `[bench]` | scipy, scikit-learn | Benchmarking |
| `[fl]` | flwr, tenseal | Federated learning |
| `[dev]` | pytest, ruff, mypy | Development |

## Notes

1. **N2HE is simulation mode**: The current N2HE implementation provides API compatibility but does not perform real homomorphic encryption. See PHASE4 documentation for plans.

2. **PQC requires liboqs**: Post-quantum algorithms require the liboqs native library and Python bindings. Without them, PQC operations will raise errors.

3. **Platform requires database**: The server platform requires SQLModel and a database. For development, SQLite is sufficient.
