# TenSafe Architecture

**Version**: 3.0.0
**Last Updated**: 2026-01-28

## Overview

TenSafe is a unified privacy-first ML platform that integrates four core subsystems:

1. **TenSafe Training API** - Privacy-preserving model fine-tuning
2. **TSSP Secure Packaging** - Cryptographically protected model distribution
3. **Platform Control Plane** - Fleet management and policy enforcement
4. **Edge Agent** - Secure deployment and attestation

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              TenSafe Platform                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌────────────────────────────────────────────────────────────────────────────┐    │
│   │                         Client Layer (tensafe SDK)                          │    │
│   │  ┌─────────────┐  ┌──────────────────┐  ┌────────────────┐                 │    │
│   │  │ServiceClient│──▶│ TrainingClient   │──▶│  FutureHandle  │                 │    │
│   │  └─────────────┘  │ • forward_backward│  │  • status()    │                 │    │
│   │                   │ • optim_step      │  │  • result()    │                 │    │
│   │                   │ • sample          │  │  • cancel()    │                 │    │
│   │                   │ • save_state      │  └────────────────┘                 │    │
│   │                   │ • load_state      │                                     │    │
│   │                   └──────────────────┘                                     │    │
│   └────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                             │
│                                        ▼ HTTPS/TLS 1.3                              │
│   ┌────────────────────────────────────────────────────────────────────────────┐    │
│   │                       Server Layer (tensafe.platform)                       │    │
│   │                                                                             │    │
│   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │    │
│   │  │ TenSafe API   │  │ Platform API  │  │  TSSP API     │  │ Telemetry   │ │    │
│   │  │ /v1/training_ │  │ /api/v1/      │  │ /api/tssp/    │  │ /api/v1/    │ │    │
│   │  │    clients    │  │ attestation   │  │ upload        │  │ telemetry   │ │    │
│   │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └──────┬──────┘ │    │
│   │          │                  │                  │                 │        │    │
│   │          ▼                  ▼                  ▼                 ▼        │    │
│   │  ┌─────────────────────────────────────────────────────────────────────┐  │    │
│   │  │                      Core Services Layer                             │  │    │
│   │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │  │    │
│   │  │  │ Job Queue│  │DP Engine │  │Key Mgmt  │  │Audit Log │            │  │    │
│   │  │  │ (Async)  │  │(RDP/PRV) │  │(KEK/DEK) │  │(Hash-    │            │  │    │
│   │  │  │          │  │          │  │          │  │ chain)   │            │  │    │
│   │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │  │    │
│   │  └─────────────────────────────────────────────────────────────────────┘  │    │
│   │                                    │                                       │    │
│   │                                    ▼                                       │    │
│   │  ┌─────────────────────────────────────────────────────────────────────┐  │    │
│   │  │                     Storage Layer                                    │  │    │
│   │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │    │
│   │  │  │Encrypted     │  │  Database    │  │ TSSP Package │              │  │    │
│   │  │  │Artifact Store│  │  (SQLite/    │  │   Registry   │              │  │    │
│   │  │  │(AES-256-GCM) │  │   Postgres)  │  │              │              │  │    │
│   │  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │    │
│   │  └─────────────────────────────────────────────────────────────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                             │
│                                        ▼ TSSP Package                               │
│   ┌────────────────────────────────────────────────────────────────────────────┐    │
│   │                         Edge Layer (tensafe.agent)                          │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │    │
│   │  │  Identity   │  │ Attestation │  │    TSSP     │  │   Runtime   │       │    │
│   │  │  Manager    │  │   Verifier  │  │   Loader    │  │  (TensorRT) │       │    │
│   │  │  (mTLS)     │  │   (TPM)     │  │             │  │             │       │    │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │    │
│   └────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
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

### 2. TSSP Secure Packaging

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

### 3. Security Architecture

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

### 4. Audit Trail Architecture

#### Hash Chain Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hash-Chained Audit Log                        │
└─────────────────────────────────────────────────────────────────┘

┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Entry 0    │    │ Entry 1    │    │ Entry 2    │    │ Entry N    │
│ (Genesis)  │───▶│            │───▶│            │───▶│            │
├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤
│ prev_hash: │    │ prev_hash: │    │ prev_hash: │    │ prev_hash: │
│ GENESIS    │    │ hash(E0)   │    │ hash(E1)   │    │ hash(E[N-1])
├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤
│ operation  │    │ operation  │    │ operation  │    │ operation  │
│ tenant_id  │    │ tenant_id  │    │ tenant_id  │    │ tenant_id  │
│ timestamp  │    │ timestamp  │    │ timestamp  │    │ timestamp  │
│ artifacts  │    │ artifacts  │    │ artifacts  │    │ artifacts  │
├────────────┤    ├────────────┤    ├────────────┤    ├────────────┤
│ record_hash│    │ record_hash│    │ record_hash│    │ record_hash│
│ = SHA256(  │    │ = SHA256(  │    │ = SHA256(  │    │ = SHA256(  │
│   entry +  │    │   entry +  │    │   entry +  │    │   entry +  │
│   prev_hash│    │   prev_hash│    │   prev_hash│    │   prev_hash│
│ )          │    │ )          │    │ )          │    │ )          │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
```

#### Tamper Detection

If any entry is modified:
1. Its `record_hash` will no longer match `SHA256(entry + prev_hash)`
2. All subsequent entries will have broken chain links
3. Verification fails immediately

---

### 5. Differential Privacy Architecture

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

---

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    Module Dependency Graph                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌───────────────┐
                    │    tensafe    │
                    │     (SDK)     │
                    └───────┬───────┘
                            │ imports
                            ▼
        ┌───────────────────────────────────────┐
        │           tensafe.platform            │
        │  ┌─────────────────────────────────┐  │
        │  │         tensafe_api             │  │
        │  │  routes, storage, audit, dp     │  │
        │  └───────────────┬─────────────────┘  │
        │                  │ imports             │
        │  ┌───────────────▼─────────────────┐  │
        │  │              api                │  │
        │  │  attestation, enablement, etc.  │  │
        │  └─────────────────────────────────┘  │
        └───────────────────┬───────────────────┘
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

### Production Multi-Node

```
┌─────────────────────────────────────────────────────────────────┐
│                      Production Deployment                       │
└─────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   Load Balancer │
                    │   (TLS Term)    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   API Pod 1   │    │   API Pod 2   │    │   API Pod N   │
│  tensafe      │    │  tensafe      │    │  tensafe      │
│   server      │    │   server      │    │   server      │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  PostgreSQL   │    │    Redis      │    │  HashiCorp    │
│  (Primary)    │    │  (Job Queue)  │    │    Vault      │
└───────────────┘    └───────────────┘    └───────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │   S3/GCS      │
                    │ Artifact Store│
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

---

## Related Documentation

- [TENSAFE_SPEC.md](TENSAFE_SPEC.md) - TenSafe API specification
- [TSSP_SPEC.md](TSSP_SPEC.md) - Secure packaging format
- [SECURITY.md](../SECURITY.md) - Security policy
