# TensorGuardFlow Architecture

**Version**: 3.0.0
**Last Updated**: 2026-01-28

## Overview

TensorGuardFlow is a unified privacy-first ML platform that integrates four core subsystems:

1. **TG-Tinker Training API** - Privacy-preserving model fine-tuning
2. **TGSP Secure Packaging** - Cryptographically protected model distribution
3. **Platform Control Plane** - Fleet management and policy enforcement
4. **Edge Agent** - Secure deployment and attestation

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              TensorGuardFlow Platform                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌────────────────────────────────────────────────────────────────────────────┐    │
│   │                         Client Layer (tg_tinker SDK)                        │    │
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
│   │                       Server Layer (tensorguard.platform)                   │    │
│   │                                                                             │    │
│   │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │    │
│   │  │ TG-Tinker API │  │ Platform API  │  │  TGSP API     │  │ Telemetry   │ │    │
│   │  │ /v1/training_ │  │ /api/v1/      │  │ /api/tgsp/    │  │ /api/v1/    │ │    │
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
│   │  │  │Encrypted     │  │  Database    │  │ TGSP Package │              │  │    │
│   │  │  │Artifact Store│  │  (SQLite/    │  │   Registry   │              │  │    │
│   │  │  │(AES-256-GCM) │  │   Postgres)  │  │              │              │  │    │
│   │  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │    │
│   │  └─────────────────────────────────────────────────────────────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                             │
│                                        ▼ TGSP Package                               │
│   ┌────────────────────────────────────────────────────────────────────────────┐    │
│   │                         Edge Layer (tensorguard.agent)                      │    │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │    │
│   │  │  Identity   │  │ Attestation │  │    TGSP     │  │   Runtime   │       │    │
│   │  │  Manager    │  │   Verifier  │  │   Loader    │  │  (TensorRT) │       │    │
│   │  │  (mTLS)     │  │   (TPM)     │  │             │  │             │       │    │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │    │
│   └────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. TG-Tinker Training API

The TG-Tinker subsystem provides privacy-first model fine-tuning with a Tinker-compatible API.

#### Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TG-Tinker Training Flow                                │
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
| `ServiceClient` | `src/tg_tinker/client.py` | Main entry point, manages HTTP sessions |
| `TrainingClient` | `src/tg_tinker/training_client.py` | Training primitives interface |
| `FutureHandle` | `src/tg_tinker/futures.py` | Async operation management |
| `DPTrainer` | `src/tensorguard/platform/tg_tinker_api/dp.py` | DP-SGD implementation |
| `RDPAccountant` | `src/tensorguard/platform/tg_tinker_api/dp.py` | Privacy budget tracking |
| `EncryptedArtifactStore` | `src/tensorguard/platform/tg_tinker_api/storage.py` | Per-tenant encrypted storage |
| `AuditLogger` | `src/tensorguard/platform/tg_tinker_api/audit.py` | Hash-chained audit trail |

---

### 2. TGSP Secure Packaging

TGSP provides cryptographically protected model distribution with post-quantum signatures.

#### Package Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TGSP Package Lifecycle                                 │
└──────────────────────────────────────────────────────────────────────────┘

  Training                    Packaging                   Edge
    │                            │                          │
    │  TG-Tinker checkpoint      │                          │
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
    │                            │  Package as .tgsp        │
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
    │                            │                          │  Decrypt & load
    │                            │                          │  └─ GPU memory
```

#### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `TGSPService` | `src/tensorguard/tgsp/service.py` | Package creation and verification |
| `sign_hybrid` | `src/tensorguard/crypto/sig.py` | Hybrid Ed25519+Dilithium3 signatures |
| `generate_hybrid_keypair` | `src/tensorguard/crypto/kem.py` | X25519+Kyber768 key generation |

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
                    │   tg_tinker   │
                    │     (SDK)     │
                    └───────┬───────┘
                            │ imports
                            ▼
        ┌───────────────────────────────────────┐
        │         tensorguard.platform          │
        │  ┌─────────────────────────────────┐  │
        │  │         tg_tinker_api           │  │
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
│    crypto     │   │   identity    │   │     tgsp      │
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
│  │      tensorguard server     │   │
│  │  ├─ TG-Tinker API (:8000)   │   │
│  │  ├─ Platform API            │   │
│  │  └─ SQLite DB               │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │      tg-tinker client       │   │
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
│ tensorguard   │    │ tensorguard   │    │ tensorguard   │
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
| `TG_ENVIRONMENT` | All | Yes (prod) | `development` |
| `TG_SECRET_KEY` | Platform | Yes (prod) | - |
| `TG_KEY_MASTER` | Identity | Yes (prod) | - |
| `DATABASE_URL` | Platform | Yes (prod) | `sqlite:///./tensorguard.db` |
| `TG_TINKER_API_KEY` | SDK | Yes | - |
| `TG_TINKER_BASE_URL` | SDK | No | `https://api.tensorguard.io` |
| `TG_PQC_REQUIRED` | Crypto | No | `false` |
| `TG_DETERMINISTIC` | Training | No | `false` |

---

## Related Documentation

- [PRIVACY_TINKER_SPEC.md](PRIVACY_TINKER_SPEC.md) - TG-Tinker API specification
- [TGSP_SPEC.md](TGSP_SPEC.md) - Secure packaging format
- [SECURITY.md](../SECURITY.md) - Security policy
- [HARDENING_REPORT.md](../HARDENING_REPORT.md) - Production hardening
