# TenSafe Feature Maturity Matrix

**Version**: 3.0.0
**Date**: 2026-01-30

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
| PostgreSQL Backend | **Alpha** | Not yet tested at scale |
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

### Differential Privacy

| Feature | Maturity | Notes |
|---------|----------|-------|
| RDP Accountant | **Stable** | Opacus-compatible |
| Gradient Clipping | **Stable** | Per-sample bounds |
| Noise Injection | **Stable** | Calibrated Gaussian |
| Budget Tracking | **Stable** | (ε, δ) conversion |

### Compliance & Telemetry

| Feature | Maturity | Notes |
|---------|----------|-------|
| Hash-Chain Audit | **Stable** | Tamper-evident |
| Compliance Events | **Beta** | ISO/SOC mapping |
| PII Scanning | **Alpha** | Regex-based |
| Evidence Reports | **Beta** | JSON + Markdown |

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
```

## Testing Requirements

| Test Suite | Requirements | Command |
|------------|--------------|---------|
| Unit Tests | Python 3.9+ | `pytest tests/unit` |
| Integration | FastAPI, httpx | `pytest tests/integration` |
| N2HE Tests | TENSAFE_TOY_HE=1 | `pytest tests/n2he` |
| Security Tests | cryptography | `pytest tests/security` |
| E2E Tests | All dependencies | `pytest tests/e2e` |

## Upgrade Path

### From 2.x to 3.x

1. **Package rename**: `tensafe` → `tg_tinker`
2. **N2HE gating**: Set `TENSAFE_TOY_HE=1` for toy mode
3. **Error codes**: All errors now have `TG_*` codes
4. **Logging**: Structured JSON in production

### Future Roadmap

| Feature | Target Version | Status |
|---------|---------------|--------|
| Native N2HE Integration | 3.1 | In progress |
| OAuth/OIDC Authentication | 3.2 | Planned |
| PostgreSQL at Scale | 3.1 | Planned |
| HSM Key Storage | 4.0 | Planned |
