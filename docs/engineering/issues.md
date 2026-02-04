# TenSafe Issue Register

**Last Updated:** 2026-02-04
**Baseline Commit:** 01b776f

---

## Issue Severity Legend

| Severity | Description |
|----------|-------------|
| **P0** | Blocks production deployment; must fix immediately |
| **P1** | High risk; fix before GA |
| **P2** | Medium risk; fix in next sprint |

---

## P0 Issues

### PKG-01: Root `tensafe/` Package Not Included in Wheel

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
After `pip install .` in a clean venv, importing `tensafe` fails unless running from repo root.

**Root Cause:**
`pyproject.toml` only packages from `src/`:
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["tensorguard*", "tg_tinker*"]
```

The root `tensafe/` directory (67 Python files) is not included.

**Repro Command:**
```bash
python -m venv /tmp/test && source /tmp/test/bin/activate
pip install .
cd /tmp
python -c "import tensafe"  # ModuleNotFoundError
```

**Proposed Fix:**
Option A: Move `tensafe/` to `src/tensafe/`
Option B: Update pyproject.toml to include root packages:
```toml
[tool.setuptools.packages.find]
where = ["src", "."]
include = ["tensorguard*", "tg_tinker*", "tensafe*"]
```

---

### PKG-02: Dockerfile Uses Editable Install

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
Docker container fails to import modules with `ModuleNotFoundError`.

**Root Cause:**
Dockerfile line 26:
```dockerfile
pip install --no-cache-dir -e .
```

Editable installs create symlinks/path entries pointing to `/build/src/` which doesn't exist in the production stage.

**Repro Command:**
```bash
docker build -t tensafe-test .
docker run tensafe-test python -c "from tensorguard.platform.main import app"
# ModuleNotFoundError
```

**Proposed Fix:**
Replace with wheel build and install:
```dockerfile
# Builder stage
RUN pip wheel . -w /wheels --no-deps

# Production stage
COPY --from=builder /wheels/*.whl /wheels/
RUN pip install --no-cache-dir /wheels/*.whl
```

---

### STATE-01: In-Memory State Dictionaries

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
All training clients, futures, artifacts, and DP trainers are lost on process restart. Multi-replica deployments have inconsistent state.

**Root Cause:**
`src/tensorguard/platform/tg_tinker_api/routes.py:35-38`:
```python
_training_clients: Dict[str, TinkerTrainingClient] = {}
_futures: Dict[str, TinkerFuture] = {}
_artifacts: Dict[str, TinkerArtifact] = {}
_dp_trainers: Dict[str, DPTrainer] = {}
```

**Repro Command:**
```bash
# Start server, create training client
curl -X POST localhost:8000/v1/training_clients -d '{"model_ref":"llama"}'
# Restart server
# Training client is gone
```

**Proposed Fix:**
1. Add PostgreSQL-backed metadata store
2. Define SQLAlchemy/SQLModel models
3. Replace dict lookups with DB queries
4. Add Alembic migrations

---

### AUTH-01: Demo Tenancy via Token Hash

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
Any arbitrary Bearer token is accepted. Tenant isolation is based on SHA256 hash of the raw token, not actual authentication.

**Root Cause:**
`src/tensorguard/platform/tg_tinker_api/routes.py:232-235`:
```python
# In production, validate token and extract tenant
# For demo, derive tenant from token hash
tenant_id = f"tenant-{hashlib.sha256(token.encode()).hexdigest()[:8]}"
return tenant_id
```

**Repro Command:**
```bash
curl -H "Authorization: Bearer anything" localhost:8000/v1/training_clients
# Returns empty list (valid response - should be 401)
curl -H "Authorization: Bearer anything_else" localhost:8000/v1/training_clients
# Different tenant, also accepted
```

**Proposed Fix:**
1. Implement proper token validation against database
2. Store token hash (not raw token)
3. Validate token signature, expiration, issuer
4. Extract tenant from validated claims

---

### STORE-01: Storage Key Format Conflicts

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
Saving artifacts fails with `ValueError: Invalid storage key format`.

**Root Cause:**
`EncryptedArtifactStore` creates keys with `/`:
```python
# storage.py:159
storage_key = f"{tenant_id}/{training_client_id}/{artifact_id}"
```

But `LocalStorageBackend` rejects them:
```python
# storage.py:69
if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$', key):
    raise ValueError(f"Invalid storage key format: {key!r}")
```

**Repro Command:**
```python
from tensorguard.platform.tg_tinker_api.storage import *
backend = LocalStorageBackend()
backend.write("tenant/client/artifact", b"data")
# ValueError: Invalid storage key format: 'tenant/client/artifact'
```

**Proposed Fix:**
Option A: Modify LocalStorageBackend to support hierarchical keys (create subdirectories)
Option B: URL-safe base64 encode keys in EncryptedArtifactStore

---

### QUEUE-01: Process-Local Thread Queue

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
Jobs are lost on worker crash. No retry mechanism. No dead-letter queue. Multi-process workers don't share queue.

**Root Cause:**
The queue implementation uses a simple in-process structure without persistence.

**Repro Command:**
```bash
# Submit job, kill worker before completion
# Job is permanently lost
```

**Proposed Fix:**
1. Implement Redis-backed queue (RQ/ARQ)
2. Add visibility timeout and automatic retry
3. Add idempotency key support
4. Add dead-letter queue for poison messages

---

### HE-01: Silent Fallback to Mock Implementation

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
When vLLM is not installed, the adapter silently falls back to a mock implementation that provides no actual HE protection.

**Root Cause:**
`he_lora_microkernel/backend/vllm_adapter/adapter.py:86-89`:
```python
except ImportError:
    # Fallback for testing without vLLM
    logger.warning("vLLM not installed, using mock implementation")
    return self._init_mock()
```

No mechanism to enforce HE path in production.

**Repro Command:**
```python
# In environment without vLLM
from he_lora_microkernel.backend.vllm_adapter import VLLMAdapter
adapter = VLLMAdapter("model", batch_config)
adapter.init()  # Silent warning, returns mock
# User thinks they have HE protection, they don't
```

**Proposed Fix:**
1. Add `ExecutionPolicy` enum: `HE_REQUIRED`, `HE_PREFERRED`, `PLAINTEXT_ONLY`
2. Default privacy endpoints to `HE_REQUIRED`
3. Raise HTTP 503 when HE unavailable and policy is `HE_REQUIRED`
4. Add attestation field to all responses

---

### KEY-01: Silent Master Key Generation in Production

**Severity:** P0
**Status:** Open
**Owner:** TBD

**Symptom:**
KeyManager silently generates a random master key if none provided, even in production. Key is lost on restart, breaking artifact decryption.

**Root Cause:**
`src/tensorguard/platform/tg_tinker_api/storage.py:288-291`:
```python
if master_key is None:
    # In production, this should come from a secure vault
    self._master_key = secrets.token_bytes(32)
    logger.warning("KeyManager using generated master key - use KMS in production")
```

**Proposed Fix:**
1. Check `ENV` or `TG_ENVIRONMENT`
2. If `prod`/`production`, require explicit master key or KMS config
3. Fail-closed: refuse to start without proper key management

---

## P1 Issues

### VLLM-01: Private API Dependencies

**Severity:** P1
**Status:** Open
**Owner:** TBD

**Symptom:**
vLLM adapter accesses internal attributes that may change between versions.

**Root Cause:**
`he_lora_microkernel/backend/vllm_adapter/adapter.py:101`:
```python
self._model = self._engine.llm_engine.model_executor.driver_worker.model_runner.model
```

This deep attribute chain accesses private/internal APIs.

**Proposed Fix:**
1. Pin vLLM version explicitly
2. Add contract tests that verify attribute paths exist
3. Document supported vLLM versions
4. Add CI check for vLLM compatibility

---

### OBS-01: Limited Observability

**Severity:** P1
**Status:** Open
**Owner:** TBD

**Symptom:**
No structured logging, no metrics export, no trace correlation. Hard to debug production issues.

**Root Cause:**
Observability module exists but is not fully integrated. No health endpoints verify all dependencies.

**Proposed Fix:**
1. Add OpenTelemetry tracing and metrics
2. Implement structured JSON logging
3. Add comprehensive `/healthz` and `/readyz` endpoints
4. Create runbooks for common failure modes

---

### MYPY-01: Duplicate Module Detection

**Severity:** P1
**Status:** Open
**Owner:** TBD

**Symptom:**
```
src/tensorguard/tgsp/format.py: error: Source file found twice under different module names:
  "tensorguard.tgsp.format" and "src.tensorguard.tgsp.format"
```

**Root Cause:**
Mixed `src/` layout with pytest path configuration causes mypy to see modules twice.

**Proposed Fix:**
1. Use consistent package layout
2. Configure mypy with explicit package bases
3. Fix pytest.ini pythonpath setting

---

## P2 Issues

### LINT-01: Import Sorting and Unused Imports

**Severity:** P2
**Status:** Open
**Owner:** TBD

**Symptom:**
Ruff reports ~50 import-related issues.

**Proposed Fix:**
```bash
python -m ruff check . --fix
```

---

### TEST-01: Test Collection Errors

**Severity:** P2
**Status:** Open
**Owner:** TBD

**Symptom:**
28 test files fail to collect due to missing dependencies.

**Root Cause:**
Tests import numpy, fastapi directly without being in a properly configured environment.

**Proposed Fix:**
1. Add test dependencies to optional extras
2. Add skip markers for tests requiring heavy deps
3. Fix cffi/cryptography version conflict

---

## Resolution Tracking

| Issue | Phase | Status | PR |
|-------|-------|--------|-----|
| PKG-01 | 1 | Open | - |
| PKG-02 | 2 | Open | - |
| STATE-01 | 4 | Open | - |
| AUTH-01 | 6 | Open | - |
| STORE-01 | 3 | Open | - |
| QUEUE-01 | 5 | Open | - |
| HE-01 | 7 | Open | - |
| KEY-01 | 3 | Open | - |
| VLLM-01 | 8 | Open | - |
| OBS-01 | 9 | Open | - |
| MYPY-01 | 1 | Open | - |
| LINT-01 | - | Open | - |
| TEST-01 | - | Open | - |
