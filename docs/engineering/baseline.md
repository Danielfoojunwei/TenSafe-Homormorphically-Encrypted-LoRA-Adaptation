# TenSafe Phase 0 Baseline Assessment

**Date:** 2026-02-04
**Commit:** 01b776f (HEAD)
**Branch:** claude/tensafe-system-remediation-xhJKs

## 1. Repository Structure Summary

### Package Layout
- **`tensafe/`** (root): 67 Python files - core LoRA training/inference framework
- **`src/tensorguard/`**: 109 Python files - server platform, API, auth
- **`src/tg_tinker/`**: 7 Python files - privacy-first training client SDK
- **`he_lora_microkernel/`**: 88 Python files - HE compilation and execution
- **`crypto_backend/ckks_moai/`**: CKKS cryptographic backend

### Build Metadata
- **pyproject.toml**: Package name `tensafe`, version 4.0.0
- **Build system**: setuptools 61.0+
- **Python versions**: 3.9, 3.10, 3.11, 3.12

### Key Files
- **Dockerfile**: Multi-stage build (Python 3.11-slim)
- **requirements.txt**: Pinned dependencies
- **pytest.ini**: Test configuration
- **Makefile**: 271 lines of targets

---

## 2. Baseline Check Results

### 2.1 Dependency Installation

```bash
$ python -m pip install -U pip setuptools wheel
# ERROR: Cannot uninstall wheel 0.42.0, RECORD file not found (Debian package conflict)
```

### 2.2 Package Installation

```bash
$ python -m venv /tmp/test_venv && source /tmp/test_venv/bin/activate
$ pip install .
# Result: SUCCESS (with warnings)

Building wheels for collected packages: tensafe
  Created wheel for tensafe: filename=tensafe-4.0.0-py3-none-any.whl
Successfully installed tensafe-4.0.0
```

**Critical Warnings During Import:**
```
WARNING:tensorguard.platform.database:DATABASE_URL not set, using local SQLite (NOT FOR PRODUCTION)
WARNING:tensorguard.platform.tg_tinker_api.storage:KeyManager using generated master key - use KMS in production
WARNING:tensorguard.platform.auth:SECURITY WARNING: TG_SECRET_KEY not set. Generating ephemeral key
WARNING:tensorguard.admin.permissions:SECURITY WARNING: TG_ADMIN_SECRET_KEY not set. Using ephemeral key
```

### 2.3 Module Import Test

```bash
$ python -c "from tensorguard.platform.main import app; print('OK')"
# Result: OK (with warnings above)

$ python -c "import tensafe; print('OK')"
# Result: OK (from current directory, NOT from installed package)
```

**CRITICAL ISSUE:** The `tensafe` package imports successfully only because Python's current working directory is added to `sys.path`. After actual installation, `tensafe/` is NOT packaged (pyproject.toml only packages from `src/`).

### 2.4 Pytest Collection

```bash
$ python -m pytest -q --co
collected 451 items / 28 errors
```

**Collection Errors:**
1. `ModuleNotFoundError: No module named 'numpy'` - 15 files
2. `ModuleNotFoundError: No module named 'fastapi'` - 1 file
3. `pyo3_runtime.PanicException: Python API call failed` - 3 files (cryptography/cffi conflict)

### 2.5 Ruff Linting

```bash
$ python -m ruff check .
# Multiple issues found:
# - I001: Import block is un-sorted or un-formatted
# - F401: Unused imports
# - F541: f-string without placeholders
```

### 2.6 MyPy Type Checking

```bash
$ python -m mypy src/tensorguard --ignore-missing-imports
Found 4 errors in 3 files

src/tensorguard/tgsp/format.py: error: Source file found twice under different module names:
  "tensorguard.tgsp.format" and "src.tensorguard.tgsp.format"
```

### 2.7 Docker Build

```bash
$ docker build .
# Environment does not have Docker installed
# Dockerfile review shows CRITICAL issue:
# Line 26: pip install --no-cache-dir -e .  # EDITABLE install in builder!
# Line 37: COPY --from=builder /opt/venv /opt/venv  # Copies broken refs
```

**Dockerfile Problem:** Editable install in builder stage creates symlinks/references to `/build/src/` which don't exist in the production stage.

---

## 3. Critical Issues Identified

| ID | Severity | Issue | Impact |
|----|----------|-------|--------|
| PKG-01 | P0 | `tensafe/` package not included in wheel | Imports fail after pip install |
| PKG-02 | P0 | Dockerfile uses editable install | Container fails to start |
| STATE-01 | P0 | In-memory dicts for state | Data loss on restart/scaling |
| AUTH-01 | P0 | Demo tenancy via token hash | No real authentication |
| STORE-01 | P0 | Storage key format conflicts | Artifacts cannot be saved |
| QUEUE-01 | P0 | Process-local thread queue | No durability, no retry |
| HE-01 | P0 | Silent fallback to mock | Privacy violation undetected |
| VLLM-01 | P1 | Private API dependencies | Breaks on vLLM updates |
| OBS-01 | P1 | Limited observability | Hard to debug production |

---

## 4. Environment Details

- **Python:** 3.11.14
- **Platform:** Linux 4.4.0
- **pytest:** 9.0.2
- **ruff:** installed
- **mypy:** installed

---

## 5. Next Steps

1. Create `docs/engineering/issues.md` with detailed issue tracking
2. Phase 1: Fix packaging (merge `tensafe/` into `src/` or add to packages)
3. Phase 2: Fix Dockerfile (build wheel, not editable)
4. Continue through remediation phases

---

## Appendix: Key File Locations

### In-Memory State Dictionaries (routes.py:35-38)
```python
_training_clients: Dict[str, TinkerTrainingClient] = {}
_futures: Dict[str, TinkerFuture] = {}
_artifacts: Dict[str, TinkerArtifact] = {}
_dp_trainers: Dict[str, DPTrainer] = {}
```

### Demo Tenancy (routes.py:234)
```python
# In production, validate token and extract tenant
# For demo, derive tenant from token hash
tenant_id = f"tenant-{hashlib.sha256(token.encode()).hexdigest()[:8]}"
```

### Storage Key Issue (storage.py:159)
```python
storage_key = f"{tenant_id}/{training_client_id}/{artifact_id}"
# LocalStorageBackend rejects keys with "/" (storage.py:69)
```

### vLLM Mock Fallback (adapter.py:87-89)
```python
except ImportError:
    # Fallback for testing without vLLM
    logger.warning("vLLM not installed, using mock implementation")
    return self._init_mock()
```
