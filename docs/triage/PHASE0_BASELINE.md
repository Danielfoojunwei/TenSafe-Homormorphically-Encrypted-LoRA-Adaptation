# PHASE 0 - Baseline Assessment

**Date:** 2026-01-30
**Python Version:** 3.11.14
**Branch:** claude/integrate-n2he-tensafe-Rv6jq

## Environment Setup

```bash
python -m venv /tmp/tensafe_baseline_venv
source /tmp/tensafe_baseline_venv/bin/activate
python -V                          # Python 3.11.14
pip install -U pip setuptools wheel  # OK
pip install -r requirements.txt      # OK (with warning)
pip install -e .                     # OK
pytest -q                            # 11 ERRORS during collection
```

## Installation Results

### 1. requirements.txt Installation

**Status:** SUCCESS (with encoding issue)

**Observed Issue:**
```
$ file requirements.txt
Unicode text, UTF-16, little-endian text, with CRLF line terminators
```

**Root Cause:** requirements.txt is UTF-16LE encoded with BOM and CRLF line endings. While pip on Linux can sometimes handle this, it's non-standard and may fail on some systems.

**Fix Plan:** Convert requirements.txt to UTF-8 without BOM, with LF line endings.

### 2. Editable Install (`pip install -e .`)

**Status:** SUCCESS

Package installed correctly as `tensafe==3.0.0`.

### 3. pytest Collection

**Status:** 11 ERRORS during collection

## Test Collection Errors

| Test File | Error | Root Cause | Fix Plan |
|-----------|-------|------------|----------|
| `tests/regression/test_concurrent_run_once.py` | `ModuleNotFoundError: No module named 'conftest'` | Direct import of `conftest` instead of using pytest fixtures | Fix import to use relative path or pytest auto-discovery |
| `tests/regression/test_dashboard_bundle_schema.py` | `ModuleNotFoundError: No module named 'conftest'` | Same as above | Same as above |
| `tests/regression/test_evidence_and_tgsp_generated.py` | `ModuleNotFoundError: No module named 'conftest'` | Same as above | Same as above |
| `tests/regression/test_failure_injection.py` | `ModuleNotFoundError: No module named 'conftest'` | Same as above | Same as above |
| `tests/regression/test_gates_block_promotion.py` | `ModuleNotFoundError: No module named 'conftest'` | Same as above | Same as above |
| `tests/regression/test_n2he_privacy_receipts.py` | `ModuleNotFoundError: No module named 'conftest'` | Same as above | Same as above |
| `tests/regression/test_rollback_correctness.py` | `ModuleNotFoundError: No module named 'conftest'` | Same as above | Same as above |
| `tests/unit/test_crypto.py` | `ModuleNotFoundError: No module named 'tensorguard.core'` | Test imports non-existent `tensorguard.core.crypto` module | Quarantine test OR create missing module |
| `tests/unit/test_rdp_accountant.py` | `ModuleNotFoundError: No module named 'tensorguard.core'` | Test imports non-existent `tensorguard.core.privacy.rdp_accountant` | Quarantine test OR create missing module |
| `tests/unit/test_tgsp_core.py` | `ModuleNotFoundError: No module named 'tensorguard.evidence'` | `tgsp/manifest.py` imports `tensorguard.evidence.canonical` which doesn't exist | Create missing `evidence` module OR refactor manifest |
| `tests/unit/tg_tinker/test_schemas.py` | `ModuleNotFoundError: No module named 'tg_tinker.schemas'` | Module path issue - should be `src/tg_tinker/schemas.py` | Fix import or package config |

## Module Graph Analysis

### Existing Modules (under `src/tensorguard/`)

| Module | Status |
|--------|--------|
| `tensorguard/__init__.py` | EXISTS |
| `tensorguard/crypto/` | EXISTS (kem.py, sig.py, payload.py, pqc/) |
| `tensorguard/edge/` | EXISTS (tgsp_client.py) |
| `tensorguard/n2he/` | EXISTS (core.py, adapter.py, keys.py, etc.) |
| `tensorguard/platform/` | EXISTS (main.py, auth.py, database.py, tg_tinker_api/) |
| `tensorguard/telemetry/` | EXISTS (compliance_events.py) |
| `tensorguard/tgsp/` | EXISTS (cli.py, manifest.py, service.py, etc.) |

### Missing Modules (referenced but non-existent)

| Module | Referenced By | Fix Plan |
|--------|---------------|----------|
| `tensorguard.core` | tests/unit/test_crypto.py, tests/unit/test_rdp_accountant.py | Quarantine tests; these reference a different codebase structure |
| `tensorguard.core.crypto` | tests/unit/test_crypto.py | Actual crypto is at `tensorguard.crypto` |
| `tensorguard.core.privacy.rdp_accountant` | tests/unit/test_rdp_accountant.py | Module doesn't exist in this repo |
| `tensorguard.evidence` | tgsp/manifest.py | **CRITICAL:** Production code depends on this |
| `tensorguard.evidence.canonical` | tgsp/manifest.py | Need to create or stub |
| `tensorguard.evidence.store` | Unknown (presumed) | May be needed |

### Import Chain Analysis

```
tests/unit/test_tgsp_core.py
  └── tensorguard.tgsp.service (TGSPService)
        └── tensorguard.tgsp.cli (run_build, run_open)
              └── tensorguard.tgsp.manifest (PackageManifest)
                    └── tensorguard.evidence.canonical  ← MISSING!
```

## Critical Issues Summary

### Priority 1 - Blocking Production Use

1. **Missing `tensorguard.evidence` module**
   - `tgsp/manifest.py` imports `from ..evidence.canonical import canonical_bytes`
   - This breaks the TGSP (TensorGuard Secure Package) core functionality
   - **Fix:** Create minimal `evidence/` package with `canonical.py`

### Priority 2 - Test Infrastructure

2. **Broken conftest imports in regression tests**
   - 7 regression tests use `from conftest import async_iter_mock`
   - Should use pytest's auto-discovery
   - **Fix:** Move `async_iter_mock` to proper conftest.py or use relative import

3. **Tests reference non-existent modules**
   - `tensorguard.core.crypto` → actual: `tensorguard.crypto`
   - `tensorguard.core.privacy.rdp_accountant` → doesn't exist
   - **Fix:** Quarantine these tests, write new tests for actual modules

### Priority 3 - Packaging

4. **requirements.txt encoding**
   - UTF-16LE with BOM is non-standard
   - **Fix:** Convert to UTF-8

5. **tg_tinker module import path**
   - Test imports `tg_tinker.schemas` but package is at `src/tg_tinker/`
   - **Fix:** Verify pyproject.toml package discovery

## Verification Commands

After fixes, run:

```bash
# Clean venv test
rm -rf /tmp/test_venv && python -m venv /tmp/test_venv
source /tmp/test_venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt  # Should succeed
pip install -e .                 # Should succeed
pytest -q                        # Should collect without errors
```

## Next Steps

1. **PHASE 1:** Fix requirements.txt encoding, verify pyproject.toml
2. **PHASE 2:** Create missing `tensorguard.evidence` module, fix imports, quarantine broken tests
3. **PHASE 3:** TGSP security hardening
4. **PHASE 4:** HE runtime truthfulness
5. **PHASE 5:** Platform server bootability
6. **PHASE 6:** Quality gates
