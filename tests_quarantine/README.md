# Quarantined Tests

This directory contains tests that have been quarantined because they reference
modules or fixtures that do not exist in the current codebase.

**These tests are excluded from pytest collection via `norecursedirs` in pytest.ini.**

## Why Quarantine?

Rather than deleting tests or making them pass with stubs, we quarantine them to:
1. Preserve the test logic for future reference
2. Document what functionality was intended
3. Allow restoration once the underlying modules are implemented

## Quarantined Tests

### Unit Tests (`unit/`)

| Test File | Issue | Required Module |
|-----------|-------|-----------------|
| `test_crypto.py` | `ModuleNotFoundError: tensorguard.core` | Tests reference `tensorguard.core.crypto` but actual crypto is at `tensorguard.crypto` |
| `test_rdp_accountant.py` | `ModuleNotFoundError: tensorguard.core` | Tests reference `tensorguard.core.privacy.rdp_accountant` which doesn't exist |

### Regression Tests (`regression/`)

| Test File | Issue | Notes |
|-----------|-------|-------|
| `test_concurrent_run_once.py` | `from conftest import async_iter_mock` | Broken import - conftest should be auto-discovered |
| `test_dashboard_bundle_schema.py` | `from conftest import async_iter_mock` | Same issue |
| `test_evidence_and_tgsp_generated.py` | `from conftest import async_iter_mock` | Same issue |
| `test_failure_injection.py` | `from conftest import async_iter_mock` | Same issue |
| `test_gates_block_promotion.py` | `from conftest import async_iter_mock` | Same issue |
| `test_n2he_privacy_receipts.py` | `from conftest import async_iter_mock` | Same issue |
| `test_rollback_correctness.py` | `from conftest import async_iter_mock` | Same issue |

## How to Restore

To restore a quarantined test:

1. Identify the missing module/fixture
2. Either implement the module or update the test imports
3. Move the test back to `tests/`
4. Verify it passes with `pytest tests/path/to/test.py -v`

## Date Quarantined

2026-01-30 - Initial quarantine during PHASE 2 of repo cleanup
