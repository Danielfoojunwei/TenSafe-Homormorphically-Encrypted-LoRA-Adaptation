# Repo Cleanup Rules

**Strict Guidelines for the Scope-Hardening Phase**

## 1. Safety First
- **Always Keep**: `src/tensorguard/tgsp`, `src/tensorguard/evidence`, `src/tensorguard/privacy` (N2HE).
- **Never Delete**: Anything without checking for imports or router links first.

## 2. N2HE Protection
- N2HE examples, tests, and providers must remain functional.
- It is a core USP. Do not "de-scope" N2HE.

## 3. MOAI Exclusion
- MOAI (Multi-Modal Orchestration AI) is **NOT** part of TensorGuardFlow Core.
- It must be removed from:
    - Default API Router
    - Default UI Navigation
    - Shipping build paths
- If kept for reference, move to `src/tensorguard/labs/moai`.

## 4. Workflows Preservation
- The **Continuous Learning Loop** (Ingest->Train->Promote) must not break.
- Existing tests `tests/integration/test_route_run_once_end_to_end.py` must pass after every major delete.

## 5. Binary Hygiene
- No `.mp4`, `.zip`, `.tgsp`, `.pt` committed to repo.
- Use `fixtures/` with tiny (<5MB) files for tests.

## 6. Labs Strategy
- Do not delete interesting R&D code if it has value.
- Move it to `src/tensorguard/labs/`.
- Ensure `from tensorguard.core` does NOT import `tensorguard.labs`.
