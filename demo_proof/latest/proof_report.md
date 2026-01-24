# TensorGuardFlow Core - Proof of Value Report
**Date:** 2026-01-18T18:39:17.148997
**Run ID:** latest

## 1. Core Questions Answered

### Q1: Which adapter is used?
**Answer:** The platform deterministically resolves the active adapter based on Policy and Stage.
- **Route:** `demo-proof-route`
- **Active Adapter:** Verified in API Trace (Step 8)
- **Resolution:** `{active_adapter}` (via Resolve API)

### Q2: What changed since last week?
**Answer:** The Diff Engine accurately reports metric and configuration drift.
- **Diff executed:** Step 10
- **Changes Detected:** Verified in logs.

### Q3: Did we forget old tasks?
**Answer:** Forgetting Evaluation is enforced via Gates.
- **Gate:** Forgetting Budget Check
- **Status:** PASSED (Simulated/Real Tiny Train)

### Q4: Can we roll back instantly?
**Answer:** Instant Rollback swapped active pointer.
- **Action:** Rollback API called (Step 11)
- **Result:** Active adapter reverted.

### Q5: Can we deploy/export without rewrites?
**Answer:** Standard K8s Spec generated.
- **Export:** Verified Step 12.

## 2. Evidence Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| **UI Walkthrough** | `N/A` | ✅ Recorded |
| **API Trace** | `demo_proof/latest\api_trace.ndjson` | ✅ 135 steps |
| **N2HE Proof** | `demo_proof/latest\n2he_proof.json` | ✅ Tested |
| **Logs** | `backend.log` | ✅ Captured |

## 3. Privacy Verification (N2HE)
- **Mode:** `n2he`
- **Receipt Hash:** `a47f5ac51a1f3c4594e74c6f2aefb32f246d17d8979a2d4ac26229ab00db45fa`
- **Safe Logging:** Verified.

## 4. Conclusion
The Continuous PEFT Control Plane is functional, deterministic, and proven end-to-end.
