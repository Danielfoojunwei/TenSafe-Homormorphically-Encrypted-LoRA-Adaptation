# TensorGuardFlow Regression Matrix

**Version:** 1.0
**Date:** 2026-01-27
**Scope:** Core (Control Plane + Trust + Dashboard + Integrations + N2HE)

---

## Overview

This matrix defines the critical invariants that must hold across all TensorGuardFlow releases. Each invariant maps to specific endpoints, test cases, and acceptance criteria.

---

## Invariant Matrix

### I1: Route Lifecycle Works

| Attribute | Value |
|-----------|-------|
| **Feature** | Complete route lifecycle |
| **Description** | Create route -> attach feed -> set policy -> run_once -> candidate -> gates |
| **Endpoints** | `POST /tgflow/routes`, `POST /routes/{key}/feed`, `POST /routes/{key}/policy`, `POST /routes/{key}/run_once` |
| **Test Name** | `test_route_happy_path.py::test_complete_route_lifecycle` |
| **Pass Criteria** | Route transitions through all states; adapter produced; events recorded in timeline |
| **Failure Signals** | Missing timeline events; adapter not registered; route stuck in intermediate state |
| **Severity** | P0 - Blocker |
| **Dependencies** | Database, IntegrationManager, PeftWorkflow |

### I2: Promotion Gating Enforced

| Attribute | Value |
|-----------|-------|
| **Feature** | Quality gates block bad adapters |
| **Description** | Failing forgetting/regression scores MUST block promotion |
| **Endpoints** | `POST /routes/{key}/run_once`, `POST /routes/{key}/promote` |
| **Test Name** | `test_gates_block_promotion.py::test_failing_gates_block_auto_promotion` |
| **Pass Criteria** | Adapter with forgetting > budget stays CANDIDATE; no promotion event recorded |
| **Failure Signals** | Adapter promoted despite failing gates; PROMOTED event with failed gates |
| **Severity** | P0 - Blocker |
| **Dependencies** | Policy engine, EvaluationService |

### I3: Rollback Instant + Correct

| Attribute | Value |
|-----------|-------|
| **Feature** | Rollback restores stable adapter |
| **Description** | Rollback changes active adapter ID; resolve returns stable adapter post-rollback |
| **Endpoints** | `POST /routes/{key}/rollback`, `POST /tgflow/resolve` |
| **Test Name** | `test_rollback_correctness.py::test_rollback_restores_fallback` |
| **Pass Criteria** | active_adapter_id == fallback_adapter_id after rollback; resolve returns correct adapter |
| **Failure Signals** | Wrong adapter returned; fallback_adapter_id unchanged; resolve inconsistent |
| **Severity** | P0 - Blocker |
| **Dependencies** | ContinuousRegistryService |

### I4: Evidence Chain Integrity

| Attribute | Value |
|-----------|-------|
| **Feature** | TGSP + evidence generated every run |
| **Description** | Every successful run produces TGSP package and evidence records; tamper detection works |
| **Endpoints** | `POST /routes/{key}/run_once`, `/community/tgsp/verify` |
| **Test Name** | `test_evidence_and_tgsp_generated.py::test_evidence_chain_complete` |
| **Pass Criteria** | TGSP path in artifacts; evidence events in timeline; verification passes |
| **Failure Signals** | Missing TGSP; no PACKAGED event; verification fails on valid package |
| **Severity** | P0 - Blocker |
| **Dependencies** | TGSPService, EvidenceStore |

### I5: Determinism (Stable Hashes)

| Attribute | Value |
|-----------|-------|
| **Feature** | Reproducible outputs in deterministic mode |
| **Description** | Same inputs produce stable manifest hash when TG_DETERMINISTIC=true |
| **Endpoints** | `POST /routes/{key}/run_once` |
| **Test Name** | `test_determinism_stable_hashes.py::test_deterministic_manifest_hash` |
| **Pass Criteria** | Two runs with identical inputs produce same content_hash |
| **Failure Signals** | Hash differs between runs; timestamps leak into hash |
| **Severity** | P1 - Major |
| **Dependencies** | tar_deterministic, TGSP manifest |

### I6: Integration Exporter Validity

| Attribute | Value |
|-----------|-------|
| **Feature** | Export specs are valid |
| **Description** | K8s Job, vLLM, TGI, Triton export schemas are valid and deployable |
| **Endpoints** | `POST /routes/{key}/export` |
| **Test Name** | `test_export_specs_valid.py::test_k8s_job_schema_valid` |
| **Pass Criteria** | Generated YAML/JSON validates against schema; contains required fields |
| **Failure Signals** | Missing required fields; invalid syntax; deployment would fail |
| **Severity** | P1 - Major |
| **Dependencies** | IntegrationConnectors |

### I7: N2HE Privacy Compliance

| Attribute | Value |
|-----------|-------|
| **Feature** | Privacy receipts and safe logging |
| **Description** | Privacy receipts emitted; no plaintext markers in logs when N2HE active |
| **Endpoints** | `POST /tgflow/resolve` |
| **Test Name** | `test_n2he_privacy_receipts.py::test_n2he_receipt_generated` |
| **Pass Criteria** | receipt_hash in response; logs contain [N2HE][PROTECTED] prefix; no plaintext vectors |
| **Failure Signals** | Missing receipt; plaintext logged; privacy_mode ignored |
| **Severity** | P1 - Major |
| **Dependencies** | N2HEProvider, SafeLogger |

### I8: Dashboard Bundle Schema

| Attribute | Value |
|-----------|-------|
| **Feature** | Dashboard data completeness |
| **Description** | Bundle endpoint returns all KPIs + topology snapshot with correct schema |
| **Endpoints** | `GET /metrics/routes/{key}/dashboard_bundle` |
| **Test Name** | `test_dashboard_bundle_schema.py::test_bundle_schema_complete` |
| **Pass Criteria** | Response contains summary, timeseries, events, topology; all required fields present |
| **Failure Signals** | Missing sections; null where required; schema validation fails |
| **Severity** | P2 - Minor |
| **Dependencies** | MetricsCollector, IntegrationManager |

---

## Additional Regression Tests

### Concurrency

| Test | Description | Endpoint | Pass Criteria |
|------|-------------|----------|---------------|
| `test_concurrent_run_once_same_route` | Two simultaneous run_once calls | `/run_once` | One succeeds, one queued/blocked; no corruption |
| `test_concurrent_promote_same_adapter` | Race condition on promotion | `/promote` | Only one promotion recorded |

### Failure Recovery

| Test | Description | Endpoint | Pass Criteria |
|------|-------------|----------|---------------|
| `test_training_failure_cleanup` | Training crashes mid-run | `/run_once` | FAILED event recorded; partial artifacts cleaned |
| `test_eval_failure_no_promotion` | Evaluation fails | `/run_once` | No PROMOTED event; route operable |
| `test_packaging_failure_recovery` | TGSP creation fails | `/run_once` | Error logged; system recoverable |

### Data Integrity

| Test | Description | Endpoint | Pass Criteria |
|------|-------------|----------|---------------|
| `test_tamper_blocks_promotion` | Modified TGSP rejected | `/verify` | Verification fails; promotion blocked |
| `test_timeline_ordering` | Events in correct order | `/timeline` | Timestamps monotonic; no gaps |
| `test_metrics_consistency` | Metrics match state | `/metrics/routes/summary` | KPIs reflect actual adapter count |

---

## Test Execution Matrix

| Phase | Tests | Command | Duration |
|-------|-------|---------|----------|
| Smoke | Health endpoints | `pytest tests/regression/test_health.py` | <10s |
| Regression | I1-I8 | `pytest tests/regression/ -v` | <2min |
| Concurrency | Race conditions | `pytest tests/regression/test_concurrent*.py` | <1min |
| Failure | Injection tests | `pytest tests/regression/test_failure*.py` | <1min |
| Performance | Latency budgets | `pytest tests/regression/test_*latency*.py` | <30s |
| Security | N2HE compliance | `pytest tests/security/ -v` | <1min |

---

## Coverage Requirements

### Minimum Coverage by Component

| Component | Required | Current | Gap |
|-----------|----------|---------|-----|
| continuous_endpoints.py | 90% | TBD | TBD |
| continuous_registry.py | 85% | TBD | TBD |
| orchestrator.py | 80% | TBD | TBD |
| n2he_provider.py | 90% | TBD | TBD |
| metrics_endpoints.py | 75% | TBD | TBD |

---

## Pass/Fail Summary Template

```
REGRESSION MATRIX STATUS - Run ID: XXXXXX
==========================================
I1 Route Lifecycle:      [ PASS / FAIL ]
I2 Promotion Gating:     [ PASS / FAIL ]
I3 Rollback Correct:     [ PASS / FAIL ]
I4 Evidence Chain:       [ PASS / FAIL ]
I5 Determinism:          [ PASS / FAIL ]
I6 Export Specs:         [ PASS / FAIL ]
I7 N2HE Privacy:         [ PASS / FAIL ]
I8 Dashboard Bundle:     [ PASS / FAIL ]
==========================================
Overall: X/8 Passed
```

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-27 | Initial matrix creation | QA Engineering |
