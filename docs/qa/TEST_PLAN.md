# QA Test Plan: TensorGuardFlow Core

**Version:** 1.0 (Candidate 2.3.0)
**Scope:** Continuous Learning Control Plane & Privacy Layer
**Status:** DRAFT

## 1. Introduction
This plan defines the certification strategy for TensorGuardFlow Core (v2.3.0). The primary goal is to ensure the **Continuous Learning Loop** operates reliably, securely, and deterministically, with full N2HE privacy support.

## 2. Scope

| In Scope (Core) | Out of Scope |
|-----------------|--------------|
| Continuous Learning Loop (Orchestrator) | MOAI (Multi-Modal Orchestration) |
| Route/Feed/Policy API | Robotics / Edge Agent logic |
| Adapter Registry & Lifecycle | Federation (unless specifically in Core) |
| N2HE Privacy Mode | Third-party cloud service validation |
| PEFT integration (Workflow) | UI Aesthetic details |
| TGSP Packaging & Evidence | |

## 3. Test Strategy

### 3.1 Backend Automation (Pytest)
- **API Tests**: Comprehensive coverage of all `/tgflow` endpoints.
- **Contract Tests**: Verify input/output schemas, error formats, and boundary conditions.
- **Integration**: "Smoke Test" script (`scripts/smoke_test_core.py`) verifying the 9-step loop.
- **Mocking**: External services (HF, Cloud) are mocked; Core logic is real.
- **Simulation**: Use `TG_SIMULATION=true` for deterministic logic verification.

### 3.2 N2HE Privacy QA
- **Mandatory**: Functional verification of Encrypted Routing.
- **Safety**: "Plaintext Leak Check" - grep logs/artifacts for sensitive strings.

### 3.3 UI Automation (Playwright)
- Focus on "Happy Path" user journeys.
- Verify critical state updates (Timeline, Status cards).
- Verify Privacy Mode indicators.

## 4. Test Environments
- **Local (Dev)**: `sqlite` + `TG_SIMULATION=true`. Primary certification target.
- **CI**: Automated regression suite.

## 5. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Plaintext Leak** | Critical (S0) | Automated SafeLogger tests; evidence audit. |
| **Determinism Drift** | High (S1) | Fix random seeds; hash verification tests. |
| **Upgrade Breakage** | Medium (S2) | Schema migration tests (if applicable). |

## 6. Success Criteria (Exit Gates)
1. 100% Pass on P0 API regression suite.
2. 100% Pass on P0 UI journeys.
3. Zero S0/S1 bugs open.
4. Privacy Verification: No plaintext leaks detected.
5. Determinism: Repeated runs produce identical manifest hashes.
