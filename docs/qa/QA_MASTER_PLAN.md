# TensorGuardFlow QA Master Plan

**Version:** 1.0
**Date:** 2026-01-27
**Run ID:** 20260127_014503
**Scope:** Core (Control Plane + Trust + Dashboard + Integrations + N2HE)

---

## 1. Executive Summary

This QA Master Plan defines the comprehensive quality assurance strategy for TensorGuardFlow Core. The plan covers:

- **Regression Testing**: Continuous learning workflow invariants
- **Performance Engineering**: Latency baselines and bottleneck detection
- **Security Hardening**: N2HE privacy compliance, trust chain integrity
- **Reliability Testing**: Concurrency, failure injection, recovery

### Exclusions
- MOAI (Modular Orchestration AI) - out of scope
- Labs/experimental features - gated behind `TG_ENABLE_LABS`

---

## 2. Test Environment Specification

### 2.1 Environment Variables (Deterministic Mode)

```bash
export TG_DETERMINISTIC=true      # Reproducible hashes/timestamps
export TG_DEMO_MODE=true          # Auto-initialize database
export TG_SIMULATION=false        # Use real connectors (mocked data)
export TG_ENABLE_LABS=false       # Disable experimental features
export TG_ENVIRONMENT=development # Development mode
export DATABASE_URL=sqlite:///:memory:  # Ephemeral test database
```

### 2.2 Required Dependencies

```bash
pip install -e ".[dev,bench]"
# Optional for PQC tests:
pip install liboqs-python
```

### 2.3 System Health Verification

Before any QA run:
1. `GET /health` - Database connectivity
2. `GET /ready` - Service readiness
3. `GET /live` - Process liveness

---

## 3. Test Categories

### 3.1 Unit Tests (`tests/unit/`)
- Cryptographic primitives
- Model parsing and validation
- Utility functions
- Metric computations

### 3.2 Integration Tests (`tests/integration/`)
- API endpoint contracts
- Database persistence
- Service interactions

### 3.3 Regression Tests (`tests/regression/`) [NEW]
- Continuous learning invariants (I1-I8)
- State machine transitions
- Data integrity assertions

### 3.4 E2E Tests (`tests/e2e/`)
- Full workflow execution
- Multi-service coordination

### 3.5 Security Tests (`tests/security/`)
- Production gates
- Trust verification
- Privacy compliance

### 3.6 Performance Tests (`benchmarks/`)
- Latency profiling
- Throughput measurement
- Resource utilization

---

## 4. Critical Invariants (I1-I8)

| ID | Invariant | Severity | Test Coverage |
|----|-----------|----------|---------------|
| I1 | Route lifecycle complete | P0 | `test_route_happy_path.py` |
| I2 | Promotion gating enforced | P0 | `test_gates_block_promotion.py` |
| I3 | Rollback instant + correct | P0 | `test_rollback_correctness.py` |
| I4 | Evidence chain integrity | P0 | `test_evidence_and_tgsp_generated.py` |
| I5 | Determinism (stable hashes) | P1 | `test_determinism_stable_hashes.py` |
| I6 | Export specs valid | P1 | `test_export_specs_valid.py` |
| I7 | N2HE privacy receipts | P1 | `test_n2he_privacy_receipts.py` |
| I8 | Dashboard bundle schema | P2 | `test_dashboard_bundle_schema.py` |

---

## 5. QA Phases

### Phase 0: Baseline + Environment Hygiene
- [ ] Verify virtualenv isolation
- [ ] Pin dependencies
- [ ] Set deterministic env vars
- [ ] Verify health endpoints

### Phase 1: Static Analysis
- [ ] Lint (`ruff check src/`)
- [ ] Type check (`mypy src/`)
- [ ] Security scan (`bandit -r src/`)
- [ ] Dependency audit (`pip-audit`)

### Phase 2: Regression Matrix Build
- [ ] Document all invariants
- [ ] Map to test cases
- [ ] Identify coverage gaps

### Phase 3: Regression Suite Implementation
- [ ] API regression tests
- [ ] Property-based tests (Hypothesis)
- [ ] Concurrency tests
- [ ] Failure injection tests

### Phase 4: UI Smoke Tests
- [ ] Dashboard loads
- [ ] Route CRUD via UI
- [ ] Timeline visualization
- [ ] Export functionality

### Phase 5: Performance Deep Dive
- [ ] Step-level timing
- [ ] Memory profiling
- [ ] DB query analysis
- [ ] Latency budgets

### Phase 6: Bug Backlog + Fix Loop
- [ ] Collect issues into JSON
- [ ] Prioritize (P0/P1/P2)
- [ ] Fix with regression tests
- [ ] Update changelog

### Phase 7: Automation Scripts
- [ ] `make qa` target
- [ ] `make perf` target
- [ ] CI/CD integration

### Phase 8: Reporting
- [ ] QA report generation
- [ ] Performance profile
- [ ] Issues export

---

## 6. Artifact Structure

```
reports/qa/<run_id>/
  ├── system_info.json       # Environment snapshot
  ├── static_checks.md       # Lint/type/security findings
  ├── code_health.md         # Dead code, circular deps
  ├── config_risks.md        # Hardcoded secrets scan
  ├── qa_report.md           # Summary report
  ├── perf_profile.md        # Performance analysis
  ├── perf_baseline.json     # Raw performance data
  ├── issues.json            # Structured bug backlog
  ├── logs/                  # Test execution logs
  ├── traces/                # Distributed traces
  ├── profiles/              # CPU/memory profiles
  └── ui_traces/             # Playwright traces
```

---

## 7. Success Criteria

### Mandatory (Must Pass)
- All invariants I1-I8 have passing tests
- Zero P0 issues (or documented mitigation)
- `make qa` completes without errors
- N2HE safe logging compliance verified

### Target Metrics
- Unit test coverage: >80%
- Regression test coverage: 100% of invariants
- API response time p95: <500ms (cold), <200ms (warm)
- Dashboard bundle latency: <500ms

---

## 8. Execution Commands

### Full QA Run
```bash
make qa
# or
./scripts/qa/run_full_qa.sh
```

### Performance Baseline
```bash
make perf
# or
./scripts/qa/run_perf_baseline.sh
```

### Individual Test Categories
```bash
pytest tests/regression/ -v
pytest tests/security/ -v
pytest tests/integration/ -v
```

---

## 9. Escalation Procedures

### P0 Issues (Data Integrity, Security)
1. Immediately halt deployment
2. Create regression test
3. Fix within 24 hours
4. Post-mortem required

### P1 Issues (Performance, UX)
1. Track in issues.json
2. Schedule for next sprint
3. Add regression test

### P2 Issues (Polish, Minor)
1. Log for backlog review
2. Address opportunistically

---

## 10. Appendix

### A. Test Markers
```python
@pytest.mark.regression  # Regression test
@pytest.mark.perf        # Performance test
@pytest.mark.n2he        # N2HE privacy test
@pytest.mark.slow        # Long-running test
@pytest.mark.crypto      # Requires cryptographic dependencies
```

### B. Environment Setup Script
```bash
#!/bin/bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,bench]"
export TG_DETERMINISTIC=true
export TG_DEMO_MODE=true
```
