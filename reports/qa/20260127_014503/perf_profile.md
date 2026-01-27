# Performance Profile Report

**Run ID:** 20260127_014503
**Date:** 2026-01-27

---

## Overview

Performance baseline partially executed. Full workflow tests blocked by infrastructure issues.

## Tested Endpoints

### Health Endpoints
| Endpoint | Avg Latency | Status |
|----------|-------------|--------|
| GET /health | <10ms | PASS |
| GET /ready | <10ms | PASS |
| GET /live | <5ms | PASS |

### API Endpoints
| Endpoint | Avg Latency | Budget | Status |
|----------|-------------|--------|--------|
| POST /tgflow/routes | <50ms | 100ms | PASS |
| GET /tgflow/routes | <30ms | 100ms | PASS |
| POST /tgflow/resolve (N2HE) | <20ms | 50ms | PASS |

### Dashboard Endpoints
| Endpoint | Cold | Warm | Budget | Status |
|----------|------|------|--------|--------|
| GET /metrics/routes/{key}/dashboard_bundle | TBD | TBD | <500ms/<200ms | UNTESTED |

---

## Bottleneck Analysis

### Identified Concerns
1. **N+1 Query Risk**: Timeline endpoint may have N+1 pattern when loading events
2. **Index Recommendations**:
   - `tenant_id + route_key + ts` for time series queries
   - `adapter_id` for adapter lookups
   - `run_id` for run step metrics

### Memory Profile
- Base memory usage: ~100MB
- Per-route overhead: ~5MB
- Peak during run_once: TBD (requires full workflow)

---

## Recommendations

1. Add database query logging in test mode to detect N+1
2. Implement query result caching for dashboard bundle
3. Consider pagination for timeline endpoint
4. Profile full run_once workflow once data issues resolved

---

## Next Steps

1. Fix P1 issues blocking full workflow
2. Re-run performance baseline with real workload
3. Establish baseline thresholds for CI/CD gates
