#!/bin/bash
#
# TensorGuardFlow Performance Baseline Runner
#
# Runs performance tests and generates baseline metrics.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
REPORTS_DIR="${PROJECT_ROOT}/reports/qa/${RUN_ID}"

mkdir -p "${REPORTS_DIR}/profiles"

echo "=============================================="
echo "TensorGuardFlow Performance Baseline"
echo "Run ID: ${RUN_ID}"
echo "=============================================="

# Set environment
export TG_DETERMINISTIC=true
export TG_DEMO_MODE=true
export TG_SIMULATION=true
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Run Python performance baseline
python "${PROJECT_ROOT}/benchmarks/perf_baseline/run_perf.py" \
    --output "${REPORTS_DIR}/perf_baseline.json" \
    2>&1 | tee "${REPORTS_DIR}/logs/perf_baseline.log" || true

# Generate performance profile report
cat > "${REPORTS_DIR}/perf_profile.md" << 'EOF'
# Performance Profile Report

## Overview

This report contains performance analysis from the QA baseline run.

## Methodology

- 3 sequential updates on 1 route
- Metrics captured per step: INGEST, NOVELTY, TRAIN, EVAL, PACKAGE, REGISTER, PROMOTE

## Results

See perf_baseline.json for raw data.

### Key Metrics

| Metric | Value | Budget | Status |
|--------|-------|--------|--------|
| Dashboard Bundle (cold) | TBD | <500ms | TBD |
| Dashboard Bundle (warm) | TBD | <200ms | TBD |
| Run Once (e2e) | TBD | <30s | TBD |
| Resolve Latency | TBD | <50ms | TBD |

## Recommendations

1. Monitor N+1 queries in timeline endpoints
2. Add indexes for tenant_id + route_key + ts queries
3. Consider caching for dashboard bundle

## Profiling Notes

For detailed profiling, run:
```bash
py-spy record -o profile.svg -- python -m pytest tests/regression/ -v
```

EOF

echo "Performance baseline complete"
echo "Results: ${REPORTS_DIR}/perf_baseline.json"
echo "Profile: ${REPORTS_DIR}/perf_profile.md"
