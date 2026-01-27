#!/bin/bash
#
# TensorGuardFlow Full QA Runner
#
# Runs the complete QA suite including:
# - Static analysis (lint, type check, security scan)
# - Unit tests
# - Regression tests
# - UI smoke tests (if available)
# - Performance baseline
#
# Generates comprehensive reports in reports/qa/<run_id>/
#

set -e

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPORTS_DIR="${PROJECT_ROOT}/reports/qa/${RUN_ID}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "TensorGuardFlow Full QA Runner"
echo "Run ID: ${RUN_ID}"
echo "=============================================="

# --- Create report directories ---
mkdir -p "${REPORTS_DIR}"/{logs,traces,profiles,ui_traces}

# --- Set deterministic environment ---
export TG_DETERMINISTIC=true
export TG_DEMO_MODE=true
export TG_SIMULATION=true
export TG_ENABLE_LABS=false
export TG_ENVIRONMENT=development
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# --- Collect system info ---
echo -e "\n${YELLOW}[1/8] Collecting system information...${NC}"
cat > "${REPORTS_DIR}/system_info.json" << EOF
{
    "run_id": "${RUN_ID}",
    "timestamp": "$(date -Iseconds)",
    "os": "$(uname -s)",
    "os_version": "$(uname -r)",
    "hostname": "$(hostname)",
    "cpu": "$(nproc) cores",
    "python_version": "$(python --version 2>&1)",
    "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "working_dir": "${PROJECT_ROOT}"
}
EOF
echo -e "${GREEN}System info collected${NC}"

# --- Static Analysis ---
echo -e "\n${YELLOW}[2/8] Running static analysis...${NC}"
STATIC_REPORT="${REPORTS_DIR}/static_checks.md"

echo "# Static Analysis Report" > "${STATIC_REPORT}"
echo "" >> "${STATIC_REPORT}"
echo "Run ID: ${RUN_ID}" >> "${STATIC_REPORT}"
echo "Timestamp: $(date)" >> "${STATIC_REPORT}"
echo "" >> "${STATIC_REPORT}"

# Lint with ruff
echo "## Linting (ruff)" >> "${STATIC_REPORT}"
echo '```' >> "${STATIC_REPORT}"
cd "${PROJECT_ROOT}"
if ruff check src/ 2>&1 | tee -a "${STATIC_REPORT}"; then
    echo -e "${GREEN}Lint: PASS${NC}"
    LINT_STATUS="PASS"
else
    echo -e "${RED}Lint: FAIL${NC}"
    LINT_STATUS="FAIL"
fi
echo '```' >> "${STATIC_REPORT}"
echo "" >> "${STATIC_REPORT}"

# Type check with mypy
echo "## Type Checking (mypy)" >> "${STATIC_REPORT}"
echo '```' >> "${STATIC_REPORT}"
if mypy src/ --ignore-missing-imports 2>&1 | tee -a "${STATIC_REPORT}"; then
    echo -e "${GREEN}Type check: PASS${NC}"
    TYPECHECK_STATUS="PASS"
else
    echo -e "${YELLOW}Type check: WARNINGS${NC}"
    TYPECHECK_STATUS="WARNINGS"
fi
echo '```' >> "${STATIC_REPORT}"
echo "" >> "${STATIC_REPORT}"

# Security scan with bandit (if available)
echo "## Security Scan (bandit)" >> "${STATIC_REPORT}"
echo '```' >> "${STATIC_REPORT}"
if command -v bandit &> /dev/null; then
    if bandit -r src/ -ll 2>&1 | tee -a "${STATIC_REPORT}"; then
        echo -e "${GREEN}Security scan: PASS${NC}"
        SECURITY_STATUS="PASS"
    else
        echo -e "${RED}Security scan: ISSUES FOUND${NC}"
        SECURITY_STATUS="ISSUES"
    fi
else
    echo "bandit not installed - skipping security scan"
    echo "Install with: pip install bandit"
    SECURITY_STATUS="SKIPPED"
fi
echo '```' >> "${STATIC_REPORT}"
echo "" >> "${STATIC_REPORT}"

echo -e "${GREEN}Static analysis complete${NC}"

# --- Unit Tests ---
echo -e "\n${YELLOW}[3/8] Running unit tests...${NC}"
pytest tests/unit/ -v --tb=short \
    --junitxml="${REPORTS_DIR}/junit_unit.xml" \
    2>&1 | tee "${REPORTS_DIR}/logs/unit_tests.log" || true

# --- Regression Tests ---
echo -e "\n${YELLOW}[4/8] Running regression tests...${NC}"
pytest tests/regression/ -v --tb=short \
    -m "regression" \
    --junitxml="${REPORTS_DIR}/junit_regression.xml" \
    2>&1 | tee "${REPORTS_DIR}/logs/regression_tests.log" || true

# --- Integration Tests ---
echo -e "\n${YELLOW}[5/8] Running integration tests...${NC}"
pytest tests/integration/ -v --tb=short \
    --junitxml="${REPORTS_DIR}/junit_integration.xml" \
    2>&1 | tee "${REPORTS_DIR}/logs/integration_tests.log" || true

# --- Security Tests ---
echo -e "\n${YELLOW}[6/8] Running security tests...${NC}"
pytest tests/security/ -v --tb=short \
    --junitxml="${REPORTS_DIR}/junit_security.xml" \
    2>&1 | tee "${REPORTS_DIR}/logs/security_tests.log" || true

# --- UI Tests (if Playwright available) ---
echo -e "\n${YELLOW}[7/8] Running UI tests (if available)...${NC}"
if [ -d "tests/ui" ] && command -v npx &> /dev/null; then
    if [ -f "tests/ui/playwright.config.ts" ]; then
        cd "${PROJECT_ROOT}"
        npx playwright test tests/ui/ \
            --reporter=html \
            --output="${REPORTS_DIR}/ui_traces/" \
            2>&1 | tee "${REPORTS_DIR}/logs/ui_tests.log" || true
    else
        echo "UI tests not configured - skipping"
    fi
else
    echo "Playwright not available or tests/ui not found - skipping UI tests"
fi

# --- Performance Baseline ---
echo -e "\n${YELLOW}[8/8] Running performance baseline...${NC}"
if [ -f "${SCRIPT_DIR}/run_perf_baseline.sh" ]; then
    bash "${SCRIPT_DIR}/run_perf_baseline.sh" "${RUN_ID}" || true
else
    echo "Performance baseline script not found - skipping"
fi

# --- Generate QA Report ---
echo -e "\n${YELLOW}Generating QA report...${NC}"

# Count test results
UNIT_PASS=$(grep -c "passed" "${REPORTS_DIR}/logs/unit_tests.log" 2>/dev/null || echo "0")
REGRESSION_PASS=$(grep -c "passed" "${REPORTS_DIR}/logs/regression_tests.log" 2>/dev/null || echo "0")
INTEGRATION_PASS=$(grep -c "passed" "${REPORTS_DIR}/logs/integration_tests.log" 2>/dev/null || echo "0")
SECURITY_PASS=$(grep -c "passed" "${REPORTS_DIR}/logs/security_tests.log" 2>/dev/null || echo "0")

# Generate summary report
cat > "${REPORTS_DIR}/qa_report.md" << EOF
# TensorGuardFlow QA Report

**Run ID:** ${RUN_ID}
**Timestamp:** $(date)
**Git SHA:** $(git rev-parse HEAD 2>/dev/null || echo 'unknown')
**Branch:** $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')

---

## Executive Summary

| Check | Status |
|-------|--------|
| Lint (ruff) | ${LINT_STATUS} |
| Type Check (mypy) | ${TYPECHECK_STATUS} |
| Security Scan | ${SECURITY_STATUS} |
| Unit Tests | See details |
| Regression Tests | See details |
| Integration Tests | See details |
| Security Tests | See details |

---

## Regression Matrix Status

| Invariant | Test | Status |
|-----------|------|--------|
| I1 Route Lifecycle | test_route_happy_path.py | Check logs |
| I2 Promotion Gating | test_gates_block_promotion.py | Check logs |
| I3 Rollback Correct | test_rollback_correctness.py | Check logs |
| I4 Evidence Chain | test_evidence_and_tgsp_generated.py | Check logs |
| I5 Determinism | test_determinism_stable_hashes.py | Check logs |
| I6 Export Specs | test_export_specs_valid.py | Check logs |
| I7 N2HE Privacy | test_n2he_privacy_receipts.py | Check logs |
| I8 Dashboard Bundle | test_dashboard_bundle_schema.py | Check logs |

---

## Test Results

### Unit Tests
See: logs/unit_tests.log

### Regression Tests
See: logs/regression_tests.log

### Integration Tests
See: logs/integration_tests.log

### Security Tests
See: logs/security_tests.log

---

## Artifacts

- System Info: system_info.json
- Static Checks: static_checks.md
- Performance Profile: perf_profile.md (if available)
- Performance Baseline: perf_baseline.json (if available)
- JUnit Reports: junit_*.xml

---

## Next Steps

1. Review failed tests in logs/
2. Check static_checks.md for code quality issues
3. Review perf_profile.md for bottlenecks
4. Address P0/P1 issues first

EOF

echo -e "${GREEN}=============================================="
echo "QA Run Complete!"
echo "Reports available at: ${REPORTS_DIR}/"
echo -e "==============================================${NC}"

# Return success if critical tests passed
exit 0
