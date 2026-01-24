#!/bin/bash
# TensorGuardFlow Integration Verification Script

echo "================================================================================"
echo "TensorGuardFlow Integration Framework - Compatibility Scorecard"
echo "================================================================================"

# 1. Environment Setup
export PYTHONPATH="src"
export TG_SIMULATION="true"

# Scorecard variables
CORE_PASS=0
OPTIONAL_PASS=0
TOTAL_FAILED=0

run_test() {
    local name=$1
    local cmd=$2
    local required=$3
    
    echo -n "Running $name... "
    if $cmd > /dev/null 2>&1; then
        echo -e "\e[32m[PASS]\e[0m"
        if [ "$required" = "true" ]; then CORE_PASS=$((CORE_PASS+1)); else OPTIONAL_PASS=$((OPTIONAL_PASS+1)); fi
    else
        echo -e "\e[31m[FAIL]\e[0m"
        if [ "$required" = "true" ]; then TOTAL_FAILED=$((TOTAL_FAILED+1)); fi
    fi
}

# 2. RUN TESTS
echo "--- Layer 1: Contract & Schema Tests ---"
run_test "Config Schema Validation" "python tests/integration/pipeline/test_config_schemas.py" "true"
run_test "Connector Protocols" "python tests/integration/pipeline/test_framework_contracts.py" "true"

echo -e "\n--- Layer 2: Local E2E Pipeline ---"
run_test "Local RunOnce End-to-End" "python tests/integration/pipeline/test_route_run_once_local_end_to_end.py" "true"

echo -e "\n--- Layer 3: Cloud Smoke Tests (Optional) ---"
# These would check for env vars like AWS_ACCESS_KEY_ID etc.
if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "S3 Feed / SageMaker: [SKIP] (Missing AWS Credentials)"
else
    # run_test "S3 Smoke Test" "..." "false"
    echo "Smoke tests would run here if credentials were present."
fi

echo -e "\n================================================================================"
echo "FINAL SCORECARD"
echo "================================================================================"
echo "Core Integrations Passed: $CORE_PASS / 3"
echo "Optional Integrations Passed: $OPTIONAL_PASS"
echo "Blocking Failures: $TOTAL_FAILED"

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "\e[32m>>> SYSTEM STATUS: GREEN <<<\e[0m"
    exit 0
else
    echo -e "\e[31m>>> SYSTEM STATUS: RED <<<\e[0m"
    exit 1
fi
