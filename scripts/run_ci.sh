#!/bin/bash
# TenSafe CI/Regression Script
#
# Runs the complete test suite including:
# - Unit tests
# - Integration tests
# - Smoke tests
# - Benchmarks
# - Linting (if available)
#
# Usage:
#   ./scripts/run_ci.sh          # Run all checks
#   ./scripts/run_ci.sh --quick  # Quick mode (skip slow tests)
#   ./scripts/run_ci.sh --tests  # Tests only
#   ./scripts/run_ci.sh --bench  # Benchmarks only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
TESTS_ONLY=false
BENCH_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --tests)
            TESTS_ONLY=true
            shift
            ;;
        --bench)
            BENCH_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "TenSafe CI/Regression Suite"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Quick mode: $QUICK_MODE"
echo "Date: $(date)"
echo "========================================"

# Track overall status
FAILED_STEPS=()

run_step() {
    local step_name="$1"
    local command="$2"

    echo ""
    echo -e "${YELLOW}>>> Running: $step_name${NC}"
    echo "Command: $command"
    echo "----------------------------------------"

    if eval "$command"; then
        echo -e "${GREEN}✓ $step_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $step_name failed${NC}"
        FAILED_STEPS+=("$step_name")
        return 1
    fi
}

# ==============================================================================
# Tests
# ==============================================================================

run_tests() {
    echo ""
    echo "========================================"
    echo "Running Tests"
    echo "========================================"

    # Baseline SFT tests
    run_step "Baseline SFT Tests" \
        "python -m pytest tests/test_baseline_sft.py -v --tb=short" || true

    # Loss registry tests
    run_step "Loss Registry Tests" \
        "python -m pytest tests/test_loss_registry.py tests/test_custom_loss_plug.py -v --tb=short" || true

    # RLVR tests
    run_step "RLVR Rollout Tests" \
        "python -m pytest tests/test_rlvr_rollout_shapes.py -v --tb=short" || true

    run_step "RLVR Reward Tests" \
        "python -m pytest tests/test_rlvr_reward_registry.py -v --tb=short" || true

    run_step "RLVR REINFORCE Tests" \
        "python -m pytest tests/test_rlvr_reinforce_improves_reward.py -v --tb=short" || true

    run_step "RLVR LoRA-Only Tests" \
        "python -m pytest tests/test_rlvr_only_lora_updates.py -v --tb=short" || true

    run_step "RLVR PPO Tests" \
        "python -m pytest tests/test_rlvr_ppo.py -v --tb=short" || true

    run_step "RLVR Checkpoint Tests" \
        "python -m pytest tests/test_rlvr_checkpoint.py -v --tb=short" || true

    if [ "$QUICK_MODE" = false ]; then
        # Run slow tests
        run_step "Extended Training Tests" \
            "python -m pytest tests/test_baseline_sft.py::TestIntegration -v --tb=short" || true
    fi
}

# ==============================================================================
# Smoke Tests
# ==============================================================================

run_smoke_tests() {
    echo ""
    echo "========================================"
    echo "Running Smoke Tests"
    echo "========================================"

    run_step "Baseline SFT Smoke Test" \
        "python scripts/baseline_sft_smoke.py" || true

    if [ -f "examples/rlvr_toy_task/run_toy_rlvr.py" ]; then
        run_step "RLVR Toy Example" \
            "python examples/rlvr_toy_task/run_toy_rlvr.py --steps 5" || true
    fi
}

# ==============================================================================
# Benchmarks
# ==============================================================================

run_benchmarks() {
    echo ""
    echo "========================================"
    echo "Running Benchmarks"
    echo "========================================"

    if [ "$QUICK_MODE" = true ]; then
        # Quick benchmark mode
        run_step "SFT Benchmarks" \
            "python scripts/benchmark_harness.py --suite sft --quiet" || true
    else
        # Full benchmark suite
        run_step "Full Benchmark Suite" \
            "python scripts/benchmark_harness.py --output outputs/benchmark_results.json" || true
    fi
}

# ==============================================================================
# Linting (optional)
# ==============================================================================

run_linting() {
    echo ""
    echo "========================================"
    echo "Running Linting (if available)"
    echo "========================================"

    # Check if ruff is available
    if command -v ruff &> /dev/null; then
        run_step "Ruff Linting" \
            "ruff check tensafe/ --ignore E501" || true
    else
        echo "ruff not available, skipping linting"
    fi

    # Check if mypy is available
    if command -v mypy &> /dev/null && [ "$QUICK_MODE" = false ]; then
        run_step "Type Checking" \
            "mypy tensafe/ --ignore-missing-imports --no-error-summary" || true
    else
        echo "mypy not available or quick mode, skipping type checking"
    fi
}

# ==============================================================================
# Golden Artifact Verification
# ==============================================================================

verify_golden_artifacts() {
    echo ""
    echo "========================================"
    echo "Verifying Golden Artifacts"
    echo "========================================"

    GOLDEN_DIR="$PROJECT_ROOT/tests/golden"

    if [ -d "$GOLDEN_DIR" ]; then
        run_step "Golden Artifacts Exist" \
            "test -f $GOLDEN_DIR/baseline_sft_metrics.json && test -f $GOLDEN_DIR/baseline_loss_curve.json" || true
    else
        echo "Golden artifacts directory not found"
    fi
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    START_TIME=$(date +%s)

    if [ "$TESTS_ONLY" = true ]; then
        run_tests
    elif [ "$BENCH_ONLY" = true ]; then
        run_benchmarks
    else
        # Run all steps
        run_tests
        run_smoke_tests

        if [ "$QUICK_MODE" = false ]; then
            run_benchmarks
        fi

        run_linting
        verify_golden_artifacts
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "========================================"
    echo "CI Summary"
    echo "========================================"
    echo "Duration: ${DURATION}s"

    if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
        echo -e "${GREEN}All steps passed!${NC}"
        exit 0
    else
        echo -e "${RED}Failed steps:${NC}"
        for step in "${FAILED_STEPS[@]}"; do
            echo "  - $step"
        done
        exit 1
    fi
}

main
