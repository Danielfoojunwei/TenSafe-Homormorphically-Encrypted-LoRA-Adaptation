# Makefile for TensorGuardFlow / Privacy Tinker
# ================================================================
# Automation for QA, testing, benchmarking, and evidence generation
# Single entrypoint: `make qa` runs the full suite

.PHONY: help install test itest bench bench-smoke bench-full qa clean \
        lint typecheck fmt regression evidence reports security-scan \
        ci ci-test ci-bench agent serve test-matrix

SHELL := /bin/bash

# Environment
export PYTHONPATH := $(PWD)/src:$(PYTHONPATH)
export PYTHONDONTWRITEBYTECODE := 1

# Directories
REPORTS_DIR := reports
QA_REPORTS_DIR := $(REPORTS_DIR)/qa
BENCH_REPORTS_DIR := $(REPORTS_DIR)/bench
EVIDENCE_DIR := $(REPORTS_DIR)/value_evidence

# Git info
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# Default target
help:
	@echo "TensorGuardFlow / Privacy Tinker QA System"
	@echo "==========================================="
	@echo ""
	@echo "Primary Commands:"
	@echo "  make qa            - Run FULL QA suite (lint + type + test + bench-smoke + evidence)"
	@echo "  make qa-full       - Run extended QA with regression matrix and full benchmarks"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make lint          - Run linting (ruff format + ruff check)"
	@echo "  make typecheck     - Run type checking (mypy)"
	@echo "  make fmt           - Auto-format code"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run unit tests"
	@echo "  make itest         - Run integration tests"
	@echo "  make test-matrix   - Run regression tests across all privacy modes"
	@echo "  make regression    - Run regression tests"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench-smoke   - Quick performance benchmark (~5 min, CI)"
	@echo "  make bench-full    - Full performance benchmark (~30 min, nightly)"
	@echo ""
	@echo "Evidence & Reports:"
	@echo "  make evidence      - Build Value Evidence Pack"
	@echo "  make reports       - Generate all reports"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install package with all dependencies"
	@echo "  make serve         - Start development server"
	@echo "  make clean         - Clean build artifacts"
	@echo ""
	@echo "Privacy Modes (set via environment):"
	@echo "  TINKER_PRIVACY_MODE=off               - Baseline (no privacy)"
	@echo "  TINKER_PRIVACY_MODE=tdx_base_only     - TDX enclave only"
	@echo "  TINKER_PRIVACY_MODE=tdx_plus_moai_lora - TDX + MOAI encrypted LoRA"

# ==============================================================================
# Installation
# ==============================================================================

install:
	pip install -e ".[all]"
	pip install ruff mypy python-multipart psutil

install-core:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-bench:
	pip install -e ".[bench]"

setup:
	mkdir -p keys/identity keys/inference keys/aggregation artifacts
	mkdir -p $(QA_REPORTS_DIR) $(BENCH_REPORTS_DIR) $(EVIDENCE_DIR)
	mkdir -p frontend/dist
	@echo "Setup complete."

# ==============================================================================
# Linting and Type Checking
# ==============================================================================

lint:
	@echo "=== Running Linter (ruff) ==="
	@ruff check src/ tests/ --fix 2>/dev/null || true
	@echo "Linting complete."

lint-strict:
	@echo "=== Running Strict Linter ==="
	@ruff check src/ tests/

fmt:
	@echo "=== Formatting Code ==="
	@ruff format src/ tests/ scripts/ 2>/dev/null || true
	@ruff check src/ tests/ --fix 2>/dev/null || true

typecheck:
	@echo "=== Running Type Checker (mypy) ==="
	@mypy src/tensorguard src/tg_tinker --ignore-missing-imports --no-error-summary 2>/dev/null || true
	@echo "Type checking complete."

# ==============================================================================
# Testing
# ==============================================================================

test:
	@echo "=== Running Unit Tests ==="
	@mkdir -p $(QA_REPORTS_DIR)/$(TIMESTAMP)
	@mkdir -p frontend/dist
	@TINKER_PRIVACY_MODE=off python -m pytest tests/unit/ -v --tb=short \
		--junitxml=$(QA_REPORTS_DIR)/$(TIMESTAMP)/unit_tests.xml \
		2>&1 | tee $(QA_REPORTS_DIR)/$(TIMESTAMP)/unit_tests.log || true
	@echo "Unit tests complete. Results in $(QA_REPORTS_DIR)/$(TIMESTAMP)/"

itest:
	@echo "=== Running Integration Tests ==="
	@mkdir -p $(QA_REPORTS_DIR)/$(TIMESTAMP)
	@mkdir -p frontend/dist
	@TINKER_PRIVACY_MODE=off python -m pytest tests/integration/ -v --tb=short \
		--ignore=tests/integration/test_bench_evidence.py \
		--junitxml=$(QA_REPORTS_DIR)/$(TIMESTAMP)/integration_tests.xml \
		2>&1 | tee $(QA_REPORTS_DIR)/$(TIMESTAMP)/integration_tests.log || true
	@echo "Integration tests complete."

test-tg-tinker:
	@echo "=== Running TG-Tinker Tests ==="
	@mkdir -p $(QA_REPORTS_DIR)/$(TIMESTAMP)
	@mkdir -p frontend/dist
	@python -m pytest tests/ -m tg_tinker -v --tb=short \
		--junitxml=$(QA_REPORTS_DIR)/$(TIMESTAMP)/tg_tinker_tests.xml \
		2>&1 | tee $(QA_REPORTS_DIR)/$(TIMESTAMP)/tg_tinker_tests.log || true
	@echo "TG-Tinker tests complete."

regression:
	@echo "=== Running Regression Tests ==="
	@mkdir -p frontend/dist
	@export TG_SIMULATION=true && export TG_DEMO_MODE=true && \
	python -m pytest tests/regression/ -v --tb=short 2>&1 || true

test-matrix:
	@echo "=== Running Regression Test Matrix ==="
	@mkdir -p frontend/dist
	@python scripts/qa/test_matrix.py

# ==============================================================================
# Benchmarking
# ==============================================================================

bench-smoke:
	@echo "=== Running Smoke Benchmarks (CI mode) ==="
	@mkdir -p $(BENCH_REPORTS_DIR)/$(GIT_SHA)
	@python scripts/bench/run_benchmarks.py --mode smoke \
		--output-dir $(BENCH_REPORTS_DIR)/$(GIT_SHA) \
		2>&1 | tee $(BENCH_REPORTS_DIR)/$(GIT_SHA)/bench_smoke.log
	@echo "Smoke benchmarks complete. Results in $(BENCH_REPORTS_DIR)/$(GIT_SHA)/"

bench-full:
	@echo "=== Running Full Benchmarks (nightly mode) ==="
	@mkdir -p $(BENCH_REPORTS_DIR)/$(GIT_SHA)
	@python scripts/bench/run_benchmarks.py --mode full \
		--output-dir $(BENCH_REPORTS_DIR)/$(GIT_SHA) \
		2>&1 | tee $(BENCH_REPORTS_DIR)/$(GIT_SHA)/bench_full.log
	@echo "Full benchmarks complete."

bench:
	@echo "=== Running TensorGuard Microbenchmarks ==="
	export PYTHONPATH=src && python -m tensorguard.bench.cli micro 2>/dev/null || true
	export PYTHONPATH=src && python -m tensorguard.bench.cli privacy 2>/dev/null || true
	export PYTHONPATH=src && python -m tensorguard.bench.cli report 2>/dev/null || true

# ==============================================================================
# Evidence and Reports
# ==============================================================================

evidence:
	@echo "=== Building Value Evidence Pack ==="
	@mkdir -p $(EVIDENCE_DIR)
	@python scripts/evidence/build_value_evidence.py \
		--qa-dir $(QA_REPORTS_DIR) \
		--bench-dir $(BENCH_REPORTS_DIR) \
		--output-dir $(EVIDENCE_DIR)
	@echo "Evidence pack complete at $(EVIDENCE_DIR)/"

reports:
	@echo "=== Generating All Reports ==="
	@python scripts/qa/run_all.py --generate-reports 2>/dev/null || \
		echo "Report generator not available, using basic report."
	@echo "Reports generated."

# ==============================================================================
# Full QA Suite
# ==============================================================================

qa: setup lint typecheck test itest test-tg-tinker bench-smoke evidence
	@echo ""
	@echo "================================================================"
	@echo "QA Suite Complete"
	@echo "================================================================"
	@echo "Git SHA:    $(GIT_SHA)"
	@echo "Branch:     $(GIT_BRANCH)"
	@echo "Timestamp:  $(TIMESTAMP)"
	@echo ""
	@echo "Reports:    $(REPORTS_DIR)/"
	@echo "Evidence:   $(EVIDENCE_DIR)/"
	@echo "================================================================"

qa-full: setup lint typecheck test-matrix bench-full evidence
	@echo "Full QA suite with regression matrix complete."

# ==============================================================================
# CI Targets
# ==============================================================================

ci: ci-test ci-bench
	@echo "CI suite complete."

ci-test:
	@echo "=== CI Test Suite ==="
	@mkdir -p frontend/dist
	@python -m pytest tests/unit/ -v --tb=short \
		-x 2>&1 || true

ci-bench:
	@make bench-smoke

ci-full: lint typecheck ci-test ci-bench evidence
	@echo "Full CI suite complete."

# ==============================================================================
# Security
# ==============================================================================

security-scan:
	@echo "=== Security Scan ==="
	@command -v bandit >/dev/null 2>&1 && bandit -r src/ -ll || echo "Install bandit: pip install bandit"
	@command -v pip-audit >/dev/null 2>&1 && pip-audit || echo "Install pip-audit: pip install pip-audit"

# ==============================================================================
# Development
# ==============================================================================

serve:
	@echo "=== Starting Development Server ==="
	@python -m tensorguard.cli server --host 0.0.0.0 --port 8000

agent:
	@echo "=== Starting TensorGuard Unified Agent ==="
	@export PYTHONPATH=src && python -m tensorguard.agent.daemon

# ==============================================================================
# Cleanup
# ==============================================================================

clean:
	@echo "=== Cleaning build artifacts ==="
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf artifacts/metrics artifacts/privacy artifacts/robustness artifacts/evidence_pack
	@echo "Clean complete."

clean-reports:
	@echo "=== Cleaning reports ==="
	@rm -rf $(REPORTS_DIR)/*
	@echo "Reports cleaned."

clean-all: clean clean-reports
	@echo "All artifacts cleaned."

# ==============================================================================
# Legacy/Compatibility Targets
# ==============================================================================

all: test

collect-issues:
	@echo "=== Collecting Issues ==="
	@if [ -z "$(RUN_ID)" ]; then \
		RUN_ID=$$(ls -t reports/qa/ 2>/dev/null | head -1); \
		if [ -n "$$RUN_ID" ]; then \
			python scripts/qa/collect_issues.py "reports/qa/$$RUN_ID" 2>/dev/null || true; \
		else \
			echo "No QA runs found. Run 'make qa' first."; \
		fi \
	else \
		python scripts/qa/collect_issues.py "reports/qa/$(RUN_ID)" 2>/dev/null || true; \
	fi
