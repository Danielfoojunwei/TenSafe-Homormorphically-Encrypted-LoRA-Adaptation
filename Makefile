# TenSafe Makefile
# Privacy-First ML Training API
# ================================================================

.PHONY: help install dev test lint typecheck serve clean qa bench evidence compliance compliance-smoke bench-llama3 bench-llama3-smoke build-n2he test-n2he bench-n2he bench-n2he-full n2he-smoke

SHELL := /bin/bash
PYTHON := python3
PIP := $(PYTHON) -m pip

# Environment
export PYTHONPATH := $(PWD)/src:$(PYTHONPATH)
export PYTHONDONTWRITEBYTECODE := 1

# Directories
REPORTS_DIR := reports

# Default target
help:
	@echo "TenSafe - Privacy-First ML Training API"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make dev           Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make serve         Start the API server (port 8000)"
	@echo "  make test          Run all tests"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make typecheck     Run type checker (mypy)"
	@echo ""
	@echo "QA:"
	@echo "  make qa            Run full QA suite (lint + typecheck + test)"
	@echo "  make bench         Run benchmarks (smoke mode)"
	@echo "  make bench-full    Run full benchmarks"
	@echo "  make evidence      Generate value evidence pack"
	@echo "  make test-matrix   Run tests across privacy modes"
	@echo ""
	@echo "Compliance (ISO 27701, ISO 27001, SOC 2):"
	@echo "  make compliance-smoke  Run quick compliance checks"
	@echo "  make compliance        Generate full compliance evidence pack"
	@echo ""
	@echo "Llama3 LoRA Benchmarks:"
	@echo "  make bench-llama3-smoke  Run smoke benchmark with compliance"
	@echo "  make bench-llama3        Run full benchmark with compliance"
	@echo ""
	@echo "N2HE Homomorphic Encryption:"
	@echo "  make build-n2he          Build N2HE native library"
	@echo "  make test-n2he           Run N2HE integration tests"
	@echo "  make bench-n2he          Run N2HE quick benchmark"
	@echo "  make bench-n2he-full     Run N2HE full benchmark"
	@echo "  make n2he-smoke          Run N2HE smoke test (tests + benchmark)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove build artifacts"

# =============================================================================
# Setup
# =============================================================================

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev,bench]"

setup: dev
	@echo "Development environment ready"

# =============================================================================
# Development
# =============================================================================

serve:
	$(PYTHON) -m uvicorn tensorguard.platform.main:app --reload --host 0.0.0.0 --port 8000

# =============================================================================
# Testing
# =============================================================================

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

test-regression:
	$(PYTHON) -m pytest tests/regression/ -v --tb=short -m regression

test-tensafe:
	$(PYTHON) -m pytest tests/ -v --tb=short -m tensafe

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# =============================================================================
# Linting & Type Checking
# =============================================================================

lint:
	$(PYTHON) -m ruff check src/ tests/ || true

lint-fix:
	$(PYTHON) -m ruff check --fix src/ tests/

fmt:
	$(PYTHON) -m ruff format src/ tests/

typecheck:
	$(PYTHON) -m mypy src/tensorguard src/tg_tinker --ignore-missing-imports || true

# =============================================================================
# QA Suite
# =============================================================================

qa: lint typecheck test
	@echo ""
	@echo "=========================================="
	@echo "QA Suite Complete"
	@echo "=========================================="

qa-full: qa test-matrix bench evidence
	@echo ""
	@echo "=========================================="
	@echo "Full QA Suite Complete"
	@echo "=========================================="

# =============================================================================
# E2E Tests
# =============================================================================

test-e2e:
	$(PYTHON) -m pytest tests/e2e/ -v --tb=short -m "not slow"

test-e2e-full:
	FULL_E2E=1 $(PYTHON) -m pytest tests/e2e/ -v --tb=short -s

# =============================================================================
# Benchmarking
# =============================================================================

bench:
	@mkdir -p $(REPORTS_DIR)/bench
	$(PYTHON) scripts/bench/run_benchmarks.py --mode smoke

bench-full:
	@mkdir -p $(REPORTS_DIR)/bench
	$(PYTHON) scripts/bench/run_benchmarks.py --mode full

bench-comparison:
	@mkdir -p $(REPORTS_DIR)/bench
	$(PYTHON) scripts/bench/comparison/tensafe_vs_baseline.py --mode smoke

bench-comparison-full:
	@mkdir -p $(REPORTS_DIR)/bench
	$(PYTHON) scripts/bench/comparison/tensafe_vs_baseline.py --mode full

# =============================================================================
# Evidence & Test Matrix
# =============================================================================

evidence:
	@mkdir -p $(REPORTS_DIR)/value_evidence
	$(PYTHON) scripts/evidence/build_value_evidence.py

test-matrix:
	@mkdir -p $(REPORTS_DIR)/qa
	$(PYTHON) scripts/qa/test_matrix.py

# =============================================================================
# Compliance Evidence (ISO 27701, ISO 27001, SOC 2)
# =============================================================================

compliance-smoke:
	@echo "Running compliance smoke checks..."
	@mkdir -p $(REPORTS_DIR)/compliance
	$(PYTHON) scripts/compliance/collect_privacy_security_metrics.py --mode smoke
	$(PYTHON) scripts/compliance/build_compliance_evidence.py
	@echo "Compliance smoke check complete."

compliance:
	@echo "Running full compliance evidence collection..."
	@mkdir -p $(REPORTS_DIR)/compliance
	$(PYTHON) scripts/compliance/collect_privacy_security_metrics.py --mode full
	$(PYTHON) scripts/compliance/build_compliance_evidence.py
	@echo "Full compliance evidence pack generated."

# =============================================================================
# Llama3 LoRA Benchmark Suite (Training + Eval + Perf + Compliance)
# =============================================================================

bench-llama3-smoke:
	@echo "Running Llama3 LoRA smoke benchmark..."
	@mkdir -p $(REPORTS_DIR)/bench
	@mkdir -p $(REPORTS_DIR)/compliance
	$(PYTHON) scripts/bench/comparison/tensafe_vs_baseline.py --mode smoke
	$(PYTHON) scripts/compliance/collect_privacy_security_metrics.py --mode smoke
	$(PYTHON) scripts/compliance/build_compliance_evidence.py
	$(PYTHON) scripts/bench/build_llama3_report.py --mode smoke
	@echo "Llama3 smoke benchmark complete."

bench-llama3:
	@echo "Running full Llama3 LoRA benchmark with compliance..."
	@mkdir -p $(REPORTS_DIR)/bench
	@mkdir -p $(REPORTS_DIR)/compliance
	$(PYTHON) scripts/bench/comparison/tensafe_vs_baseline.py --mode full
	$(PYTHON) scripts/compliance/collect_privacy_security_metrics.py --mode full
	$(PYTHON) scripts/compliance/build_compliance_evidence.py
	$(PYTHON) scripts/bench/build_llama3_report.py --mode full
	@echo "Full Llama3 benchmark with compliance complete."

# =============================================================================
# N2HE Homomorphic Encryption
# =============================================================================

build-n2he:
	@echo "Building N2HE native library..."
	bash scripts/n2he/build_n2he.sh
	@echo "N2HE native library built."

test-n2he:
	@echo "Running N2HE integration tests..."
	$(PYTHON) -m pytest tests/n2he/ -v --tb=short
	@echo "N2HE tests complete."

bench-n2he:
	@echo "Running N2HE quick benchmark..."
	@mkdir -p $(REPORTS_DIR)/n2he
	$(PYTHON) -c "from tensorguard.n2he.benchmark import run_quick_benchmark, generate_benchmark_report; suite = run_quick_benchmark(); print(generate_benchmark_report(suite))"
	@echo "N2HE quick benchmark complete."

bench-n2he-full:
	@echo "Running N2HE full benchmark..."
	@mkdir -p $(REPORTS_DIR)/n2he
	$(PYTHON) -c "from tensorguard.n2he.benchmark import run_full_benchmark, generate_benchmark_report; suite = run_full_benchmark(); print(generate_benchmark_report(suite))"
	@echo "N2HE full benchmark complete."

n2he-smoke:
	@echo "Running N2HE smoke test (tests + quick benchmark)..."
	@mkdir -p $(REPORTS_DIR)/n2he
	$(PYTHON) -m pytest tests/n2he/ -v --tb=short -x
	$(PYTHON) -c "from tensorguard.n2he.benchmark import run_quick_benchmark, generate_benchmark_report; suite = run_quick_benchmark(); print(generate_benchmark_report(suite))"
	@echo "N2HE smoke test complete."

# =============================================================================
# Utilities
# =============================================================================

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf $(REPORTS_DIR)/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t tensafe:latest .

docker-run:
	docker run -p 8000:8000 -e TS_ENVIRONMENT=development tensafe:latest
