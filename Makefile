# Makefile for TensorGuardFlow
# Automation for build, test, and security verification

.PHONY: install test agent bench bench-full bench-quick bench-report bench-regression bench-ci clean reports lint setup ci typecheck

# Default target
all: test

# Installation (uses pyproject.toml for dependency management)
install:
	pip install -e ".[all]"

# Install core only (minimal dependencies)
install-core:
	pip install -e .

# Install with specific extras
install-dev:
	pip install -e ".[dev]"

install-bench:
	pip install -e ".[bench]"

# Project Setup
setup:
	mkdir -p keys/identity keys/inference keys/aggregation artifacts
	python scripts/setup_env.py

# Testing
test:
	@echo "--- Running Holistic Security Fabric Tests ---"
	python -m pytest tests/

# Agent Orchestration
agent:
	@echo "--- Starting TensorGuard Unified Agent ---"
	export PYTHONPATH=src && python -m tensorguard.agent.daemon

# Benchmarking Subsystem (Legacy Microbenchmarks)
bench:
	@echo "--- Running TensorGuard Microbenchmarks ---"
	export PYTHONPATH=src && python -m tensorguard.bench.cli micro
	@echo "--- Running Privacy Eval ---"
	export PYTHONPATH=src && python -m tensorguard.bench.cli privacy
	@echo "--- Generating Benchmarking Report ---"
	export PYTHONPATH=src && python -m tensorguard.bench.cli report

# Performance Benchmarking Suite (HTTP, Telemetry, Resources)
bench-full:
	@echo "=== TensorGuardFlow Full Performance Benchmark Suite ==="
	@echo "This requires a running TensorGuardFlow server at http://localhost:8000"
	@mkdir -p artifacts/benchmarks
	python -m benchmarks.runner --load moderate --duration 60 --output artifacts/benchmarks
	@echo "=== Generating Analysis Report ==="
	python -c "from benchmarks.analyzer import analyze_results; analyze_results('artifacts/benchmarks/benchmark_results_latest.json', 'docs')"

# Quick benchmark for development (light load, short duration)
bench-quick:
	@echo "=== Quick Performance Benchmark ==="
	@mkdir -p artifacts/benchmarks
	python -m benchmarks.runner --load light --duration 30 --output artifacts/benchmarks

# Generate analysis report from existing results
bench-report:
	@echo "=== Generating Benchmark Analysis Report ==="
	@test -f artifacts/benchmarks/benchmark_results_latest.json || (echo "Error: No benchmark results found. Run 'make bench-full' first." && exit 1)
	python -c "from benchmarks.analyzer import analyze_results; analyze_results('artifacts/benchmarks/benchmark_results_latest.json', 'docs')"
	@echo "Report generated at: docs/performance_benchmark_report.md"

# Run regression tests against thresholds
bench-regression:
	@echo "=== Performance Regression Test ==="
	@test -f artifacts/benchmarks/benchmark_results_latest.json || (echo "Error: No benchmark results found. Run 'make bench-full' first." && exit 1)
	python -m benchmarks.regression_test artifacts/benchmarks/benchmark_results_latest.json

# CI-friendly benchmark: quick test with regression check
bench-ci:
	@echo "=== CI Performance Benchmark ==="
	@mkdir -p artifacts/benchmarks
	python -m benchmarks.runner --load light --duration 30 --output artifacts/benchmarks
	python -m benchmarks.regression_test artifacts/benchmarks/benchmark_results_latest.json --junit artifacts/benchmarks/regression_junit.xml

# Linting
lint:
	@echo "--- Running Linter (ruff) ---"
	ruff check src/

# Type checking
typecheck:
	@echo "--- Running Type Checker (mypy) ---"
	mypy src/

# CI target: install, lint, (optional type check), tests
ci: install lint typecheck test
	@echo "--- CI checks completed ---"

# Cleanup
clean:
	@echo "--- Cleaning temporary files ---"
	rm -rf .pytest_cache
	rm -rf artifacts/metrics artifacts/privacy artifacts/robustness artifacts/evidence_pack
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f artifacts/report.html
