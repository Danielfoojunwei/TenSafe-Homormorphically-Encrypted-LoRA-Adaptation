# Makefile for TensorGuardFlow
# Automation for build, test, and security verification

.PHONY: install test agent bench bench-full bench-quick bench-report bench-regression bench-ci clean reports lint setup ci typecheck qa perf regression security-scan fmt

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

# === QA Engineering Suite ===

# Full QA run (static analysis + all tests + perf baseline)
qa:
	@echo "=== TensorGuardFlow Full QA Suite ==="
	./scripts/qa/run_full_qa.sh

# Performance baseline only
perf:
	@echo "=== Performance Baseline ==="
	./scripts/qa/run_perf_baseline.sh

# Regression tests only
regression:
	@echo "=== Regression Tests ==="
	export TG_SIMULATION=true && export TG_DEMO_MODE=true && \
	python -m pytest tests/regression/ -v -m "regression"

# Security scan (bandit + pip-audit)
security-scan:
	@echo "=== Security Scan ==="
	@command -v bandit >/dev/null 2>&1 && bandit -r src/ -ll || echo "Install bandit: pip install bandit"
	@command -v pip-audit >/dev/null 2>&1 && pip-audit || echo "Install pip-audit: pip install pip-audit"

# Format code
fmt:
	@echo "=== Formatting Code ==="
	@command -v ruff >/dev/null 2>&1 && ruff check src/ --fix || echo "Install ruff: pip install ruff"
	@command -v black >/dev/null 2>&1 && black src/ || echo "Install black: pip install black"

# Collect issues from QA run
collect-issues:
	@echo "=== Collecting Issues ==="
	@if [ -z "$(RUN_ID)" ]; then \
		RUN_ID=$$(ls -t reports/qa/ 2>/dev/null | head -1); \
		if [ -n "$$RUN_ID" ]; then \
			python scripts/qa/collect_issues.py "reports/qa/$$RUN_ID"; \
		else \
			echo "No QA runs found. Run 'make qa' first."; \
		fi \
	else \
		python scripts/qa/collect_issues.py "reports/qa/$(RUN_ID)"; \
	fi

# Cleanup
clean:
	@echo "--- Cleaning temporary files ---"
	rm -rf .pytest_cache
	rm -rf artifacts/metrics artifacts/privacy artifacts/robustness artifacts/evidence_pack
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f artifacts/report.html
