# Quality Assurance and Benchmarking

This document explains how to run the QA suite, interpret the evidence pack, and reproduce benchmark results.

## Quick Start

```bash
# Run the full QA suite
make qa

# Run just the tests
make test

# Run benchmarks
make bench-smoke    # Quick (~5 min)
make bench-full     # Full (~30 min)

# Build evidence pack
make evidence
```

## QA Architecture

### Test Organization

```
tests/
├── unit/                    # Fast, isolated unit tests
├── integration/             # End-to-end integration tests
├── regression/              # Privacy invariant + stability tests
│   ├── canonical_prompts.jsonl  # Standard prompt set
│   └── test_privacy_invariants.py
└── security/                # Security-focused tests
```

### Privacy Modes

The test matrix runs across three privacy modes:

| Mode | Description | Environment Variables |
|------|-------------|----------------------|
| `off` | Baseline (no privacy) | `TINKER_PRIVACY_MODE=off` |
| `tdx_base_only` | TDX enclave only | `TINKER_PRIVACY_MODE=tdx_base_only`, `TINKER_TDX_MODE=mock` |
| `tdx_plus_moai_lora` | TDX + MOAI encrypted LoRA | `TINKER_PRIVACY_MODE=tdx_plus_moai_lora`, `TINKER_TDX_MODE=mock`, `TINKER_MOAI_MODE=mock`, `TINKER_STRICT_PRIVACY=1` |

### Running the Test Matrix

```bash
# Run all modes
python scripts/qa/test_matrix.py

# Run specific modes
python scripts/qa/test_matrix.py --modes off tdx_base_only

# Custom output directory
python scripts/qa/test_matrix.py --output-dir reports/qa/custom
```

## Benchmarking

### Scenarios

| Scenario | Description | Metrics |
|----------|-------------|---------|
| `schema_validation` | Pydantic schema validation | p50/p95 latency, throughput |
| `encryption_1mb` | AES-256-GCM encrypt/decrypt | p50/p95 latency, throughput |
| `hash_chain` | SHA-256 audit chain hashing | p50/p95 latency, throughput |
| `dp_accountant` | RDP privacy accounting | p50/p95 latency, throughput |
| `artifact_store_100kb` | Encrypted artifact save/load | p50/p95 latency, memory delta |

### Running Benchmarks

```bash
# Smoke mode (~5 min, fewer iterations)
python scripts/bench/run_benchmarks.py --mode smoke

# Full mode (~30 min, statistical significance)
python scripts/bench/run_benchmarks.py --mode full

# Custom output
python scripts/bench/run_benchmarks.py --mode full --output-dir reports/bench/custom
```

### Output Format

Benchmarks produce:
- `bench_<mode>.json`: Machine-readable results
- `bench_<mode>.md`: Human-readable summary

Example output:

```json
{
  "timestamp": "20260129_120000",
  "git_sha": "abc123",
  "scenarios": [
    {
      "scenario": "encryption_1mb",
      "latency_p50_ms": 12.5,
      "latency_p95_ms": 15.2,
      "throughput_ops": 78.5
    }
  ]
}
```

## Value Evidence Pack

The evidence pack proves three value claims:

### Claim 1: Privacy Enforcement

**What we measure**: Privacy invariant tests across all modes
**How we measure**: Functional tests verify:
- Encryption always applied to artifacts
- Hash chain integrity maintained
- No banned substrings in logs
- Key isolation per tenant

**Passing criteria**: All invariant tests pass in all modes

### Claim 2: Overhead Bounded

**What we measure**: Performance overhead vs baseline
**How we measure**: Benchmarks comparing:
- Encryption overhead < 50ms p95 for 1MB
- Hash chain overhead < 1ms p95 per operation
- DP accounting overhead < 5ms p95 per step

**Passing criteria**: All overhead thresholds met

### Claim 3: Reproducibility

**What we measure**: Result stability across runs
**How we measure**:
- Determinism with fixed seeds
- Test stability across modes
- Flakiness rate tracking

**Passing criteria**: >95% test stability

### Building the Evidence Pack

```bash
python scripts/evidence/build_value_evidence.py \
  --qa-dir reports/qa \
  --bench-dir reports/bench \
  --output-dir reports/value_evidence
```

### Output Files

- `evidence.json`: Machine-readable evidence
- `evidence.md`: Human-readable report

## CI Integration

### On Every PR

1. Lint (ruff)
2. Type check (mypy)
3. Unit tests
4. Integration tests
5. Smoke benchmarks
6. Evidence pack generation

### Nightly

1. Full test matrix across all privacy modes
2. Full benchmark suite
3. Evidence pack with statistical analysis

### Artifacts

All CI runs produce downloadable artifacts:
- `test-results-<mode>`: Test XML reports
- `bench-results`: Benchmark JSON/MD
- `value-evidence-pack`: Evidence pack

## Reproducing Results

### Local Reproduction

```bash
# 1. Set up environment
pip install -e ".[all]"
mkdir -p frontend/dist

# 2. Run QA suite
make qa

# 3. Check results
cat reports/value_evidence/evidence.md
cat reports/bench/*/bench_smoke.md
```

### CI Reproduction

1. Find the workflow run in GitHub Actions
2. Download artifacts
3. Compare with local run

### Seed Control

For deterministic results:

```bash
export TG_DETERMINISTIC=1
export PYTHONHASHSEED=42
python scripts/bench/run_benchmarks.py --mode full
```

## Interpreting Results

### Test Results

```
PASS: All tests in mode pass
PARTIAL: >50% tests pass
FAIL: <50% tests pass
```

### Benchmark Thresholds

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| p50 latency | <10ms | <50ms | >50ms |
| p95 latency | <20ms | <100ms | >100ms |
| Throughput | >100 ops/s | >10 ops/s | <10 ops/s |

### Evidence Pack Scores

| Score | Status |
|-------|--------|
| 100% | All claims pass |
| 67-99% | Partial evidence |
| <67% | Insufficient evidence |

## Limitations

1. **Mock Mode**: TDX and MOAI run in mock mode, not real hardware
2. **Single Machine**: Benchmarks run on one machine, may vary
3. **Statistical Power**: Smoke mode has fewer iterations

## Troubleshooting

### Tests Fail to Import

```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### Missing Dependencies

```bash
pip install python-multipart psutil
```

### Frontend Not Found

```bash
mkdir -p frontend/dist
```

## Contributing

When adding new tests:

1. Add to appropriate directory (`unit/`, `integration/`, `regression/`)
2. Use pytest markers (`@pytest.mark.regression`)
3. Update this documentation if adding new scenarios
