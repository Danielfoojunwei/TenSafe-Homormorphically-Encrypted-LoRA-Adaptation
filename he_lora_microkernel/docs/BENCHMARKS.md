# HE-LoRA Microkernel Benchmarks

## Benchmark Types

### 1. Microbenchmarks

Measure individual operation costs:
- Encryption
- Decryption
- Ct×Pt multiplication
- Rotation
- Rescale

### 2. End-to-End Benchmarks

Measure complete HE-LoRA inference:
- Tokens per second
- Aggregate throughput
- Operation counts
- HE time percentage

## Running Benchmarks

### Microbenchmarks

```bash
# Run with simulation backend
python -m he_lora_microkernel.bench.bench_micro

# With specific backend (when available)
python -m he_lora_microkernel.bench.bench_micro --backend HEONGPU

# Save results
python -m he_lora_microkernel.bench.bench_micro --output micro_results.json
```

### End-to-End Benchmarks

```bash
# Default sweep
python -m he_lora_microkernel.bench.bench_end2end

# Custom parameters
python -m he_lora_microkernel.bench.bench_end2end \
    --batch-sizes 1 4 8 16 \
    --hidden-sizes 512 1024 2048 \
    --ranks 8 16 32 \
    --num-tokens 100

# Save results
python -m he_lora_microkernel.bench.bench_end2end --output e2e_results.json
```

## Metrics Reported

### Throughput Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| tokens_per_second | Per-sequence throughput | tok/s |
| aggregate_tokens_per_second | batch × tok/s | tok/s |
| ms_per_token | Latency per token | ms |

### Operation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| avg_rotations_per_token | Average rotations | ≤ 16 |
| avg_keyswitches_per_token | Average keyswitches | ≤ 16 |
| avg_rescales_per_token | Average rescales | ≤ 8 |

### Time Breakdown

| Metric | Description |
|--------|-------------|
| avg_he_time_ms | Total HE time per token |
| avg_encrypt_time_ms | Encryption time |
| avg_compute_time_ms | HE computation time |
| avg_decrypt_time_ms | Decryption time |
| he_time_percentage | HE time / total time |

## Reference Results

### Simulation Backend

Results with `BackendType.SIMULATION` (CPU, for testing):

| hidden_size | rank | batch_size | tok/s | rotations |
|-------------|------|------------|-------|-----------|
| 512 | 8 | 1 | ~1000 | 0-2 |
| 512 | 8 | 4 | ~250 | 0-2 |
| 512 | 16 | 8 | ~125 | 0-2 |
| 1024 | 16 | 4 | ~100 | 1-3 |

*Note: Simulation backend does not reflect real GPU performance.*

### GPU Backend (Expected)

With real GPU HE backend (HEonGPU, etc.):

| hidden_size | rank | batch_size | tok/s | rotations |
|-------------|------|------------|-------|-----------|
| 512 | 8 | 1 | ~50 | 0-2 |
| 512 | 8 | 4 | ~40 | 0-2 |
| 1024 | 16 | 8 | ~20 | 1-3 |
| 4096 | 16 | 1 | ~10 | 3-5 |

*Performance varies by GPU model and backend implementation.*

## Reproducing Results

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pytest

# For GPU backends (when available)
pip install heongpu  # or fideslib, openfhe-gpu
```

### Run Full Benchmark Suite

```bash
# Microbenchmarks
python -m he_lora_microkernel.bench.bench_micro \
    --iterations 1000 \
    --output results/micro.json

# End-to-end (comprehensive)
python -m he_lora_microkernel.bench.bench_end2end \
    --batch-sizes 1 2 4 8 16 \
    --hidden-sizes 256 512 1024 2048 4096 \
    --ranks 4 8 16 32 \
    --num-tokens 100 \
    --output results/e2e.json
```

### Generate Report

```python
import json

with open('results/e2e.json') as f:
    results = json.load(f)

# Best configurations
print("Best throughput:", results['best_configurations']['highest_throughput'])
print("Lowest rotations:", results['best_configurations']['lowest_rotations'])
```

## Benchmark Configurations

### Common Model Sizes

| Model | hidden_size | Recommended batch_size |
|-------|-------------|------------------------|
| Llama-2 7B | 4096 | 1-4 |
| Llama-2 13B | 5120 | 1-2 |
| Llama-2 70B | 8192 | 1 |
| Mistral 7B | 4096 | 1-4 |

### CI Benchmark Configuration

For CI, use a fast configuration:

```bash
python -m he_lora_microkernel.bench.bench_end2end \
    --batch-sizes 1 4 \
    --hidden-sizes 256 512 \
    --ranks 8 16 \
    --num-tokens 20
```

## Performance Analysis

### Identifying Bottlenecks

1. **Rotation-bound**: High rotation count, rotation_time > 50% of compute
2. **Encryption-bound**: encrypt_time > compute_time
3. **Memory-bound**: Large slot counts, high memory usage

### Optimization Opportunities

| Symptom | Cause | Solution |
|---------|-------|----------|
| High rotations | Many blocks | Reduce hidden_size or batch_size |
| Low utilization | Small batch | Increase batch_size |
| High encrypt time | Large N | Use FAST profile |

## Continuous Integration

### CI Benchmark Checks

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmarks
  run: |
    python -m pytest he_lora_microkernel/tests/test_rotation_budget.py
    python -m he_lora_microkernel.bench.bench_end2end \
      --batch-sizes 4 \
      --hidden-sizes 512 \
      --ranks 16 \
      --num-tokens 10

- name: Check budgets
  run: |
    python -c "
    from he_lora_microkernel.compiler import *
    config = LoRAConfig(hidden_size=512, rank=16, alpha=32, ...)
    schedule = compile_schedule(config, get_profile(CKKSProfile.FAST))
    assert schedule.predicted_costs.rotations_per_token <= 16
    "
```

### Regression Detection

Compare against baseline:
```python
def check_regression(current_results, baseline_file):
    with open(baseline_file) as f:
        baseline = json.load(f)

    for config, result in current_results.items():
        baseline_rot = baseline[config]['rotations']
        current_rot = result['rotations']
        if current_rot > baseline_rot * 1.1:
            raise RegressionError(f"Rotation regression: {config}")
```
