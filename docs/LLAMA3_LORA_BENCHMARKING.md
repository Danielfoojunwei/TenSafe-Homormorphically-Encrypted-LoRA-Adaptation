# Llama 3 LoRA Fine-Tuning & Benchmarking Pipeline

This document describes the reproducible LoRA fine-tuning and performance benchmarking pipeline for **Meta-Llama-3-8B-Instruct**.

## Overview

The pipeline provides:

1. **LoRA Fine-Tuning** - Parameter-efficient training with QLoRA support
2. **Evaluation Suite** - GSM8K, MMLU, MT-Bench, AlpacaEval 2.0
3. **Performance Benchmarks** - Inference latency, throughput, memory usage
4. **Unified Reporting** - Markdown + JSON reports with academic citations

## Quick Start

### Smoke Test (CPU-Only)

Validate the full pipeline on CPU with a tiny model:

```bash
make lora-smoke
```

This runs:
- Training with `hf-internal-testing/tiny-random-LlamaForCausalLM`
- Synthetic dataset (10 samples)
- Mock evaluation metrics
- Basic performance timing

### Full Pipeline (GPU Required)

```bash
# Set your Hugging Face token (required for Llama 3)
export HF_TOKEN=hf_your_token_here

# Run the complete benchmark pipeline
make bench-llama3
```

## Prerequisites

### Hardware Requirements

| Mode | GPU VRAM | RAM | Storage |
|------|----------|-----|---------|
| Smoke | None (CPU) | 4GB | 1GB |
| Full (QLoRA) | 16GB+ | 32GB | 50GB |
| Full (LoRA) | 40GB+ | 64GB | 50GB |

### Software Dependencies

```bash
# Install all dependencies
pip install -e ".[all]"

# Or install benchmarking extras only
pip install -e ".[bench]"
```

Required packages:
- `transformers>=4.40.0`
- `peft>=0.10.0`
- `trl>=0.8.0`
- `bitsandbytes>=0.43.0` (for QLoRA)
- `datasets>=2.18.0`
- `lm-eval>=0.4.0`
- `flash-attn>=2.5.0` (optional, for FlashAttention 2)

### Hugging Face Access

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the [Llama 3 license](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
3. Generate an access token with `read` permissions
4. Set the token:

```bash
export HF_TOKEN=hf_your_token_here

# Or use huggingface-cli
huggingface-cli login
```

## Pipeline Components

### 1. LoRA Training (`make lora-train-llama3`)

Fine-tunes Llama 3 8B Instruct using LoRA adapters.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 64 | LoRA rank (adaptation capacity) |
| `lora_alpha` | 128 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout for regularization |
| `target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Modules to adapt |
| `use_4bit` | True | Enable QLoRA (4-bit quantization) |
| `batch_size` | 4 | Per-device batch size |
| `gradient_accumulation` | 4 | Effective batch = 16 |
| `learning_rate` | 2e-4 | Peak learning rate |
| `num_epochs` | 3 | Training epochs |
| `max_seq_length` | 2048 | Maximum sequence length |

**Dataset:**

- **Source**: OpenAssistant/oasst1 (Apache 2.0 license)
- **Format**: Multi-turn conversations with Llama 3 Instruct template
- **Splits**: ~9,000 train, ~500 validation examples

**Output:**

```
artifacts/lora-llama3/
├── adapter_config.json    # PEFT adapter configuration
├── adapter_model.safetensors  # LoRA weights
├── run_config.json        # Full training configuration
└── train_metrics.json     # Loss curves, timing, memory
```

**Usage:**

```bash
# Default training
make lora-train-llama3

# Custom output directory
make lora-train-llama3 LORA_OUTPUT_DIR=my/custom/path

# Smoke mode (tiny model)
make lora-train-llama3 LORA_SMOKE=1
```

### 2. Evaluation Suite (`make eval-llama3`)

Runs comprehensive benchmarks across multiple evaluation frameworks.

**Benchmarks:**

| Benchmark | Tasks | Metric | Description |
|-----------|-------|--------|-------------|
| **GSM8K** | 8,500 | Accuracy | Grade school math reasoning |
| **MMLU** | 14,042 | Accuracy | Multi-task language understanding (57 subjects) |
| **MT-Bench** | 80 | Score 1-10 | Multi-turn conversation quality |
| **AlpacaEval 2.0** | 805 | Win Rate | Instruction-following vs GPT-4 |

**Framework:**

- GSM8K/MMLU: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- MT-Bench: Custom implementation based on [llm-judge](https://github.com/lm-sys/FastChat)
- AlpacaEval: [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)

**Output:**

```
reports/bench/<git-sha>/
├── eval_results.json      # All evaluation scores
├── gsm8k_results.json     # GSM8K detailed results
├── mmlu_results.json      # MMLU per-subject breakdown
├── mtbench_outputs.json   # MT-Bench generations
└── alpaca_outputs.json    # AlpacaEval generations
```

**Usage:**

```bash
# Default evaluation
make eval-llama3

# With custom adapter
make eval-llama3 ADAPTER_PATH=path/to/adapter

# Smoke mode
make eval-llama3 LORA_SMOKE=1
```

**MT-Bench Judge:**

MT-Bench requires an LLM judge. Configure with:

```bash
export OPENAI_API_KEY=sk-...  # For GPT-4 judge
# OR
export MT_BENCH_JUDGE_MODEL=claude-3-opus-20240229
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Performance Benchmark (`make perf-llama3`)

Measures inference performance across model variants.

**Variants Tested:**

1. **Base Model** - Original Llama 3 8B Instruct
2. **Base + LoRA** - Runtime adapter application
3. **Merged Model** - LoRA weights merged into base

**Metrics:**

| Metric | Description |
|--------|-------------|
| `ttft_ms` | Time to first token (latency) |
| `tokens_per_sec` | Generation throughput |
| `total_time_sec` | End-to-end generation time |
| `p50_latency_ms` | Median per-token latency |
| `p95_latency_ms` | 95th percentile latency |
| `peak_rss_mb` | Peak RAM usage |
| `peak_vram_mb` | Peak GPU memory |

**Benchmark Prompts:**

Uses 20 fixed prompts covering:
- Code generation
- Mathematical reasoning
- Creative writing
- Question answering
- Summarization

**Output:**

```
reports/bench/<git-sha>/
├── perf_results.json      # All performance metrics
├── perf_base.json         # Base model metrics
├── perf_lora.json         # LoRA adapter metrics
└── perf_merged.json       # Merged model metrics
```

**Usage:**

```bash
# Default performance benchmark
make perf-llama3

# With custom adapter
make perf-llama3 ADAPTER_PATH=path/to/adapter

# Smoke mode
make perf-llama3 LORA_SMOKE=1
```

### 4. Unified Report (`make bench-llama3`)

Generates a comprehensive benchmark report combining all results.

**Report Contents:**

- Training configuration and metrics
- Evaluation scores with comparisons
- Performance benchmarks with statistics
- Compliance evidence summary
- Academic citations with URLs

**Output:**

```
reports/bench/<git-sha>/
├── report.md              # Human-readable markdown
├── report.json            # Machine-readable JSON
└── ...                    # Individual component outputs
```

## CI Integration

### Smoke Tests (PR Checks)

The smoke test runs on every PR to validate the pipeline:

```yaml
# .github/workflows/qa.yml
bench-smoke:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run benchmark smoke test
      run: make bench-smoke
```

### Full Benchmarks (Nightly)

Full GPU benchmarks run nightly:

```yaml
# .github/workflows/compliance-full.yml
bench-llama3-full:
  runs-on: [self-hosted, gpu]
  if: github.event_name == 'schedule'
  steps:
    - uses: actions/checkout@v4
    - name: Run full Llama3 benchmark
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: make bench-llama3
```

## Runtime Expectations

| Task | Smoke Mode | Full Mode (A100 80GB) |
|------|------------|----------------------|
| LoRA Training | ~30 seconds | ~2-4 hours |
| GSM8K Eval | ~10 seconds | ~30 minutes |
| MMLU Eval | ~10 seconds | ~2 hours |
| MT-Bench | ~5 seconds | ~1 hour |
| AlpacaEval | ~5 seconds | ~2 hours |
| Performance | ~20 seconds | ~15 minutes |
| **Total** | **~1-2 minutes** | **~6-10 hours** |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```bash
# Enable gradient checkpointing (automatic with QLoRA)
# Reduce batch size
export BATCH_SIZE=2
```

**2. Llama 3 Access Denied**

```
Error: You need to agree to the license
```

Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and click "Agree and access repository".

**3. FlashAttention Not Available**

```
Warning: FlashAttention 2 not available, using standard attention
```

Install with:
```bash
pip install flash-attn --no-build-isolation
```

**4. bitsandbytes Issues**

```
Error: bitsandbytes CUDA setup failed
```

Ensure CUDA toolkit matches your PyTorch version:
```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### Validation

Verify installation:

```bash
# Check all dependencies
python -c "
import transformers
import peft
import trl
import datasets
print('Core dependencies OK')

try:
    import bitsandbytes
    print('bitsandbytes OK')
except ImportError:
    print('bitsandbytes not available (QLoRA disabled)')

try:
    import flash_attn
    print('FlashAttention OK')
except ImportError:
    print('FlashAttention not available')
"
```

## Citations

When using this pipeline, please cite:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and others},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

@misc{oasst1,
  title={OpenAssistant Conversations},
  author={OpenAssistant Contributors},
  year={2023},
  url={https://huggingface.co/datasets/OpenAssistant/oasst1}
}

@misc{llama3,
  title={Llama 3 Model Card},
  author={Meta AI},
  year={2024},
  url={https://github.com/meta-llama/llama3}
}
```

## License

- **Pipeline Code**: Apache 2.0
- **Llama 3 Model**: [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/)
- **OASST1 Dataset**: Apache 2.0
- **Benchmark Frameworks**: Various (see individual licenses)
