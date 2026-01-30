# Baseline SFT Documentation

This document describes how to run baseline SFT (Supervised Fine-Tuning) with TenSafe, expected outputs, and known nondeterminism sources.

## Overview

TenSafe's training stack supports SFT through the `TrainingClient` API, which provides:

- `forward_backward(batch)` - Compute loss and gradients
- `optim_step()` - Apply optimizer update
- `sample(prompts)` - Generate text from current model
- `save_state()` / `load_state()` - Checkpoint management

## Running Baseline SFT

### Quick Smoke Test

Run a minimal SFT training loop (20 steps) to verify the training stack works:

```bash
python scripts/baseline_sft_smoke.py
```

This script:
1. Creates a mock training client
2. Runs 20 forward-backward + optimizer steps
3. Asserts loss decreases over training
4. Samples text from the trained model
5. Saves and loads a checkpoint

### Expected Output

```
=== TenSafe Baseline SFT Smoke Test ===

[Step 1/20] loss=2.490, grad_norm=1.5xx
[Step 2/20] loss=2.480, grad_norm=1.5xx
...
[Step 20/20] loss=2.300, grad_norm=1.5xx

Training complete!
  Initial loss: 2.490
  Final loss: 2.300
  Loss decreased: True

Sampling from trained model...
  Prompt: "Once upon a time"
  Completion: [Mock completion for step 20]

Checkpoint test...
  Saved checkpoint: art-xxx
  Loaded checkpoint successfully

=== All baseline tests passed! ===
```

### Running as a Test

```bash
pytest tests/test_baseline_sft.py -v
```

## TrainingClient API

### Creating a Training Client

```python
from tg_tinker import ServiceClient

# Connect to TenSafe service
client = ServiceClient(config)

# Create training client
tc = client.create_training_client({
    "model_ref": "meta-llama/Llama-3-8B",
    "lora_config": {
        "rank": 16,
        "alpha": 32.0,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    "optimizer": {
        "name": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 0.01
    },
    "batch_size": 8
})
```

### Training Loop

```python
for batch in dataloader:
    # Forward-backward pass (async)
    fb_future = tc.forward_backward(batch)

    # Optimizer step (can overlap with next batch prep)
    opt_future = tc.optim_step()

    # Wait for results
    fb_result = fb_future.result()
    opt_result = opt_future.result()

    print(f"Step {opt_result.step}: loss={fb_result.loss:.4f}")
```

### Checkpointing

```python
# Save checkpoint
save_result = tc.save_state(metadata={"notes": "epoch 1"})
print(f"Saved: {save_result.artifact_id}")

# Load checkpoint
load_result = tc.load_state(save_result.artifact_id)
print(f"Loaded step: {load_result.step}")
```

### Text Generation

```python
result = tc.sample(
    prompts=["Once upon a time"],
    max_tokens=100,
    temperature=0.7
)
print(result.samples[0].completion)
```

## Known Nondeterminism Sources

### 1. Mock Backend Randomness

The `MockMLBackend` uses `secrets.randbelow()` for gradient norm simulation, which introduces nondeterminism. For deterministic testing, use the seed control features.

### 2. Floating Point Accumulation

When running on GPU, floating point accumulation order can vary between runs. Use:
- `torch.use_deterministic_algorithms(True)`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` environment variable

### 3. DP Noise Injection

When differential privacy is enabled, Gaussian noise is injected into gradients. This is inherently random but can be seeded for reproducibility.

### Achieving Reproducibility

For reproducible runs:

```python
import torch
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Enable deterministic mode
torch.use_deterministic_algorithms(True)

# Set seeds before training
set_seed(42)
```

## Golden Artifacts

Baseline golden artifacts are stored in `tests/golden/`:

- `tests/golden/baseline_sft_metrics.json` - Expected metrics from smoke test
- `tests/golden/baseline_loss_curve.json` - Expected loss progression

These artifacts are used for regression testing to ensure training behavior remains consistent.

## Configuration Reference

### TrainingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_ref` | str | required | Model reference (HF hub or path) |
| `lora_config` | LoRAConfig | None | LoRA configuration |
| `optimizer` | OptimizerConfig | AdamW | Optimizer settings |
| `dp_config` | DPConfig | None | Differential privacy settings |
| `batch_size` | int | 8 | Training batch size |
| `gradient_accumulation_steps` | int | 1 | Gradient accumulation |
| `max_steps` | int | None | Maximum training steps |

### LoRAConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rank` | int | 16 | LoRA rank (r) |
| `alpha` | float | 32.0 | LoRA alpha scaling |
| `dropout` | float | 0.05 | Dropout rate |
| `target_modules` | list | ["q_proj", "v_proj", "k_proj", "o_proj"] | Target modules |
| `bias` | str | "none" | Bias handling |

### OptimizerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | "adamw" | Optimizer name |
| `learning_rate` | float | 1e-4 | Learning rate |
| `weight_decay` | float | 0.01 | Weight decay |
| `betas` | tuple | (0.9, 0.999) | Adam betas |
| `eps` | float | 1e-8 | Epsilon |

## Next Steps

After validating baseline SFT:

1. **Pluggable Loss Functions** - See `docs/custom_loss_quickstart.md`
2. **RLVR Training** - See `docs/rlvr_quickstart.md`
3. **Benchmarking** - Run `scripts/run_ci.sh` for full validation
