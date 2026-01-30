# Training Guide

This guide covers training workflows and best practices with TenSafe.

## Training Architecture

TenSafe uses an asynchronous training model where operations are queued and executed on remote compute:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Queue     │────▶│   Compute   │
│   SDK       │◀────│   Service   │◀────│   Backend   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      │            FutureHandle                │
      └────────────────────────────────────────┘
```

## Core Training Primitives

### forward_backward()

Computes the forward pass and backpropagates gradients:

```python
from tg_tinker import BatchData

# Using BatchData object
batch = BatchData(
    input_ids=[[1, 2, 3, 4], [5, 6, 7, 8]],
    attention_mask=[[1, 1, 1, 1], [1, 1, 1, 1]],
    labels=[[2, 3, 4, -100], [6, 7, 8, -100]],
)

future = tc.forward_backward(batch)

# Or using dict
future = tc.forward_backward({
    "input_ids": tokens,
    "attention_mask": mask,
    "labels": labels,
})

# Get result
result = future.result()
print(f"Loss: {result.loss}")
print(f"Grad norm: {result.grad_norm}")
print(f"Tokens processed: {result.tokens_processed}")
```

### optim_step()

Applies accumulated gradients with the optimizer:

```python
# Basic optimizer step
opt_future = tc.optim_step()
opt_result = opt_future.result()

print(f"New step: {opt_result.step}")
print(f"Learning rate: {opt_result.learning_rate}")

# With DP noise (when DP is enabled)
opt_future = tc.optim_step(apply_dp_noise=True)
opt_result = opt_future.result()

if opt_result.dp_metrics:
    print(f"Epsilon spent: {opt_result.dp_metrics.epsilon_spent}")
```

## Async Execution Patterns

### Overlapping Operations

Queue multiple operations before waiting:

```python
# Queue forward-backward immediately
fb_future = tc.forward_backward(batch)

# Queue optim step without waiting
opt_future = tc.optim_step()

# Now wait for results
fb_result = fb_future.result()
opt_result = opt_future.result()
```

### Pipelining Batches

Process multiple batches in a pipeline:

```python
futures = []

for i, batch in enumerate(batches):
    # Submit forward-backward
    fb_future = tc.forward_backward(batch)
    futures.append(("fb", i, fb_future))

    # Submit optim step
    opt_future = tc.optim_step()
    futures.append(("opt", i, opt_future))

# Collect all results
for op_type, batch_idx, future in futures:
    result = future.result()
    if op_type == "fb":
        print(f"Batch {batch_idx}: loss={result.loss:.4f}")
```

### Timeouts and Cancellation

```python
from tg_tinker import FutureTimeoutError

future = tc.forward_backward(batch)

try:
    # Wait with timeout
    result = future.result(timeout=120)
except FutureTimeoutError:
    # Cancel if taking too long
    cancelled = future.cancel()
    print(f"Operation cancelled: {cancelled}")
```

## LoRA Configuration

### Basic LoRA Setup

```python
from tg_tinker import LoRAConfig

lora = LoRAConfig(
    rank=16,           # Rank of low-rank matrices
    alpha=32,          # Scaling factor (typically 2x rank)
    dropout=0.05,      # Dropout rate
    target_modules=[   # Modules to apply LoRA
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
    bias="none",       # "none", "all", or "lora_only"
)
```

### Choosing LoRA Rank

| Rank | Memory | Quality | Use Case |
|------|--------|---------|----------|
| 4-8 | Low | Good | Simple tasks, quick experiments |
| 16-32 | Medium | Very Good | General fine-tuning |
| 64-128 | High | Excellent | Complex tasks, domain adaptation |
| 256+ | Very High | Marginal gains | Specialized research |

### Full Fine-tuning

For full parameter updates, omit lora_config:

```python
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=None,  # Full fine-tuning
    optimizer=OptimizerConfig(learning_rate=1e-5),
)
```

## Optimizer Configuration

### AdamW (Default)

```python
from tg_tinker import OptimizerConfig

optimizer = OptimizerConfig(
    name="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

### SGD with Momentum

```python
optimizer = OptimizerConfig(
    name="sgd",
    learning_rate=1e-3,
    weight_decay=0.0,
)
```

### Supported Optimizers

- `adamw` - AdamW with decoupled weight decay
- `adam` - Standard Adam
- `sgd` - Stochastic Gradient Descent
- `adafactor` - Memory-efficient Adam alternative

## Pluggable Loss Functions

TenSafe supports custom loss functions for advanced training scenarios.

### Built-in Losses

```python
from tensafe.training.losses import resolve_loss

# Token cross-entropy (default for language modeling)
loss_fn = resolve_loss("token_ce", ignore_index=-100)

# Margin ranking loss
loss_fn = resolve_loss("margin_ranking", margin=0.5)

# Contrastive loss
loss_fn = resolve_loss("contrastive", temperature=0.07)

# Mean squared error
loss_fn = resolve_loss("mse")
```

### Custom Loss Functions

Register your own loss:

```python
from tensafe.training.losses import register_loss

@register_loss("focal_loss")
def focal_loss(outputs, batch, gamma=2.0, **kwargs):
    """Focal loss for imbalanced data."""
    logits = outputs.logits
    labels = batch["labels"]

    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * ce_loss

    return {
        "loss": focal.mean(),
        "metrics": {"gamma": gamma, "pt_mean": pt.mean().item()}
    }

# Use custom loss
loss_fn = resolve_loss("focal_loss", gamma=2.5)
```

### Dotted Path Import

Load loss from external module:

```python
# From my_package/losses.py
loss_fn = resolve_loss("my_package.losses:weighted_ce")
```

### YAML Configuration

```yaml
# training_config.yaml
training:
  mode: sft

loss:
  type: token_ce
  kwargs:
    ignore_index: -100
    label_smoothing: 0.1
```

See [Custom Loss Quickstart](../custom_loss_quickstart.md) for more examples.

## RLVR Mode (Reinforcement Learning with Verifiable Rewards)

Train models using RL with custom reward functions.

### Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `sft` | Supervised Fine-Tuning | Standard loss-based training |
| `rlvr` | RL with Verifiable Rewards | Policy gradient optimization |

### Basic RLVR Setup

```python
from tensafe.rlvr import (
    MockRolloutSampler,
    REINFORCE,
    REINFORCEConfig,
    resolve_reward,
)

# Create reward function
reward_fn = resolve_reward("keyword_contains", keywords=["solution", "answer"])

# Create RL algorithm
algo = REINFORCE(REINFORCEConfig(
    use_baseline=True,
    normalize_advantages=True,
    entropy_coef=0.01,
))

# Training loop
sampler = MockRolloutSampler(max_new_tokens=64)
prompts = ["Solve the equation x + 2 = 5", "What is 3 * 4?"]

for epoch in range(10):
    batch = sampler.generate_trajectories(prompts)
    for traj in batch:
        traj.reward = reward_fn(traj.prompt, traj.response)

    result = algo.update(batch, training_client)
    print(f"Mean reward: {batch.mean_reward:.3f}")
```

### PPO Algorithm

For more stable training:

```python
from tensafe.rlvr import PPO, PPOConfig

algo = PPO(PPOConfig(
    clip_range=0.2,
    ppo_epochs=4,
    target_kl=0.01,
    entropy_coef=0.01,
))
```

### Custom Reward Functions

```python
from tensafe.rlvr import register_reward

@register_reward("code_quality")
def code_quality_reward(prompt: str, response: str, meta=None) -> float:
    """Reward based on code quality metrics."""
    score = 0.0
    if "def " in response:
        score += 0.3
    if "return" in response:
        score += 0.3
    if len(response) > 50:
        score += 0.4
    return score

reward_fn = resolve_reward("code_quality")
```

See [RLVR Quickstart](../rlvr_quickstart.md) for comprehensive guide.

## Gradient Accumulation

Simulate larger batch sizes:

```python
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
    batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size: 32
)

tc = service.create_training_client(config)

# Gradients accumulate across forward_backward calls
for micro_batch in micro_batches:
    tc.forward_backward(micro_batch).result()

# Single optimizer step uses accumulated gradients
tc.optim_step().result()
```

## Checkpoint Management

### Saving Checkpoints

```python
# Save full state
checkpoint = tc.save_state(
    include_optimizer=True,
    metadata={
        "epoch": 1,
        "validation_loss": 0.42,
        "dataset_version": "v2",
    }
)

print(f"Artifact ID: {checkpoint.artifact_id}")
print(f"Size: {checkpoint.size_bytes} bytes")
print(f"Hash: {checkpoint.content_hash}")
```

### Loading Checkpoints

```python
# Resume from checkpoint
result = tc.load_state(artifact_id="art-xxx-yyy")

print(f"Restored to step: {result.step}")
print(f"Status: {result.status}")
```

### Downloading Artifacts

```python
# Download encrypted artifact
data = service.pull_artifact(checkpoint.artifact_id)

# Save locally (still encrypted)
with open("checkpoint.bin", "wb") as f:
    f.write(data)
```

## Training Client State

### Monitoring Progress

```python
# Check current state
print(f"Step: {tc.step}")
print(f"Status: {tc.status}")

# Refresh from server
tc.refresh()
print(f"Updated step: {tc.step}")
```

### Client Status Values

| Status | Description |
|--------|-------------|
| `INITIALIZING` | Client being set up |
| `READY` | Ready for operations |
| `BUSY` | Processing an operation |
| `ERROR` | Error state |
| `TERMINATED` | Client shut down |

## Best Practices

### 1. Use Appropriate Batch Sizes

```python
# Start conservative, increase gradually
config = TrainingConfig(
    batch_size=4,
    gradient_accumulation_steps=8,  # Effective: 32
)
```

### 2. Monitor Loss and Gradients

```python
for batch in dataloader:
    result = tc.forward_backward(batch).result()

    # Check for training issues
    if result.loss > 10.0:
        print("Warning: High loss detected")
    if result.grad_norm > 100.0:
        print("Warning: Gradient explosion")
```

### 3. Regular Checkpointing

```python
checkpoint_every = 100

for step, batch in enumerate(dataloader):
    tc.forward_backward(batch).result()
    tc.optim_step().result()

    if (step + 1) % checkpoint_every == 0:
        checkpoint = tc.save_state()
        print(f"Checkpoint saved: {checkpoint.artifact_id}")
```

### 4. Handle Errors Gracefully

```python
from tg_tinker import TGTinkerError

try:
    for batch in dataloader:
        tc.forward_backward(batch).result()
        tc.optim_step().result()
except TGTinkerError as e:
    print(f"Training error: {e}")
    # Save emergency checkpoint
    tc.save_state(metadata={"error": str(e)})
    raise
```

## Next Steps

- [Sampling Guide](sampling.md) - Text generation
- [Privacy Guide](privacy.md) - DP and homomorphic encryption
- [API Reference](../api-reference/training-client.md) - Full TrainingClient API
