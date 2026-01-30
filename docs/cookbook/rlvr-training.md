# Cookbook: RLVR Training

Complete walkthrough for Reinforcement Learning with Verifiable Rewards on TenSafe.

## Overview

This tutorial covers:
- Setting up RLVR training
- Implementing custom reward functions
- Using REINFORCE and PPO algorithms
- Checkpoint management for RL training

## Prerequisites

```bash
pip install tg-tinker
export TS_API_KEY=ts-your-key
```

## Step 1: Understanding RLVR

RLVR (Reinforcement Learning with Verifiable Rewards) trains language models using policy gradients with custom reward functions. Unlike supervised fine-tuning, RLVR optimizes model outputs based on reward signals rather than labeled data.

**When to use RLVR:**
- Task completion with verifiable outcomes
- Code generation with execution tests
- Math problems with checkable answers
- Format compliance and structured outputs

## Step 2: Basic RLVR Setup

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig
from tensafe.rlvr import (
    MockRolloutSampler,
    REINFORCE,
    REINFORCEConfig,
    TrajectoryBatch,
    resolve_reward,
)

# Initialize service
service = ServiceClient()

# Create training client
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
)
tc = service.create_training_client(config)

# Create rollout sampler
sampler = MockRolloutSampler(max_new_tokens=64)

# Create RL algorithm
algo = REINFORCE(REINFORCEConfig(
    learning_rate=1e-5,
    use_baseline=True,
    normalize_advantages=True,
    entropy_coef=0.01,
))
```

## Step 3: Define Reward Functions

### Built-in Rewards

```python
# Keyword-based reward
reward_fn = resolve_reward("keyword_contains", keywords=["solution", "answer"])

# Length penalty
reward_fn = resolve_reward("length_penalty", target_length=100, penalty_scale=0.01)

# Format compliance
reward_fn = resolve_reward("format_compliance", patterns=[r"^\d+\.", r"Step \d+:"])
```

### Custom Reward Function

```python
from tensafe.rlvr import register_reward
import re

@register_reward("math_correctness")
def math_correctness_reward(prompt: str, response: str, meta=None) -> float:
    """
    Reward for math problems with verifiable answers.
    Expects prompt to contain expected answer in meta.
    """
    expected = meta.get("expected_answer") if meta else None
    if expected is None:
        return 0.0

    # Extract answer from response
    match = re.search(r"(?:answer|result).*?(\d+(?:\.\d+)?)", response.lower())
    if not match:
        return 0.0

    try:
        actual = float(match.group(1))
        expected = float(expected)
        # Full reward for exact match, partial for close
        if abs(actual - expected) < 0.01:
            return 1.0
        elif abs(actual - expected) < 1.0:
            return 0.5
        return 0.0
    except ValueError:
        return 0.0

# Use custom reward
reward_fn = resolve_reward("math_correctness")
```

### Code Execution Reward

```python
import subprocess
import tempfile

@register_reward("code_execution")
def code_execution_reward(prompt: str, response: str, meta=None) -> float:
    """
    Execute Python code and check output.
    """
    # Extract code block
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        return 0.0

    code = code_match.group(1)
    expected_output = meta.get("expected_output", "") if meta else ""

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            result = subprocess.run(
                ['python', f.name],
                capture_output=True,
                text=True,
                timeout=5
            )

            actual_output = result.stdout.strip()
            if actual_output == expected_output:
                return 1.0
            elif expected_output in actual_output:
                return 0.5
            return 0.0
    except (subprocess.TimeoutExpired, Exception):
        return 0.0
```

## Step 4: Training Loop

```python
# Prepare training prompts
prompts = [
    {"prompt": "What is 5 + 3?", "meta": {"expected_answer": "8"}},
    {"prompt": "What is 12 * 4?", "meta": {"expected_answer": "48"}},
    {"prompt": "What is 100 / 5?", "meta": {"expected_answer": "20"}},
]

# Training configuration
num_epochs = 20
batch_size = 4
log_interval = 5

# Training loop
for epoch in range(num_epochs):
    # Sample trajectories
    batch_prompts = [p["prompt"] for p in prompts[:batch_size]]
    batch_meta = [p.get("meta", {}) for p in prompts[:batch_size]]

    batch = sampler.generate_trajectories(batch_prompts)

    # Compute rewards
    for i, traj in enumerate(batch):
        traj.reward = reward_fn(traj.prompt, traj.response, batch_meta[i])

    # Update policy
    result = algo.update(batch, tc)

    # Log progress
    if epoch % log_interval == 0:
        print(f"Epoch {epoch}")
        print(f"  Mean reward: {batch.mean_reward:.3f}")
        print(f"  Policy loss: {result.policy_loss:.4f}")
        print(f"  Entropy: {result.entropy:.4f}")
```

## Step 5: Using PPO

PPO provides more stable training for complex tasks:

```python
from tensafe.rlvr import PPO, PPOConfig

ppo = PPO(PPOConfig(
    learning_rate=1e-5,
    clip_range=0.2,           # PPO clipping parameter
    ppo_epochs=4,             # Epochs per batch
    target_kl=0.01,           # KL divergence target for early stopping
    value_coef=0.5,           # Value function coefficient
    entropy_coef=0.01,        # Entropy bonus
    normalize_advantages=True,
    reward_normalization=True,
))

# PPO training loop
for epoch in range(num_epochs):
    batch = sampler.generate_trajectories(prompts)

    for traj in batch:
        traj.reward = reward_fn(traj.prompt, traj.response)

    result = ppo.update(batch, tc)

    print(f"Epoch {epoch}")
    print(f"  Reward: {batch.mean_reward:.3f}")
    print(f"  Policy loss: {result.policy_loss:.4f}")
    print(f"  Value loss: {result.value_loss:.4f}")
    print(f"  KL div: {result.kl_divergence:.4f}")
```

## Step 6: Checkpoint Management

```python
import json

# Save checkpoint with RL state
def save_rlvr_checkpoint(tc, algo, epoch, metrics):
    # Save model checkpoint
    checkpoint = tc.save_state(
        metadata={
            "training_mode": "rlvr",
            "epoch": epoch,
            "mean_reward": metrics["mean_reward"],
        }
    )

    # Save algorithm state
    algo_state = algo.get_state()
    with open(f"rlvr_algo_state_epoch{epoch}.json", "w") as f:
        json.dump(algo_state, f)

    print(f"Checkpoint saved: {checkpoint.artifact_id}")
    return checkpoint

# Load checkpoint
def load_rlvr_checkpoint(service, config, checkpoint_id, algo_state_path):
    # Load model
    tc = service.create_training_client(config, checkpoint_id=checkpoint_id)

    # Load algorithm state
    with open(algo_state_path, "r") as f:
        algo_state = json.load(f)

    algo = REINFORCE.from_state(algo_state)
    return tc, algo

# Usage
checkpoint = save_rlvr_checkpoint(tc, algo, epoch, {"mean_reward": batch.mean_reward})
```

## Step 7: Complete RLVR Script

```python
#!/usr/bin/env python
"""Complete RLVR training script."""

from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig
from tensafe.rlvr import (
    MockRolloutSampler,
    PPO,
    PPOConfig,
    resolve_reward,
    register_reward,
)
import re


@register_reward("structured_output")
def structured_output_reward(prompt: str, response: str, meta=None) -> float:
    """Reward responses with proper structure."""
    score = 0.0

    # Check for numbered steps
    if re.search(r"\d+\.", response):
        score += 0.3

    # Check for clear conclusion
    if any(word in response.lower() for word in ["therefore", "thus", "so", "answer"]):
        score += 0.3

    # Length appropriateness (50-200 words ideal)
    word_count = len(response.split())
    if 50 <= word_count <= 200:
        score += 0.4
    elif 30 <= word_count <= 300:
        score += 0.2

    return score


def main():
    # Setup
    service = ServiceClient()
    config = TrainingConfig(
        model_ref="meta-llama/Llama-3-8B",
        lora_config=LoRAConfig(rank=16, alpha=32),
    )
    tc = service.create_training_client(config)

    # RLVR components
    sampler = MockRolloutSampler(max_new_tokens=128)
    reward_fn = resolve_reward("structured_output")
    algo = PPO(PPOConfig(
        learning_rate=1e-5,
        clip_range=0.2,
        ppo_epochs=4,
    ))

    # Training data
    prompts = [
        "Explain how photosynthesis works.",
        "What are the three laws of thermodynamics?",
        "Describe the water cycle.",
        "How does an internal combustion engine work?",
    ]

    # Training loop
    best_reward = -float("inf")
    for epoch in range(50):
        batch = sampler.generate_trajectories(prompts)

        for traj in batch:
            traj.reward = reward_fn(traj.prompt, traj.response)

        result = algo.update(batch, tc)

        # Log every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            print(f"  Mean reward: {batch.mean_reward:.3f}")
            print(f"  Std reward: {batch.std_reward:.3f}")

            # Sample response
            print(f"  Sample: {batch.trajectories[0].response[:100]}...")

        # Save best model
        if batch.mean_reward > best_reward:
            best_reward = batch.mean_reward
            checkpoint = tc.save_state(metadata={
                "best": True,
                "epoch": epoch,
                "reward": best_reward,
            })
            print(f"New best! Reward: {best_reward:.3f}, Checkpoint: {checkpoint.artifact_id}")

    # Final save
    tc.save_state(metadata={"final": True, "epochs": 50})
    print("Training complete!")


if __name__ == "__main__":
    main()
```

## Advanced Topics

### Reward Shaping

Combine multiple rewards for complex objectives:

```python
def composite_reward(prompt: str, response: str, meta=None) -> float:
    """Combine multiple reward signals."""
    keyword_reward = resolve_reward("keyword_contains", keywords=["step"])
    length_reward = resolve_reward("length_penalty", target_length=150)

    return 0.6 * keyword_reward(prompt, response) + 0.4 * length_reward(prompt, response)
```

### KL Penalty

Prevent policy drift from reference model:

```python
from tensafe.rlvr import PPOConfig

config = PPOConfig(
    kl_penalty_coef=0.1,  # Penalize deviation from reference
    target_kl=0.02,       # Target KL divergence
)
```

### Curriculum Learning

Start with easy examples, progressively increase difficulty:

```python
def get_curriculum_prompts(epoch):
    if epoch < 10:
        return easy_prompts
    elif epoch < 30:
        return medium_prompts
    else:
        return hard_prompts
```

## Debugging Tips

### Monitor Reward Distribution

```python
import numpy as np

rewards = [traj.reward for traj in batch.trajectories]
print(f"Reward stats: min={min(rewards):.3f}, max={max(rewards):.3f}, "
      f"mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
```

### Check for Mode Collapse

```python
# Monitor response diversity
responses = [traj.response for traj in batch.trajectories]
unique_prefixes = len(set(r[:50] for r in responses))
print(f"Unique response prefixes: {unique_prefixes}/{len(responses)}")
```

### Gradient Monitoring

```python
if result.grad_norm > 10.0:
    print("Warning: Large gradient norm, consider reducing learning rate")
```

## Next Steps

- [Custom Loss Functions](../custom_loss_quickstart.md)
- [Privacy Guide](../guides/privacy.md)
- [API Reference](../api-reference/training-client.md)
