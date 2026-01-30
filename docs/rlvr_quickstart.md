# RLVR (Reinforcement Learning with Verifiable Rewards) Quickstart

This guide shows how to use TenSafe's RLVR training mode for fine-tuning language models with reward functions.

## Overview

RLVR enables fine-tuning language models using reinforcement learning with user-defined reward functions. This is useful when:

- You want to optimize for specific behaviors (e.g., format compliance)
- You have a reward model or verifier
- You want to fine-tune based on task-specific metrics

## Key Components

1. **Rollout Sampler**: Generates response trajectories from prompts
2. **Reward Function**: Scores trajectories based on desired behavior
3. **RL Algorithm**: Updates policy based on rewards (REINFORCE or PPO)
4. **Training Loop**: Orchestrates the RLVR pipeline

## Quick Start

### Minimal RLVR Example

```python
from tensafe.rlvr import (
    MockRolloutSampler,
    REINFORCE,
    REINFORCEConfig,
    TrajectoryBatch,
    resolve_reward,
)

# 1. Create rollout sampler
sampler = MockRolloutSampler(max_new_tokens=64, seed=42)

# 2. Create reward function
reward_fn = resolve_reward("keyword_contains", keywords=["solution", "answer"])

# 3. Create RL algorithm
reinforce = REINFORCE(REINFORCEConfig(
    use_baseline=True,
    normalize_advantages=True,
    entropy_coef=0.01,
))

# 4. Training loop
prompts = ["Explain quantum computing", "What is machine learning?"]

for epoch in range(10):
    # Generate trajectories
    batch = sampler.generate_trajectories(prompts)

    # Compute rewards
    for traj in batch:
        traj.reward = reward_fn(traj.prompt, traj.response)

    # Update policy
    result = reinforce.update(batch, training_client)

    print(f"Epoch {epoch}: loss={result.policy_loss:.4f}, reward={batch.mean_reward:.4f}")
```

## Reward Functions

### Built-in Rewards

```python
from tensafe.rlvr.reward import resolve_reward

# Keyword-based reward
reward_fn = resolve_reward("keyword_contains", keywords=["important", "key"])

# Length-based reward
reward_fn = resolve_reward("length_penalty", target_length=50)

# Format compliance
reward_fn = resolve_reward("format_compliance", required_format="json")
```

### Custom Reward Functions

```python
from tensafe.rlvr.reward import register_reward

@register_reward("my_reward")
def my_custom_reward(prompt, response, meta=None, **kwargs):
    """Custom reward based on response quality."""
    score = 0.0

    # Check for specific content
    if "the answer is" in response.lower():
        score += 0.5

    # Penalize very short responses
    if len(response) < 20:
        score -= 0.3

    return score

# Use the registered reward
reward_fn = resolve_reward("my_reward")
```

### Composite Rewards

```python
reward_fn = resolve_reward(
    "composite",
    rewards=[
        {"name": "keyword_contains", "weight": 0.5, "kwargs": {"keywords": ["solution"]}},
        {"name": "length_penalty", "weight": 0.3, "kwargs": {"target_length": 100}},
        {"name": "format_compliance", "weight": 0.2, "kwargs": {"required_format": "json"}},
    ],
)
```

## Algorithms

### REINFORCE

Basic policy gradient algorithm with variance reduction:

```python
from tensafe.rlvr.algorithms import REINFORCE, REINFORCEConfig

config = REINFORCEConfig(
    learning_rate=1e-5,
    use_baseline=True,
    baseline_decay=0.99,
    normalize_advantages=True,
    entropy_coef=0.01,
)

algo = REINFORCE(config)
result = algo.update(batch, client)
```

### PPO (Proximal Policy Optimization)

More stable training with clipped objective:

```python
from tensafe.rlvr.algorithms import PPO, PPOConfig

config = PPOConfig(
    learning_rate=1e-5,
    clip_range=0.2,
    ppo_epochs=4,
    entropy_coef=0.01,
    target_kl=0.01,
)

algo = PPO(config)
result = algo.update(batch, client)
```

## Trajectories and Batches

### Trajectory Data Structure

```python
from tensafe.rlvr.rollout import Trajectory, TrajectoryBatch

# Create a trajectory
traj = Trajectory(
    prompt="What is 2+2?",
    prompt_tokens=[1, 2, 3, 4],
    response="The answer is 4",
    response_tokens=[5, 6, 7, 8, 9],
    logprobs=[-0.5, -0.3, -0.2, -0.4, -0.3],
    attention_mask=[1, 1, 1, 1, 1, 1, 1, 1, 1],
    reward=1.0,
)

# Access properties
print(f"Full tokens: {traj.full_tokens}")
print(f"Response length: {traj.num_response_tokens}")
print(f"Mean logprob: {traj.mean_logprob}")
```

### Batch Operations

```python
# Create batch
batch = TrajectoryBatch(trajectories=[traj1, traj2, traj3])

# Batch statistics
print(f"Mean reward: {batch.mean_reward}")
print(f"Std reward: {batch.std_reward}")

# Compute advantages
batch.compute_advantages(baseline=0.5, normalize=True)

# Iterate
for traj in batch:
    print(f"Prompt: {traj.prompt}, Advantage: {traj.advantage}")
```

## Training Configuration

Configure RLVR in `configs/train_rlvr.yaml`:

```yaml
training:
  mode: rlvr
  seed: 42
  max_steps: 1000

rlvr:
  algorithm: reinforce  # or ppo
  rollout:
    max_new_tokens: 64
    temperature: 0.7
    top_p: 0.9

  reward:
    type: keyword_contains
    kwargs:
      keywords: ["answer", "solution"]

  algorithm_config:
    learning_rate: 1e-5
    use_baseline: true
    normalize_advantages: true
    entropy_coef: 0.01
```

## Checkpointing

Save and resume RLVR training:

```python
# Save checkpoint
state = algo.get_state()
save_checkpoint(state, "checkpoint.json")

# Resume training
new_algo = REINFORCE(config)
new_algo.load_state(load_checkpoint("checkpoint.json"))
# Training continues from saved step
```

## Best Practices

### 1. Start with REINFORCE

REINFORCE is simpler and easier to debug. Switch to PPO only if you need more stability.

### 2. Use Advantage Normalization

Always normalize advantages to reduce variance:

```python
config = REINFORCEConfig(normalize_advantages=True)
```

### 3. Add Entropy Bonus

Prevent policy collapse with entropy regularization:

```python
config = REINFORCEConfig(entropy_coef=0.01)
```

### 4. Monitor KL Divergence

Track how much the policy changes:

```python
result = algo.update(batch, client)
if result.kl_div > 0.1:
    print("Warning: Large policy change")
```

### 5. Use Multiple Samples Per Prompt

Generate multiple responses per prompt to reduce variance:

```python
# Generate 4 responses per prompt
prompts = ["Prompt 1"] * 4 + ["Prompt 2"] * 4
batch = sampler.generate_trajectories(prompts)
```

## Example: Toy RLVR Task

See the complete example in `examples/rlvr_toy_task/run_toy_rlvr.py`:

```bash
python examples/rlvr_toy_task/run_toy_rlvr.py --steps 50
```

This example trains a mock model to include specific keywords in responses.

## Troubleshooting

### Reward Not Improving

- Check that reward function returns correct values
- Increase learning rate or number of steps
- Add entropy bonus to encourage exploration

### High Variance in Returns

- Enable advantage normalization
- Use PPO instead of REINFORCE
- Increase batch size

### Policy Collapse

- Add entropy bonus
- Reduce learning rate
- Use KL penalty with PPO

## See Also

- [Custom Loss Quickstart](custom_loss_quickstart.md)
- [Baseline SFT Documentation](dev/baseline.md)
- [API Reference](api/rlvr.md)
