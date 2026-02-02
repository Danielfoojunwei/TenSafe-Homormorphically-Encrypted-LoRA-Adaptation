# Ray Train Distributed Training Guide

**Version**: 4.0.0
**Last Updated**: 2026-02-02

This guide covers using TenSafe with Ray Train for distributed privacy-preserving training.

## Overview

TenSafe's Ray Train integration provides:
- **Distributed DP-SGD**: Differential privacy across multiple workers
- **Secure Gradient Aggregation**: Privacy-preserving gradient sharing
- **Coordinated Privacy Accounting**: Global (ε, δ) tracking
- **Fault Tolerance**: Worker recovery with checkpointing

## Prerequisites

```bash
# Install TenSafe with Ray support
pip install tensafe[ray]>=4.0.0

# Or install dependencies separately
pip install "ray[train]>=2.9.0" torch>=2.0
```

## Quick Start

### Basic Distributed Training

```python
import ray
from tensorguard.distributed import TenSafeTrainer, TenSafeTrainingConfig
from tensorguard.distributed import DistributedDPConfig

# Initialize Ray
ray.init()

# Configure training
config = TenSafeTrainingConfig(
    # Model
    model_name="meta-llama/Llama-3-8B",

    # LoRA
    lora_rank=16,
    lora_alpha=32,

    # Training
    num_epochs=3,
    per_device_batch_size=4,
    learning_rate=1e-4,

    # Distributed
    num_workers=4,
    use_gpu=True,

    # Privacy
    dp_config=DistributedDPConfig(
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
    ),
)

# Create trainer
trainer = TenSafeTrainer(config=config)

# Train
result = trainer.fit(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print(f"Final epsilon: {result.metrics['dp_epsilon']}")
print(f"Best eval loss: {result.metrics['best_eval_loss']}")
```

### With Secure Gradient Aggregation

```python
from tensorguard.distributed import SecureAggregationConfig

config = TenSafeTrainingConfig(
    # ... other config ...

    secure_aggregation=SecureAggregationConfig(
        enabled=True,
        protocol="pairwise_masking",
        key_exchange="diffie_hellman",
    ),
)
```

## Configuration Reference

### TenSafeTrainingConfig

```python
@dataclass
class TenSafeTrainingConfig:
    # Model
    model_name: str                     # HuggingFace model ID
    tokenizer_name: Optional[str]       # Tokenizer (defaults to model)

    # LoRA
    lora_rank: int = 8                  # LoRA rank
    lora_alpha: float = 16.0            # LoRA alpha
    lora_dropout: float = 0.1           # LoRA dropout
    target_modules: List[str] = None    # Target modules

    # Training
    num_epochs: int = 3                 # Number of epochs
    per_device_batch_size: int = 8      # Batch size per worker
    learning_rate: float = 1e-4         # Learning rate
    weight_decay: float = 0.01          # Weight decay
    warmup_ratio: float = 0.1           # Warmup ratio
    max_seq_length: int = 512           # Max sequence length

    # Distributed
    num_workers: int = 4                # Number of Ray workers
    use_gpu: bool = True                # Use GPUs
    resources_per_worker: Dict = None   # Custom resource requirements

    # Checkpointing
    checkpoint_frequency: int = 100     # Steps between checkpoints
    checkpoint_dir: str = "/tmp/checkpoints"

    # Privacy
    dp_config: DistributedDPConfig = None
    secure_aggregation: SecureAggregationConfig = None
```

### DistributedDPConfig

```python
@dataclass
class DistributedDPConfig:
    # Privacy budget
    target_epsilon: float = 8.0         # Target epsilon
    target_delta: float = 1e-5          # Target delta

    # DP-SGD parameters
    max_grad_norm: float = 1.0          # Gradient clipping bound
    noise_multiplier: float = 1.1       # Noise multiplier

    # Accounting
    accountant_type: str = "rdp"        # "rdp", "prv", or "moments"

    # Early stopping
    stop_on_budget_exceeded: bool = True
```

### SecureAggregationConfig

```python
@dataclass
class SecureAggregationConfig:
    enabled: bool = False               # Enable secure aggregation
    protocol: str = "pairwise_masking"  # Protocol type
    key_exchange: str = "diffie_hellman" # Key exchange method
    verify_aggregation: bool = True      # Verify results
```

## Distributed DP-SGD

### How It Works

1. **Local Gradient Computation**: Each worker computes gradients on local batch
2. **Per-Sample Clipping**: Gradients clipped to bound sensitivity
3. **Secure Aggregation**: Gradients aggregated with masking protocol
4. **Noise Addition**: Calibrated Gaussian noise added to aggregate
5. **Privacy Accounting**: Global epsilon tracked across workers

```
Worker 1          Worker 2          Worker 3          Worker N
    │                │                │                │
    ▼                ▼                ▼                ▼
 Compute          Compute          Compute          Compute
 Gradients        Gradients        Gradients        Gradients
    │                │                │                │
    ▼                ▼                ▼                ▼
 Per-Sample       Per-Sample       Per-Sample       Per-Sample
 Clipping         Clipping         Clipping         Clipping
    │                │                │                │
    └────────────────┼────────────────┼────────────────┘
                     │
                     ▼
              ┌─────────────────┐
              │ Secure          │
              │ Aggregation     │
              │ (Pairwise Mask) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Add DP Noise    │
              │ N(0, σ²C²I)     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Update Model    │
              │ θ ← θ - η·g     │
              └─────────────────┘
```

### Noise Calibration

For distributed training with `N` workers:

```
σ_local = σ / √N

Aggregated noise: Σ σ_local² = N × (σ/√N)² = σ²
```

Each worker adds `σ/√N` noise, and after aggregation the total noise is `σ`.

## Secure Gradient Aggregation

### Pairwise Masking Protocol

The protocol ensures individual gradients are never revealed:

```python
# Conceptual implementation
def pairwise_masking(worker_id, gradient, all_workers):
    masked = gradient.clone()

    for other_id in all_workers:
        if other_id == worker_id:
            continue

        # Generate pairwise random mask
        seed = derive_seed(worker_id, other_id)
        mask = generate_mask(seed, gradient.shape)

        if worker_id < other_id:
            masked += mask  # Add mask
        else:
            masked -= mask  # Subtract same mask

    return masked

# After aggregation: Σ masks = 0 (they cancel out)
```

### Security Properties

- **Privacy**: No single party learns individual gradients
- **Correctness**: Aggregate equals sum of true gradients
- **Fault Tolerance**: Can handle worker failures (with reduced privacy)

## Multi-Node Training

### Cluster Setup

```python
# Start Ray cluster head
# ray start --head --port=6379

# Connect workers
import ray
ray.init(address="ray://head-node:6379")

# Configure multi-node training
config = TenSafeTrainingConfig(
    num_workers=16,  # Across multiple nodes
    resources_per_worker={
        "CPU": 4,
        "GPU": 1,
    },
)
```

### Kubernetes Ray Cluster

```yaml
# ray-cluster.yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: tensafe-training
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      spec:
        containers:
        - name: ray-head
          image: tensafe/ray-train:4.0.0
          resources:
            limits:
              nvidia.com/gpu: 1
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 4
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: tensafe/ray-train:4.0.0
          resources:
            limits:
              nvidia.com/gpu: 1
```

## Checkpointing

### Automatic Checkpointing

```python
config = TenSafeTrainingConfig(
    checkpoint_frequency=100,
    checkpoint_dir="s3://bucket/checkpoints",
)

trainer = TenSafeTrainer(config=config)

# Checkpoints saved automatically
result = trainer.fit(train_dataset)
```

### TSSP-Compatible Checkpoints

```python
from tensorguard.distributed import TSSPCheckpointCallback

callback = TSSPCheckpointCallback(
    output_dir="/path/to/tssp_checkpoints",
    sign_checkpoints=True,
    encrypt_weights=True,
)

trainer = TenSafeTrainer(
    config=config,
    callbacks=[callback],
)
```

### Resume Training

```python
trainer = TenSafeTrainer(config=config)
result = trainer.fit(
    train_dataset,
    resume_from_checkpoint="s3://bucket/checkpoints/step_1000",
)
```

## Privacy Budget Tracking

### Real-Time Monitoring

```python
from tensorguard.distributed import DistributedRDPAccountant

accountant = DistributedRDPAccountant(
    target_epsilon=8.0,
    target_delta=1e-5,
)

# During training
for epoch in range(num_epochs):
    for batch in train_loader:
        # Train step
        loss = trainer.train_step(batch)

        # Check privacy budget
        epsilon, delta = accountant.get_epsilon(target_delta=1e-5)
        print(f"Current epsilon: {epsilon:.2f}")

        if epsilon >= target_epsilon:
            print("Privacy budget exhausted!")
            break
```

### Budget Composition

For distributed training:

```
ε_total = √(Σ ε_i²)  (for RDP composition)
```

The `DistributedRDPAccountant` handles this automatically:

```python
# Accountant tracks across all workers
result = trainer.fit(train_dataset)
final_epsilon = result.metrics['dp_epsilon']
final_delta = result.metrics['dp_delta']
```

## Callbacks

### Custom Training Callback

```python
from tensorguard.distributed import TenSafeCallback

class MyCallback(TenSafeCallback):
    def on_train_begin(self, trainer, **kwargs):
        print("Training started!")

    def on_step(self, trainer, step, metrics, **kwargs):
        if step % 100 == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}")

    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        print(f"Epoch {epoch} complete: eval_loss={metrics['eval_loss']:.4f}")

    def on_train_end(self, trainer, **kwargs):
        print(f"Training complete! Final epsilon: {trainer.epsilon_spent:.2f}")

trainer = TenSafeTrainer(
    config=config,
    callbacks=[MyCallback()],
)
```

### MLOps Integration Callbacks

```python
from tensorguard.integrations import TenSafeWandbCallback, TenSafeMLflowCallback

trainer = TenSafeTrainer(
    config=config,
    callbacks=[
        TenSafeWandbCallback(project="tensafe-training"),
        TenSafeMLflowCallback(experiment_name="dp-lora-training"),
    ],
)
```

## Performance Tuning

### Scaling Workers

```python
# Linear scaling: batch_size × num_workers
config = TenSafeTrainingConfig(
    per_device_batch_size=8,
    num_workers=8,
    # Effective batch size: 64

    # Adjust learning rate with linear scaling
    learning_rate=1e-4 * 8,  # Scale with workers
)
```

### Gradient Accumulation

```python
config = TenSafeTrainingConfig(
    per_device_batch_size=2,
    gradient_accumulation_steps=4,
    # Effective per-device batch: 8
)
```

### Mixed Precision

```python
config = TenSafeTrainingConfig(
    mixed_precision="bf16",  # or "fp16"
    # Reduces memory, enables larger batches
)
```

## Troubleshooting

### Common Issues

**Workers Out of Memory**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce `per_device_batch_size` or enable gradient checkpointing

**Gradient Explosion**
```
RuntimeError: gradient overflow detected
```
Solution: Reduce `max_grad_norm` in DP config

**Privacy Budget Exceeded**
```
PrivacyBudgetExceeded: epsilon > target
```
Solution: Reduce `num_epochs` or increase `noise_multiplier`

**Worker Timeout**
```
RayTimeoutError: worker timed out
```
Solution: Increase `ray_timeout` or check network connectivity

### Debug Mode

```python
import logging
logging.getLogger("tensorguard.distributed").setLevel(logging.DEBUG)

config = TenSafeTrainingConfig(
    # ... config ...
)

trainer = TenSafeTrainer(config=config)
# Detailed logs for debugging
```

## Security Considerations

1. **Secure Channels**: Use TLS between Ray nodes
2. **Key Management**: Secure storage for aggregation keys
3. **Attestation**: Enable TPM attestation for workers
4. **Audit Logging**: Enable comprehensive training logs

```python
config = TenSafeTrainingConfig(
    secure_aggregation=SecureAggregationConfig(
        enabled=True,
        verify_aggregation=True,
    ),
    enable_audit_logging=True,
)
```

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [vllm-integration.md](vllm-integration.md) - Inference with trained adapters
- [observability.md](observability.md) - Monitoring setup
- [mlops.md](mlops.md) - Experiment tracking
