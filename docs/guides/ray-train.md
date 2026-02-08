# Ray Train Distributed Training Guide

**Version**: 4.1.0
**Last Updated**: 2026-02-08

This guide covers using TenSafe with Ray Train for distributed, privacy-preserving training across multi-GPU and multi-node clusters.

## Overview

TenSafe's Ray Train integration provides:
- **Distributed DP-SGD**: Differential privacy coordinated across multiple workers.
- **Secure Gradient Aggregation**: Privacy-preserving gradient sharing via pairwise masking.
- **Evidence Fabric Integration**: TEE attestation for every distributed worker.
- **Unified Privacy Accounting**: Centralized (ε, δ) tracking via the Production RDP Accountant.

## Primary Configuration

```python
import ray
from tensorguard.distributed import TenSafeRayTrainer, TenSafeRayConfig

# Initialize Ray
ray.init()

# Configure distributed training
config = TenSafeRayConfig(
    num_workers=4,
    use_gpu=True,
    batch_size_per_worker=4,
    dp_config=DPConfig(
        enabled=True,
        target_epsilon=8.0,
        target_delta=1e-5
    ),
    secure_aggregation=True
)

# Create trainer
trainer = TenSafeRayTrainer(config=config, model_init_fn=my_model_fn)

# Fit model
result = trainer.train()
print(f"Final Epsilon: {result['metrics']['privacy_epsilon']}")
```

---

## Secure Gradient Aggregation

To prevent a central aggregator from seeing individual client updates, TenSafe employs a **Pairwise Masking** protocol. 

### How It Works:
1. Every worker generates a set of secret masks for every other worker.
2. Masks are applied to local gradients before transmission.
3. When summed at the aggregator, the masks cancel out (Σ masks = 0), leaving only the aggregated gradient.
4. Total privacy is maintained as long as not all workers collude.

---

## v4.1 Feature: Evidence Fabric for Workers

In version 4.1, the `TenSafeRayTrainer` requires all workers to provide an **Evidence Fabric Proof**.

1. **Attestation Collection**: Each worker generates a TEE quote at boot time.
2. **Verification Gate**: The Ray Head node (acting as the Evidence Fabric Orchestrator) verifies these quotes.
3. **Key Distribution**: Training keys (including the evaluation key for HE-LoRA) are only distributed to verified workers.

---

## Performance Tuning for v4.1

### Liger Kernels
Enable memory-optimized Triton kernels to reduce VRAM usage on large models:
```python
config = TenSafeRayConfig(
    num_workers=8,
    use_liger_kernels=True  # v4.1 optimization
)
```

### Checkpointing
Distributed checkpoints are saved as TSSP packages:
```python
config = TenSafeRayConfig(
    checkpoint_frequency=100,
    checkpoint_dir="s3://tensafe-vault/checkpoints"
)
```

## Related Documentation

- [Training Guide](training.md) - Core training primitives.
- [Observability Guide](observability.md) - Monitoring distributed runs.
- [MLOps Guide](mlops.md) - Logging to W&B and MLflow.
