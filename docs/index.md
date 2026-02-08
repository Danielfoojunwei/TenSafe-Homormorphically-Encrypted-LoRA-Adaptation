# TenSafe Documentation

**TenSafe** is a privacy-first ML training platform that provides encrypted artifacts, signed requests, immutable audit logs, differential privacy, and homomorphic encryption capabilities for secure model fine-tuning.

## What is TenSafe?

TenSafe enables organizations to fine-tune large language models while maintaining strong privacy guarantees:

- **Encrypted Artifacts** - All model checkpoints, gradients, and training data are encrypted at rest
- **Differential Privacy** - Built-in DP-SGD with configurable privacy budgets
- **Encrypted Artifacts** - All model checkpoints, gradients, and training data are encrypted at rest
- **Differential Privacy** - Built-in DP-SGD with configurable privacy budgets
- **Homomorphic Encryption** - N2HE integration with **Zero-Rotation (MOAI)** optimization
- **Evidence Fabric** - TEE-backed hardware attestation for secure model inference (NEW)
- **Immutable Audit Logs** - Hash-chained audit trail for compliance
- **LoRA Support** - Efficient fine-tuning with Low-Rank Adaptation
- **Pluggable Loss Functions** - Bring-your-own loss with stable contract (NEW)
- **RLVR Mode** - Reinforcement Learning with Verifiable Rewards (NEW)

## Key Features

### Privacy-Preserving Training

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig, DPConfig

# Configure differential privacy
dp_config = DPConfig(
    enabled=True,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    target_epsilon=8.0,
)

config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    dp_config=dp_config,
)

service = ServiceClient()
tc = service.create_training_client(config)
```

### Async Training Operations

```python
# Queue operations asynchronously
fb_future = tc.forward_backward(batch)
opt_future = tc.optim_step()

# Retrieve results when needed
result = fb_future.result()
print(f"Loss: {result.loss}")
```

### Encrypted Checkpoints

```python
# Save encrypted checkpoint
checkpoint = tc.save_state(
    include_optimizer=True,
    metadata={"epoch": 1, "notes": "After warmup"}
)
print(f"Saved: {checkpoint.artifact_id}")
print(f"Encrypted with: {checkpoint.encryption.algorithm}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TenSafe Platform                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   SDK        │  │   API        │  │   Compute    │       │
│  │   tg_tinker  │→ │   Gateway    │→ │   Backend    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                           │                  │               │
│                           ▼                  ▼               │
│                    ┌──────────────┐  ┌──────────────┐       │
│                    │   Audit      │  │   Artifact   │       │
│                    │   Log        │  │   Store      │       │
│                    └──────────────┘  └──────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Privacy Layer                        │       │
│  │  ┌────────┐  ┌────────┐  ┌────────┐             │       │
│  │  │  DP    │  │  N2HE  │  │  KEK/  │             │       │
│  │  │ Engine │  │  HE    │  │  DEK   │             │       │
│  │  └────────┘  └────────┘  └────────┘             │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

1. [How to Use TenSafe](HOW_TO_USE.md) - Comprehensive end-to-end usage guide
2. [Installation](installation.md) - Install the TenSafe SDK
3. [Quickstart](quickstart.md) - Run your first fine-tuning job
4. [Training Guide](guides/training.md) - Deep dive into training
5. [API Reference](api-reference/service-client.md) - Full API documentation

## Documentation

### Guides
- [Training](guides/training.md) - Training workflows and best practices
- [Sampling](guides/sampling.md) - Text generation and inference
- [Privacy](guides/privacy.md) - DP and homomorphic encryption
- [Custom Loss Functions](custom_loss_quickstart.md) - Pluggable loss guide (NEW)
- [RLVR Training](rlvr_quickstart.md) - Reinforcement Learning with Verifiable Rewards (NEW)

### API Reference
- [ServiceClient](api-reference/service-client.md) - Main entry point
- [TrainingClient](api-reference/training-client.md) - Training operations
- [FutureHandle](api-reference/futures.md) - Async operation handling
- [Configuration](api-reference/configuration.md) - Config options
- [Exceptions](api-reference/exceptions.md) - Error handling

### Cookbook
- [LoRA Fine-tuning](cookbook/lora-finetuning.md) - Complete LoRA tutorial with custom loss
- [Encrypted Inference](cookbook/encrypted-inference.md) - N2HE inference
- [Privacy Budget Management](cookbook/privacy-budget.md) - DP budgeting
- [RLVR Training](cookbook/rlvr-training.md) - Complete RLVR workflow (NEW)

## Support

- GitHub Issues: [TenSafe Repository](https://github.com/your-org/tensafe)
- Documentation: [https://docs.tensafe.dev](https://docs.tensafe.dev)
