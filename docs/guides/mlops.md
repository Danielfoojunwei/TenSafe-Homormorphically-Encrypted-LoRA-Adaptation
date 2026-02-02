# MLOps Integration Guide

**Version**: 4.0.0
**Last Updated**: 2026-02-02

This guide covers integrating TenSafe with MLOps platforms for experiment tracking, model registry, and collaboration.

## Overview

TenSafe provides native integrations with:
- **Weights & Biases**: Experiment tracking with privacy metrics
- **MLflow**: Model registry with DP certificates
- **HuggingFace Hub**: TSSP-verified model sharing

## Prerequisites

```bash
# Install TenSafe with MLOps support
pip install tensafe[mlops]>=4.0.0

# Or install dependencies separately
pip install wandb>=0.15.0
pip install mlflow>=2.0.0
pip install huggingface_hub>=0.16.0
```

## Weights & Biases Integration

### Quick Start

```python
from tensorguard.integrations import TenSafeWandbCallback

# Create callback
wandb_callback = TenSafeWandbCallback(
    project="tensafe-training",
    name="llama3-lora-dp",
    config={
        "model": "meta-llama/Llama-3-8B",
        "lora_rank": 16,
        "target_epsilon": 8.0,
    },
)

# Use with trainer
from tensorguard.distributed import TenSafeTrainer

trainer = TenSafeTrainer(
    config=training_config,
    callbacks=[wandb_callback],
)

result = trainer.fit(train_dataset)
```

### Configuration

```python
@dataclass
class TenSafeWandbConfig:
    # W&B settings
    project: str                        # W&B project name
    entity: Optional[str] = None        # W&B entity (team/user)
    name: Optional[str] = None          # Run name
    tags: List[str] = None              # Run tags
    notes: Optional[str] = None         # Run notes

    # Logging settings
    log_frequency: int = 10             # Log every N steps
    log_model: bool = True              # Log model artifacts
    log_gradients: bool = True          # Log gradient histograms
    log_privacy_metrics: bool = True    # Log DP metrics

    # Privacy settings
    redact_config_secrets: bool = True  # Redact secrets from config
```

### Logged Metrics

The callback automatically logs:

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss |
| `train/learning_rate` | Learning rate |
| `train/gradient_norm` | Gradient norm after clipping |
| `privacy/epsilon_spent` | Cumulative epsilon |
| `privacy/delta` | Privacy delta |
| `privacy/noise_multiplier` | Noise multiplier |
| `privacy/steps_remaining` | Steps until budget exhausted |
| `eval/loss` | Evaluation loss |
| `eval/perplexity` | Evaluation perplexity |

### Custom Logging

```python
class CustomWandbCallback(TenSafeWandbCallback):
    def on_step(self, trainer, step, metrics, **kwargs):
        # Call parent
        super().on_step(trainer, step, metrics, **kwargs)

        # Add custom metrics
        if step % 100 == 0:
            import wandb
            wandb.log({
                "custom/my_metric": compute_my_metric(),
                "custom/privacy_efficiency": metrics['loss'] / metrics['epsilon_spent'],
            })

    def on_train_end(self, trainer, **kwargs):
        import wandb

        # Log final privacy report
        privacy_report = trainer.get_privacy_report()
        wandb.log({"privacy_report": wandb.Table(dataframe=privacy_report)})

        # Log model artifact
        artifact = wandb.Artifact("model", type="tssp-package")
        artifact.add_file(trainer.checkpoint_path)
        wandb.log_artifact(artifact)

        super().on_train_end(trainer, **kwargs)
```

### Privacy Dashboard

```python
# Create custom W&B dashboard for privacy monitoring
import wandb

# Privacy budget gauge
wandb.log({
    "privacy/budget_gauge": wandb.plot.bar(
        wandb.Table(
            data=[["Spent", epsilon_spent], ["Remaining", target_epsilon - epsilon_spent]],
            columns=["Category", "Epsilon"]
        ),
        "Category",
        "Epsilon",
        title="Privacy Budget"
    )
})

# Epsilon vs Loss tradeoff
wandb.log({
    "privacy/epsilon_loss_tradeoff": wandb.plot.scatter(
        wandb.Table(
            data=list(zip(epsilons, losses)),
            columns=["Epsilon", "Loss"]
        ),
        "Epsilon",
        "Loss",
        title="Privacy-Utility Tradeoff"
    )
})
```

## MLflow Integration

### Quick Start

```python
from tensorguard.integrations import TenSafeMLflowCallback

# Create callback
mlflow_callback = TenSafeMLflowCallback(
    experiment_name="tensafe-dp-training",
    tracking_uri="http://mlflow:5000",
    artifact_location="s3://bucket/mlflow",
)

# Use with trainer
trainer = TenSafeTrainer(
    config=training_config,
    callbacks=[mlflow_callback],
)

result = trainer.fit(train_dataset)
```

### Configuration

```python
@dataclass
class TenSafeMLflowConfig:
    # MLflow settings
    experiment_name: str                # Experiment name
    tracking_uri: Optional[str] = None  # MLflow tracking server
    artifact_location: Optional[str] = None  # Artifact storage

    # Run settings
    run_name: Optional[str] = None      # Run name
    tags: Dict[str, str] = None         # Run tags

    # Logging
    log_frequency: int = 10             # Log every N steps
    log_model: bool = True              # Register model
    log_artifacts: bool = True          # Log artifacts

    # Privacy
    log_dp_certificate: bool = True     # Log DP certificate
```

### Logged Artifacts

| Artifact | Description |
|----------|-------------|
| `model/` | TSSP model package |
| `dp_certificate.json` | Privacy guarantee certificate |
| `training_config.json` | Training configuration |
| `privacy_report.json` | Detailed privacy report |

### Model Registry

```python
from tensorguard.integrations import TenSafeMLflowCallback
import mlflow

# Register model with TSSP verification
mlflow_callback = TenSafeMLflowCallback(
    experiment_name="tensafe-training",
    log_model=True,
)

# After training
model_uri = f"runs:/{run_id}/model"

# Register with tags
mlflow.register_model(
    model_uri,
    name="llama3-lora-dp",
    tags={
        "epsilon": str(final_epsilon),
        "delta": str(final_delta),
        "tssp_verified": "true",
    },
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="llama3-lora-dp",
    version=1,
    stage="Production",
)
```

### DP Certificate

```python
# Automatic DP certificate logging
{
    "model_name": "llama3-lora-dp",
    "training_timestamp": "2026-02-02T12:00:00Z",
    "privacy_guarantee": {
        "epsilon": 7.5,
        "delta": 1e-5,
        "mechanism": "DP-SGD",
        "accountant": "RDP"
    },
    "training_parameters": {
        "noise_multiplier": 1.1,
        "max_grad_norm": 1.0,
        "batch_size": 32,
        "epochs": 3,
        "total_steps": 1000
    },
    "data_summary": {
        "num_samples": 10000,
        "sample_rate": 0.032
    },
    "signature": "..."  # Cryptographic signature
}
```

### Custom MLflow Logging

```python
class CustomMLflowCallback(TenSafeMLflowCallback):
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        import mlflow

        # Log epoch-level metrics
        mlflow.log_metrics({
            f"epoch_{epoch}/eval_loss": metrics['eval_loss'],
            f"epoch_{epoch}/epsilon": metrics['epsilon_spent'],
        })

        # Log model checkpoint as artifact
        if epoch % 5 == 0:
            mlflow.log_artifact(
                trainer.checkpoint_path,
                artifact_path=f"checkpoints/epoch_{epoch}"
            )

        super().on_epoch_end(trainer, epoch, metrics, **kwargs)
```

## HuggingFace Hub Integration

### Quick Start

```python
from tensorguard.integrations import TenSafeHFHubIntegration, TenSafeHFHubConfig

# Create integration
hub = TenSafeHFHubIntegration(
    config=TenSafeHFHubConfig(
        token="hf_xxx",
        private=True,
        require_tssp_verification=True,
    )
)

# Push model to Hub
url = hub.push_to_hub(
    tssp_package_path="/path/to/model.tssp",
    repo_id="username/llama3-lora-dp",
    privacy_info={
        "epsilon": 7.5,
        "delta": 1e-5,
        "noise_multiplier": 1.1,
    },
)

print(f"Model available at: {url}")
```

### Configuration

```python
@dataclass
class TenSafeHFHubConfig:
    # Authentication
    token: Optional[str] = None         # HF API token

    # Repository
    private: bool = True                # Private repository
    repo_type: str = "model"            # Repository type

    # TSSP
    require_tssp_verification: bool = True  # Verify before push
    include_encrypted_weights: bool = False # Include weights

    # Model card
    generate_model_card: bool = True    # Auto-generate README
    license: str = "apache-2.0"         # License
    language: str = "en"                # Language
```

### Push to Hub

```python
# Push with full privacy information
url = hub.push_to_hub(
    tssp_package_path="/path/to/model.tssp",
    repo_id="username/llama3-lora-dp",
    commit_message="Upload DP-trained LoRA adapter",
    privacy_info={
        "epsilon": 7.5,
        "delta": 1e-5,
        "noise_multiplier": 1.1,
        "max_grad_norm": 1.0,
        "mechanism": "DP-SGD",
        "accountant": "RDP",
    },
)
```

### Pull from Hub

```python
# Pull and verify model
manifest = hub.pull_from_hub(
    repo_id="username/llama3-lora-dp",
    local_dir="/path/to/download",
    verify=True,
)

print(f"Model verified: {manifest is not None}")
print(f"Privacy guarantee: ε={manifest['privacy']['epsilon']}")
```

### Auto-Generated Model Card

```markdown
---
language: en
license: apache-2.0
library_name: tensafe
tags:
  - tensafe
  - privacy-preserving
  - homomorphic-encryption
  - lora
  - differential-privacy
---

# llama3-lora-dp

This model was trained using **TenSafe**, a privacy-preserving ML platform.

## Model Details

| Property | Value |
|----------|-------|
| Base Model | meta-llama/Llama-3-8B |
| Training Method | LoRA with Differential Privacy |
| TSSP Package ID | `pkg_abc123` |
| Framework | TenSafe v4.0.0 |

## Privacy Information

This model was trained with differential privacy guarantees:

| Parameter | Value |
|-----------|-------|
| Epsilon (ε) | 7.5 |
| Delta (δ) | 1e-5 |
| Noise Multiplier | 1.1 |
| Max Gradient Norm | 1.0 |

## Security Features

This model is distributed as a **TSSP (TenSafe Secure Package)** with:

- **Cryptographic Signatures**: Ed25519 + Dilithium3 (post-quantum)
- **Encrypted Weights**: AES-256-GCM encryption
- **Tamper-Evident Manifest**: SHA-256 hash verification

## Usage

```python
from tensafe import load_model

# Load and verify model
model = load_model("pkg_abc123")

# Generate text
output = model.generate("Hello, world!")
```
```

### List TenSafe Models

```python
# Find TenSafe models
models = hub.list_models(author="username")

for model in models:
    print(f"{model['id']}: {model['downloads']} downloads")
```

## Combined Pipeline

### Training with All Integrations

```python
from tensorguard.distributed import TenSafeTrainer, TenSafeTrainingConfig
from tensorguard.integrations import (
    TenSafeWandbCallback,
    TenSafeMLflowCallback,
    TenSafeHFHubIntegration,
)

# Configure training
config = TenSafeTrainingConfig(
    model_name="meta-llama/Llama-3-8B",
    lora_rank=16,
    num_epochs=3,
    dp_config=DistributedDPConfig(target_epsilon=8.0),
)

# Create callbacks
callbacks = [
    TenSafeWandbCallback(
        project="tensafe-production",
        log_privacy_metrics=True,
    ),
    TenSafeMLflowCallback(
        experiment_name="tensafe-production",
        log_dp_certificate=True,
    ),
]

# Train
trainer = TenSafeTrainer(config=config, callbacks=callbacks)
result = trainer.fit(train_dataset)

# Push to HuggingFace Hub
hub = TenSafeHFHubIntegration()
url = hub.push_to_hub(
    tssp_package_path=result.checkpoint_path,
    repo_id="company/llama3-lora-dp-v1",
    privacy_info={
        "epsilon": result.metrics['dp_epsilon'],
        "delta": result.metrics['dp_delta'],
    },
)

print(f"Training complete!")
print(f"W&B Run: {wandb.run.url}")
print(f"MLflow Run: {mlflow.active_run().info.run_id}")
print(f"HF Hub: {url}")
```

### CI/CD Integration

```yaml
# .github/workflows/train.yml
name: Train Model

on:
  push:
    branches: [main]
    paths: ['configs/**']

jobs:
  train:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install tensafe[all]

      - name: Train model
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python scripts/train.py \
            --config configs/production.yaml \
            --wandb-project tensafe-ci \
            --mlflow-experiment tensafe-ci \
            --push-to-hub company/model-${{ github.sha }}

      - name: Verify privacy guarantee
        run: |
          python scripts/verify_privacy.py \
            --model company/model-${{ github.sha }} \
            --max-epsilon 10.0
```

## Security Considerations

### API Key Management

```python
# Use environment variables
import os

wandb_callback = TenSafeWandbCallback(
    project="tensafe",
    # Key from WANDB_API_KEY env var
)

mlflow_callback = TenSafeMLflowCallback(
    tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
)

hub = TenSafeHFHubIntegration(
    config=TenSafeHFHubConfig(
        token=os.environ.get("HF_TOKEN"),
    )
)
```

### Config Redaction

```python
# Automatically redact secrets from logged configs
wandb_callback = TenSafeWandbCallback(
    project="tensafe",
    redact_config_secrets=True,  # Default: True
)

# Config logged without:
# - API keys
# - Tokens
# - Passwords
# - Database URLs with credentials
```

### Private Repositories

```python
# Always use private repos for sensitive models
hub = TenSafeHFHubIntegration(
    config=TenSafeHFHubConfig(
        private=True,  # Default: True
        require_tssp_verification=True,
    )
)
```

## Troubleshooting

### W&B Connection Issues

```python
import wandb

# Check connection
wandb.login()
wandb.api.default_entity

# Offline mode fallback
os.environ["WANDB_MODE"] = "offline"
```

### MLflow Server Issues

```bash
# Check MLflow server
curl http://mlflow:5000/health

# Debug logging
export MLFLOW_TRACKING_INSECURE_TLS=true
export MLFLOW_VERBOSE=true
```

### HuggingFace Auth Issues

```python
from huggingface_hub import whoami

# Verify authentication
try:
    info = whoami()
    print(f"Logged in as: {info['name']}")
except Exception as e:
    print(f"Auth failed: {e}")
```

## Related Documentation

- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture
- [ray-train.md](ray-train.md) - Distributed training
- [observability.md](observability.md) - Monitoring setup
