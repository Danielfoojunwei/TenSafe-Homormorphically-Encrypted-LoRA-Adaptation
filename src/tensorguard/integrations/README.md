# TenSafe MLOps Integrations

This component provides native callbacks and integrations with external MLOps platforms, ensuring privacy metrics are first-class citizens in the experiment tracking workflow.

## Component Overview

- **`TenSafeWandbCallback`**: Comprehensive Weights & Biases integration for logging DP epsilon vs. Loss, gradient histograms, and artifact versioning.
- **`TenSafeMLflowCallback`**: MLflow integration focused on Model Registry and DP Certificate generation.
- **`TenSafeHFHubIntegration`**: Secure model sharing with the HuggingFace Hub, including TSSP-verified pushes.

## Key Features

1. **Privacy-Utility Tradeoff Logging**: Automatic generation of ε-vs-Loss plots in W&B and MLflow.
2. **DP Certificates**: Generation of cryptographic certificates representing the differential privacy guarantees achieved during a run.
3. **TSSP-Verified Hub Uploads**: ensures that every model pushed to a hub repository includes its security manifest and integrity hashes.

## Usage

```python
from tensorguard.integrations import TenSafeWandbCallback

trainer = TenSafeTrainer(
    config=config,
    callbacks=[TenSafeWandbCallback(project="privacy-project")]
)
```

## Dependencies

- `wandb`
- `mlflow`
- `huggingface_hub`
