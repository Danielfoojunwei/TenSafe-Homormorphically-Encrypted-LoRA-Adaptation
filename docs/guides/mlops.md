# MLOps Integration Guide

**Version**: 4.1.0
**Last Updated**: 2026-02-08

This guide explains how to connect TenSafe to standard MLOps platforms like Weights & Biases (W&B), MLflow, and the HuggingFace Hub.

## Weights & Biases (W&B)

Use the `TenSafeWandbCallback` to automatically log privacy metrics alongside your model performance.

```python
from tensorguard.integrations import TenSafeWandbCallback

# Configure callback
wb_callback = TenSafeWandbCallback(
    project="tensafe-llama3",
    entity="my-team",
    log_privacy_report=True,  # Log (ε, δ) every 10 steps
    log_model=True           # Upload TSSP checkpoints to W&B Artifacts
)

# Pass to trainer
trainer = TenSafeTrainer(
    config=config,
    callbacks=[wb_callback]
)
```

### Key W&B Visualizations
- **Epsilon vs. Training Steps**: Monitor privacy consumption.
- **Privacy-Utility Tradeoff**: Scatter plot of Final Epsilon vs. Best Loss.
- **Gradient Norm Histograms**: Ensure DP clipping isn't destroying signal.

---

## MLflow

TenSafe's MLflow integration focuses on the **Model Registry** and **DP Certificates**.

### Registering Privacy-Preserving Models
When you log a model to MLflow, TenSafe attaches a **Cryptographic Privacy Certificate**.

```python
import mlflow
from tensorguard.integrations import TenSafeMLflowLogger

logger = TenSafeMLflowLogger()

with mlflow.start_run():
    # ... training ...
    logger.log_tensafe_model(
        model_name="privacy-llama-8b",
        adapter_path="final.tssp",
        epsilon=4.2
    )
```

---

## HuggingFace Hub Integration

TenSafe supports secure push/pull of HE-LoRA adapters to the HuggingFace Hub.

### Pushing a TSSP Package

```bash
python -m tensorguard.integrations.hf_hub \
    --upload-tssp my_model.tssp \
    --repo-id my-org/privacy-llama-adapter \
    --private True
```

- **Manifest Integrity**: Every push includes a `metadata.json` containing the TSSP security manifest.
- **Provenance Verification**: Downstream users of the adapter can verify its origin and intended privacy parameters via the hub's integration hooks.

---

## v4.1 Feature: Automated Compliance Reports

In version 4.1, you can export a **Privacy Audit Report** directly from your MLOps run.

```python
# Generates a PDF/Markdown summary of the run's privacy parameters
report_path = logger.generate_compliance_report(run_id="...")
```

This report is suitable for SOC2, HIPAA, and GDPR compliance audits.

## Related Documentation

- [Observability Guide](observability.md) - For low-level metrics and tracing.
- [Training Guide](training.md) - Setting up the training runs to be logged.
