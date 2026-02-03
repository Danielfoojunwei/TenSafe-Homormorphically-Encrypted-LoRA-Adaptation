# TenSafe Examples

This directory contains comprehensive examples demonstrating how to use TenSafe's privacy-preserving ML platform.

## Quick Start

```bash
# Install TenSafe
pip install tensafe

# Set your API key
export TENSAFE_API_KEY="your-api-key"

# Run your first example
python quickstart/01_hello_tensafe.py
```

## Example Categories

### 📚 Quickstart (`quickstart/`)

Get started with TenSafe in minutes:

| Example | Description |
|---------|-------------|
| `01_hello_tensafe.py` | Verify your setup and API connection |
| `02_basic_training.py` | Train a simple LoRA adapter |
| `03_inference.py` | Generate text with your model |
| `04_dp_training.py` | Training with differential privacy |
| `05_encrypted_inference.py` | HE-LoRA encrypted inference |

### 🎓 Training (`training/`)

In-depth training examples:

| Example | Description |
|---------|-------------|
| `lora_finetuning.py` | Complete LoRA fine-tuning workflow |
| `qlora_training.py` | Memory-efficient QLoRA training |
| `distributed_training.py` | Multi-node training with Ray |
| `mixed_precision.py` | FP16/BF16 mixed precision |
| `gradient_accumulation.py` | Large batch training |
| `checkpoint_resume.py` | Save and resume training |
| `hyperparameter_tuning.py` | Hyperparameter optimization |
| `custom_loss.py` | Custom loss functions |
| `data_loading.py` | Efficient data loading |
| `evaluation.py` | Model evaluation metrics |

### 🔮 Inference (`inference/`)

Model inference patterns:

| Example | Description |
|---------|-------------|
| `basic_inference.py` | Simple text generation |
| `streaming_inference.py` | Token-by-token streaming |
| `batch_inference.py` | Process multiple prompts |
| `encrypted_inference.py` | HE-LoRA private inference |
| `multi_adapter.py` | Switch between LoRA adapters |
| `temperature_sampling.py` | Sampling parameters |
| `chat_completion.py` | Chat-style completions |
| `structured_output.py` | JSON/structured outputs |

### 🔒 Privacy (`privacy/`)

Privacy-preserving techniques:

| Example | Description |
|---------|-------------|
| `dp_training_basic.py` | Differential privacy basics |
| `dp_budget_tracking.py` | Privacy budget monitoring |
| `dp_hyperparameters.py` | DP hyperparameter tuning |
| `he_lora_setup.py` | HE-LoRA configuration |
| `tgsp_packaging.py` | Create TGSP packages |
| `tgsp_verification.py` | Verify TGSP signatures |
| `pqc_encryption.py` | Post-quantum cryptography |
| `audit_compliance.py` | Audit trails and compliance |

### 🔌 Integrations (`integrations/`)

Third-party integrations:

| Example | Description |
|---------|-------------|
| `wandb_tracking.py` | Weights & Biases logging |
| `mlflow_registry.py` | MLflow model registry |
| `huggingface_hub.py` | HuggingFace Hub integration |
| `ray_distributed.py` | Ray Train distributed training |
| `kubernetes_deploy.py` | Kubernetes deployment |
| `vllm_serving.py` | vLLM serving setup |
| `prometheus_metrics.py` | Prometheus metrics |
| `webhook_events.py` | Webhook event handling |

### 🚀 Advanced (`advanced/`)

Advanced techniques:

| Example | Description |
|---------|-------------|
| `custom_optimizer.py` | Custom optimizer implementation |
| `gradient_checkpointing.py` | Memory-efficient training |
| `multi_gpu_training.py` | Multi-GPU data parallel |
| `fsdp_training.py` | Fully Sharded Data Parallel |
| `continual_learning.py` | Continual/incremental learning |
| `model_merging.py` | LoRA adapter merging |
| `quantization.py` | Model quantization |
| `knowledge_distillation.py` | Knowledge distillation |
| `federated_learning.py` | Federated learning |
| `secure_aggregation.py` | Secure gradient aggregation |

## Prerequisites

### Required
- Python 3.9+
- TenSafe account and API key
- `pip install tensafe`

### Optional (for specific examples)
- `pip install tensafe[distributed]` - For Ray distributed training
- `pip install tensafe[mlops]` - For W&B, MLflow integrations
- `pip install tensafe[cuda]` - For GPU acceleration

## Running Examples

### Basic Usage

```bash
# Set API key
export TENSAFE_API_KEY="ts_..."

# Run an example
python examples/quickstart/01_hello_tensafe.py
```

### With Custom Configuration

```bash
# Use staging environment
export TENSAFE_BASE_URL="https://api.staging.tensafe.io"

# Enable debug logging
export TENSAFE_DEBUG=1

python examples/training/lora_finetuning.py
```

### In Jupyter Notebook

```python
import os
os.environ["TENSAFE_API_KEY"] = "ts_..."

# Then run any example code
from tensafe import TenSafeClient
client = TenSafeClient()
```

## Example Structure

Each example follows this structure:

```python
"""
Example Title

Brief description of what this example demonstrates.

Requirements:
- Any specific requirements

Usage:
    python example_name.py [options]
"""

import tensafe
from tensafe import TenSafeClient

def main():
    # Example code here
    pass

if __name__ == "__main__":
    main()
```

## Contributing Examples

We welcome example contributions! Please follow these guidelines:

1. **Clear docstring** explaining what the example demonstrates
2. **Self-contained** - should run with minimal setup
3. **Well-commented** - explain key concepts
4. **Error handling** - handle common errors gracefully
5. **Tested** - verify the example works

### Submission Process

1. Fork the repository
2. Add your example in the appropriate category
3. Update this README with your example
4. Submit a pull request

## Troubleshooting

### Common Issues

**API Key not set:**
```
Error: TENSAFE_API_KEY environment variable not set
```
Solution: `export TENSAFE_API_KEY="your-key"`

**Rate limited:**
```
Error: Rate limit exceeded (429)
```
Solution: Wait and retry, or upgrade your plan

**Model not found:**
```
Error: Model 'xxx' not found
```
Solution: Check model name spelling and availability

### Getting Help

- **Documentation:** https://docs.tensafe.io
- **Discord:** https://discord.gg/tensafe
- **GitHub Issues:** https://github.com/tensafe/tensafe/issues
- **Email:** support@tensafe.io

## License

These examples are provided under the Apache 2.0 license. See [LICENSE](../LICENSE) for details.
