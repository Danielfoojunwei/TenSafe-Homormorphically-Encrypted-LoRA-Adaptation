# How to Use TenSafe

A practical, end-to-end guide for getting started with TenSafe — the privacy-first ML training and serving platform.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Your First Training Job](#your-first-training-job)
5. [Adding Differential Privacy](#adding-differential-privacy)
6. [Homomorphic Encryption (HE-LoRA)](#homomorphic-encryption-he-lora)
7. [Text Generation / Inference](#text-generation--inference)
8. [Serving with vLLM](#serving-with-vllm)
9. [Distributed Training with Ray](#distributed-training-with-ray)
10. [Checkpoint Management](#checkpoint-management)
11. [Custom Loss Functions](#custom-loss-functions)
12. [RLVR Training](#rlvr-training)
13. [Running the Server](#running-the-server)
14. [Deploying to Kubernetes](#deploying-to-kubernetes)
15. [Running Tests and Benchmarks](#running-tests-and-benchmarks)
16. [Error Handling](#error-handling)
17. [Where to Go Next](#where-to-go-next)

---

## Overview

TenSafe enables organizations to fine-tune and serve large language models with strong privacy guarantees. It combines:

- **DP-SGD** — Differential privacy during training with configurable budgets
- **HE-LoRA** — Homomorphically encrypted LoRA adapters (CKKS scheme, zero-rotation MOAI optimization)
- **Encrypted Artifacts** — AES-256-GCM encryption for all checkpoints and model weights
- **Immutable Audit Logs** — Hash-chained records for compliance (GDPR, HIPAA, SOC 2)
- **Async Training** — Queue-based forward/backward/optimizer operations via `FutureHandle`

The system has three layers:

```
SDK (tg_tinker)  -->  Server (tensorguard / FastAPI)  -->  Core Services (DP, HE, Audit, Ray)
```

---

## Installation

### From PyPI

```bash
# Core SDK
pip install tg-tinker

# With vLLM serving support
pip install tg-tinker[vllm]

# With distributed Ray training
pip install tg-tinker[ray]

# With MLOps integrations (W&B, MLflow, HuggingFace Hub)
pip install tg-tinker[mlops]

# Everything
pip install tg-tinker[all]
```

### From Source (Development)

```bash
git clone https://github.com/your-org/tensafe.git
cd tensafe
pip install -e ".[dev,bench]"
```

### Verify Installation

```python
from tg_tinker import ServiceClient, __version__

print(f"TenSafe SDK version: {__version__}")
client = ServiceClient()
print("Connected successfully!")
```

**Requirements:** Python 3.9+

---

## Environment Setup

### Required

```bash
export TS_API_KEY="ts-your-api-key"
```

### Optional

```bash
export TS_BASE_URL="https://api.tensafe.dev"   # API endpoint
export TS_TENANT_ID="your-tenant-id"           # Multi-tenant isolation
export TS_TIMEOUT="300"                        # Request timeout (seconds)
export TS_RETRY_COUNT="3"                      # Auto-retry count
export TENSAFE_TOY_HE=1                        # Use toy HE simulation (testing only)
export TG_ENVIRONMENT="development"            # Environment mode
```

### Using a `.env` File

Create a `.env` file in your project root — the SDK loads it automatically:

```bash
TS_API_KEY=ts-your-api-key
TS_BASE_URL=https://api.tensafe.dev
TS_TENANT_ID=your-tenant-id
```

### Programmatic Configuration

```python
from tg_tinker import ServiceClient

client = ServiceClient(
    api_key="ts-your-api-key",
    base_url="https://api.tensafe.dev",
    timeout=600.0,
)
```

---

## Your First Training Job

A minimal LoRA fine-tuning loop in under 30 lines:

```python
from tg_tinker import (
    ServiceClient,
    TrainingConfig,
    LoRAConfig,
    OptimizerConfig,
)

# 1. Connect
service = ServiceClient()

# 2. Configure
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        learning_rate=1e-4,
        weight_decay=0.01,
    ),
    batch_size=8,
    gradient_accumulation_steps=4,
)

# 3. Create training client
tc = service.create_training_client(config)

# 4. Training loop
for batch in dataloader:
    fb_future = tc.forward_backward({
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    })
    opt_future = tc.optim_step()

    fb_result = fb_future.result()
    opt_result = opt_future.result()

    print(f"Step {tc.step}: loss={fb_result.loss:.4f}")

# 5. Save checkpoint (encrypted automatically)
checkpoint = tc.save_state()
print(f"Saved: {checkpoint.artifact_id}")
```

### How the Async Pattern Works

Operations return a `FutureHandle`. Call `.result()` to block until the result is ready, or queue multiple operations before waiting:

```python
# Queue both operations without blocking
fb_future = tc.forward_backward(batch)
opt_future = tc.optim_step()

# Now block and collect results
fb_result = fb_future.result()
opt_result = opt_future.result()
```

---

## Adding Differential Privacy

Add mathematically bounded privacy guarantees to any training run:

```python
from tg_tinker import DPConfig

config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    dp_config=DPConfig(
        enabled=True,
        noise_multiplier=1.0,    # Controls privacy/utility tradeoff
        max_grad_norm=1.0,       # Per-sample gradient clipping bound
        target_epsilon=8.0,      # Total privacy budget
        target_delta=1e-5,       # Failure probability
        accountant_type="rdp",   # Renyi Differential Privacy accounting
    ),
    batch_size=8,
    gradient_accumulation_steps=4,
)

tc = service.create_training_client(config)

for batch in dataloader:
    fb_future = tc.forward_backward(batch)
    opt_future = tc.optim_step(apply_dp_noise=True)  # DP noise applied here

    fb_result = fb_future.result()
    opt_result = opt_future.result()

    if opt_result.dp_metrics:
        print(f"Loss: {fb_result.loss:.4f}")
        print(f"Epsilon spent: {opt_result.dp_metrics.total_epsilon:.4f}")
        print(f"Noise applied: {opt_result.dp_metrics.noise_applied}")
```

### Choosing Epsilon

| Epsilon | Privacy Level | Typical Use Case |
|---------|---------------|------------------|
| 1-3     | Strong        | Medical/financial data |
| 3-8     | Moderate      | Most production workloads |
| 8-15    | Relaxed       | Less sensitive data |
| 15+     | Weak          | Research/experimentation |

### Budget Exhaustion Handling

```python
from tg_tinker import DPBudgetExceededError

try:
    for batch in dataloader:
        tc.forward_backward(batch).result()
        tc.optim_step(apply_dp_noise=True).result()
except DPBudgetExceededError as e:
    print(f"Privacy budget exhausted: {e}")
    tc.save_state(metadata={"reason": "dp_budget_exceeded"})
```

---

## Homomorphic Encryption (HE-LoRA)

HE-LoRA runs the base model in plaintext (fast) while keeping LoRA adapter deltas encrypted under CKKS homomorphic encryption (private). This is the "two-plane inference" approach.

### Key Management

```python
from tensorguard.n2he import HEKeyManager, HESchemeParams, N2HEScheme

key_manager = HEKeyManager(tenant_id="tenant-123")

params = HESchemeParams(
    scheme=N2HEScheme.CKKS,
    poly_modulus_degree=8192,
    security_level=128,
)

bundle = key_manager.generate_key_bundle(params=params)
```

### Encrypted LoRA Runtime

```python
from tensorguard.n2he import (
    EncryptedLoRARuntime,
    AdapterEncryptionConfig,
    create_encrypted_runtime,
)

runtime = create_encrypted_runtime(
    config=AdapterEncryptionConfig(
        rank=16,
        encrypted_layers=["q_proj", "v_proj"],
    ),
    key_bundle=bundle,
)

# Compute encrypted LoRA delta
import numpy as np
weights = np.random.randn(16, 768).astype(np.float32)
encrypted_delta = runtime.forward(weights)
```

### Private Inference

```python
from tensorguard.n2he import create_private_inference_mode

inference = create_private_inference_mode(key_bundle=bundle, params=params)

# Encrypt -> Compute -> Decrypt
encrypted_input = inference.encrypt_input(embedding)
encrypted_output = inference.process(encrypted_input)
output = inference.decrypt_output(encrypted_output)
```

---

## Text Generation / Inference

Sample text from a fine-tuned model:

```python
result = tc.sample(
    prompts=["Once upon a time", "The quick brown fox"],
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
)

for sample in result.samples:
    print(f"Prompt: {sample.prompt}")
    print(f"Completion: {sample.completion}\n")
```

---

## Serving with vLLM

TenSafe integrates with vLLM for high-throughput, OpenAI-compatible inference with encrypted LoRA adapters.

```bash
pip install tensafe[vllm]>=4.0.0
```

```python
from tensorguard.backends.vllm import TenSafeAsyncEngine, TenSafeVLLMConfig

config = TenSafeVLLMConfig(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    enable_he_lora=True,
    he_lora_adapter_path="/path/to/encrypted_lora.tssp",
    he_lora_rank=16,
    he_lora_alpha=32,
)

engine = TenSafeAsyncEngine(config)
await engine.initialize()

async for output in engine.generate("Explain privacy-preserving ML:", "req-001"):
    print(output.outputs[0].text, end="", flush=True)
```

See [vLLM Integration Guide](guides/vllm-integration.md) for OpenAI-compatible server setup, multi-LoRA, and benchmarks.

---

## Distributed Training with Ray

Scale training across multiple GPUs/nodes with secure gradient aggregation:

```bash
pip install tensafe[ray]>=4.0.0
```

```python
import ray
from tensorguard.distributed import TenSafeTrainer, TenSafeTrainingConfig, DistributedDPConfig

ray.init()

config = TenSafeTrainingConfig(
    model_name="meta-llama/Llama-3-8B",
    lora_rank=16,
    lora_alpha=32,
    num_epochs=3,
    per_device_batch_size=4,
    learning_rate=1e-4,
    num_workers=4,
    use_gpu=True,
    dp_config=DistributedDPConfig(
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
    ),
)

trainer = TenSafeTrainer(config)
result = trainer.fit()
```

See [Ray Train Guide](guides/ray-train.md) for fault tolerance, checkpointing, and multi-node setup.

---

## Checkpoint Management

All checkpoints are automatically encrypted with AES-256-GCM.

### Save

```python
checkpoint = tc.save_state(
    include_optimizer=True,
    metadata={
        "epoch": 1,
        "validation_loss": 0.42,
        "dataset_version": "v2",
    },
)

print(f"Artifact ID: {checkpoint.artifact_id}")
print(f"Encryption: {checkpoint.encryption.algorithm}")  # AES-256-GCM
```

### Load

```python
result = tc.load_state(artifact_id="art-xxx-yyy")
print(f"Restored to step: {result.step}")
```

### Download

```python
encrypted_data = service.pull_artifact(checkpoint.artifact_id)
# Data remains encrypted — decryption requires your tenant key
```

---

## Custom Loss Functions

TenSafe supports pluggable loss functions for specialized training scenarios.

### Built-in Losses

```python
from tensafe.training.losses import resolve_loss

loss_fn = resolve_loss("token_ce", ignore_index=-100)       # Language modeling
loss_fn = resolve_loss("margin_ranking", margin=0.5)         # Ranking
loss_fn = resolve_loss("contrastive", temperature=0.07)      # Contrastive
loss_fn = resolve_loss("mse")                                # Regression
```

### Register a Custom Loss

```python
from tensafe.training.losses import register_loss

@register_loss("focal_loss")
def focal_loss(outputs, batch, gamma=2.0, **kwargs):
    logits = outputs.logits
    labels = batch["labels"]
    ce_loss = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * ce_loss
    return {"loss": focal.mean(), "metrics": {"gamma": gamma}}

loss_fn = resolve_loss("focal_loss", gamma=2.5)
```

### YAML Configuration

```yaml
training:
  mode: sft

loss:
  type: token_ce
  kwargs:
    ignore_index: -100
    label_smoothing: 0.1
```

---

## RLVR Training

Reinforcement Learning with Verifiable Rewards for policy gradient optimization:

```python
from tensafe.rlvr import (
    MockRolloutSampler,
    REINFORCE,
    REINFORCEConfig,
    resolve_reward,
)

reward_fn = resolve_reward("keyword_contains", keywords=["solution", "answer"])

algo = REINFORCE(REINFORCEConfig(
    use_baseline=True,
    normalize_advantages=True,
    entropy_coef=0.01,
))

sampler = MockRolloutSampler(max_new_tokens=64)
prompts = ["Solve the equation x + 2 = 5", "What is 3 * 4?"]

for epoch in range(10):
    batch = sampler.generate_trajectories(prompts)
    for traj in batch:
        traj.reward = reward_fn(traj.prompt, traj.response)

    result = algo.update(batch, training_client)
    print(f"Epoch {epoch}: mean_reward={batch.mean_reward:.3f}")
```

PPO is also available for more stable training — see [RLVR Quickstart](rlvr_quickstart.md).

---

## Running the Server

### Development

```bash
make serve
# or
python -m uvicorn tensorguard.platform.main:app --reload --host 0.0.0.0 --port 8000
```

The API runs at `http://localhost:8000` with health checks at `/health`.

### Production (Docker)

```bash
docker build -t tensafe:4.0.0 .
docker run -e TS_API_KEY="..." -p 8000:8000 -p 9090:9090 tensafe:4.0.0
```

Port 8000 serves the API; port 9090 exposes Prometheus metrics.

---

## Deploying to Kubernetes

```bash
# Install the Helm chart
helm install tensafe ./deploy/helm/tensafe \
  --namespace tensafe \
  --create-namespace \
  --set tensafe.environment=production

# Verify
kubectl get pods -n tensafe
kubectl get svc -n tensafe
```

The Helm chart includes PostgreSQL, Redis, KEDA auto-scaling, and NVIDIA GPU scheduling. See [Kubernetes Guide](guides/kubernetes.md) for full production configuration.

---

## Running Tests and Benchmarks

### Tests

```bash
make test              # All 103 tests
make test-unit         # 76 unit tests
make test-integration  # 16 integration tests
make test-e2e          # 2 end-to-end tests
make test-cov          # With coverage report
make qa                # lint + typecheck + test
```

### Benchmarks

```bash
make bench             # Quick benchmarks
make bench-full        # Full benchmark suite
make bench-n2he        # HE-LoRA benchmarks
make bench-llama3      # Llama-3 specific benchmarks
```

### Compliance

```bash
make compliance-smoke  # Quick ISO 27701/27001 checks
make compliance        # Full compliance evidence pack
```

---

## Error Handling

TenSafe provides a structured exception hierarchy:

```python
from tg_tinker import (
    TGTinkerError,           # Base exception
    RateLimitedError,        # 429 rate limit
    DPBudgetExceededError,   # Privacy budget exhausted
    FutureTimeoutError,      # Operation timed out
)

try:
    fb_future = tc.forward_backward(batch)
    result = fb_future.result(timeout=60)

except FutureTimeoutError:
    print("Operation timed out")

except DPBudgetExceededError:
    tc.save_state()  # Save before stopping

except RateLimitedError as e:
    print(f"Retry after {e.retry_after}s")

except TGTinkerError as e:
    print(f"API error [{e.code}]: {e.message}")
```

---

## Where to Go Next

| Topic | Document |
|-------|----------|
| Installation details | [installation.md](installation.md) |
| Quick start walkthrough | [quickstart.md](quickstart.md) |
| System architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Training workflows | [guides/training.md](guides/training.md) |
| Privacy (DP + HE) | [guides/privacy.md](guides/privacy.md) |
| vLLM inference | [guides/vllm-integration.md](guides/vllm-integration.md) |
| Distributed training | [guides/ray-train.md](guides/ray-train.md) |
| Kubernetes deployment | [guides/kubernetes.md](guides/kubernetes.md) |
| Observability | [guides/observability.md](guides/observability.md) |
| MLOps integrations | [guides/mlops.md](guides/mlops.md) |
| LoRA fine-tuning cookbook | [cookbook/lora-finetuning.md](cookbook/lora-finetuning.md) |
| Encrypted inference cookbook | [cookbook/encrypted-inference.md](cookbook/encrypted-inference.md) |
| RLVR training cookbook | [cookbook/rlvr-training.md](cookbook/rlvr-training.md) |
| API: ServiceClient | [api-reference/service-client.md](api-reference/service-client.md) |
| API: TrainingClient | [api-reference/training-client.md](api-reference/training-client.md) |
| API: FutureHandle | [api-reference/futures.md](api-reference/futures.md) |
| API: Configuration | [api-reference/configuration.md](api-reference/configuration.md) |
| API: Exceptions | [api-reference/exceptions.md](api-reference/exceptions.md) |
| Security policy | [../SECURITY.md](../SECURITY.md) |
| Production readiness | [../PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md) |

### Examples

Run the included examples to see each feature in action:

```bash
python examples/quickstart/01_hello_tensafe.py       # Hello world
python examples/quickstart/02_basic_training.py       # Basic training loop
python examples/quickstart/04_dp_training.py          # Differential privacy
python examples/training/lora_finetuning.py           # LoRA fine-tuning
python examples/training/distributed_training.py      # Multi-GPU training
python examples/inference/encrypted_inference.py      # HE-LoRA inference
python examples/privacy/he_lora_setup.py              # HE key setup
python examples/integrations/wandb_tracking.py        # W&B tracking
python examples/integrations/ray_distributed.py       # Ray distributed
```
