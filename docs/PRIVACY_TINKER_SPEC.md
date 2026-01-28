# Privacy Tinker Specification

**Version**: 1.0.0
**Status**: Draft
**Last Updated**: 2026-01-28

## Overview

Privacy Tinker (TG-Tinker) is a privacy-first alternative to Thinking Machines' Tinker API. It provides a Python SDK and CLI for fine-tuning ML models with strong privacy guarantees, including encrypted artifacts, signed requests, immutable audit logs, and optional differential privacy.

This specification defines the public SDK surface, server endpoints, privacy model, and threat model.

---

## Table of Contents

1. [SDK Surface](#1-sdk-surface)
2. [Endpoint Specifications](#2-endpoint-specifications)
3. [Privacy Model](#3-privacy-model)
4. [Threat Model](#4-threat-model)
5. [Configuration](#5-configuration)
6. [Error Handling](#6-error-handling)

---

## 1. SDK Surface

### 1.1 ServiceClient

The `ServiceClient` is the primary entry point for interacting with the TG-Tinker API.

```python
from tg_tinker import ServiceClient

# Initialize with explicit configuration
client = ServiceClient(
    base_url="https://api.tensorguard.io",
    api_key="tg-key-xxxx",
    tenant_id="tenant-uuid"  # Optional, derived from API key if not provided
)

# Or use environment variables
client = ServiceClient()  # Uses TG_TINKER_BASE_URL, TG_TINKER_API_KEY
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `create_training_client(config)` | Create a new training client | `TrainingClient` |
| `get_training_client(client_id)` | Retrieve existing training client | `TrainingClient` |
| `list_training_clients()` | List all training clients for tenant | `List[TrainingClientInfo]` |
| `get_future(future_id)` | Retrieve a future by ID | `FutureHandle` |
| `get_audit_logs(client_id, ...)` | Retrieve audit logs | `List[AuditLogEntry]` |

### 1.2 TrainingClient

The `TrainingClient` exposes primitives for training loop control.

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig

service = ServiceClient()
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"]),
    optimizer="adamw",
    learning_rate=1e-4,
    dp_config=None  # Optional DPConfig for differential privacy
)

training_client = service.create_training_client(config)
```

#### Primitives

| Primitive | Description | Returns | Async |
|-----------|-------------|---------|-------|
| `forward_backward(batch)` | Compute forward pass and gradients | `FutureHandle` | Yes |
| `optim_step()` | Apply optimizer update | `FutureHandle` | Yes |
| `sample(prompt, **kwargs)` | Generate samples from current model | `SampleResult` | No |
| `save_state(path)` | Save encrypted checkpoint | `SaveStateResult` | No |
| `load_state(artifact_id)` | Load checkpoint from artifact | `LoadStateResult` | No |

#### Async Execution Pattern

```python
# Primitives return futures immediately - server queues jobs
future_fb = training_client.forward_backward(batch)
future_opt = training_client.optim_step()  # Can be called before waiting

# Wait for results
fb_result = future_fb.result(timeout=300)  # Blocks until complete
opt_result = future_opt.result()

# Or check status
print(future_fb.status())  # "pending" | "running" | "completed" | "failed"
```

### 1.3 FutureHandle

`FutureHandle` represents an asynchronous operation.

```python
class FutureHandle:
    @property
    def future_id(self) -> str:
        """Unique identifier for this future."""

    def status(self) -> FutureStatus:
        """Get current status of the future."""

    def result(self, timeout: Optional[float] = None) -> Any:
        """Block until result is available or timeout."""

    def cancel(self) -> bool:
        """Attempt to cancel the future. Returns True if cancelled."""

    def done(self) -> bool:
        """Return True if future is complete (success or failure)."""

    def exception(self) -> Optional[Exception]:
        """Return exception if future failed, None otherwise."""
```

#### FutureStatus Enum

```python
class FutureStatus(str, Enum):
    PENDING = "pending"      # Queued, not yet started
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"        # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user
```

### 1.4 Configuration Classes

```python
from tg_tinker import (
    TrainingConfig,
    LoRAConfig,
    OptimizerConfig,
    DPConfig,
    SamplingConfig
)

# LoRA Configuration
lora_config = LoRAConfig(
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"],
    bias: str = "none"  # "none" | "all" | "lora_only"
)

# Optimizer Configuration
optim_config = OptimizerConfig(
    name: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
)

# Differential Privacy Configuration
dp_config = DPConfig(
    enabled: bool = True,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,  # Per-batch gradient clipping
    target_epsilon: Optional[float] = 8.0,
    target_delta: Optional[float] = 1e-5,
    accountant_type: str = "rdp"  # "rdp" | "moments" | "prv"
)

# Full Training Configuration
training_config = TrainingConfig(
    model_ref: str,
    lora_config: Optional[LoRAConfig] = None,
    optimizer: OptimizerConfig = OptimizerConfig(),
    dp_config: Optional[DPConfig] = None,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    max_steps: Optional[int] = None,
    metadata: Dict[str, Any] = {}
)
```

---

## 2. Endpoint Specifications

### 2.1 Base URL and Versioning

```
Base URL: https://api.tensorguard.io/v1
API Version: v1
Content-Type: application/json
```

### 2.2 Authentication

All requests must include an API key in the `Authorization` header:

```
Authorization: Bearer tg-key-xxxx
```

API keys are scoped to tenants and have configurable permissions.

### 2.3 Endpoints

#### POST /v1/training_clients

Create a new training client.

**Request:**
```json
{
  "model_ref": "meta-llama/Llama-3-8B",
  "lora_config": {
    "rank": 16,
    "alpha": 32.0,
    "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none"
  },
  "optimizer": {
    "name": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 0.01
  },
  "dp_config": null,
  "batch_size": 8,
  "metadata": {}
}
```

**Response (201 Created):**
```json
{
  "training_client_id": "tc-uuid-xxxx",
  "tenant_id": "tenant-uuid",
  "model_ref": "meta-llama/Llama-3-8B",
  "status": "ready",
  "step": 0,
  "created_at": "2026-01-28T10:00:00Z",
  "config": { ... }
}
```

#### POST /v1/training_clients/{id}/forward_backward

Queue a forward-backward pass computation.

**Request:**
```json
{
  "batch": {
    "input_ids": [[...], [...], ...],
    "attention_mask": [[...], [...], ...],
    "labels": [[...], [...], ...]
  },
  "batch_hash": "sha256:xxxx"  // Optional client-side hash for verification
}
```

**Response (202 Accepted):**
```json
{
  "future_id": "fut-uuid-xxxx",
  "status": "pending",
  "created_at": "2026-01-28T10:00:01Z",
  "training_client_id": "tc-uuid-xxxx"
}
```

#### POST /v1/training_clients/{id}/optim_step

Queue an optimizer step.

**Request:**
```json
{
  "apply_dp_noise": true  // If DP enabled, apply noise before update
}
```

**Response (202 Accepted):**
```json
{
  "future_id": "fut-uuid-xxxx",
  "status": "pending",
  "created_at": "2026-01-28T10:00:02Z",
  "training_client_id": "tc-uuid-xxxx",
  "dp_metrics": {  // Only if DP enabled
    "noise_applied": true,
    "epsilon_spent": 0.1,
    "total_epsilon": 0.1
  }
}
```

#### POST /v1/training_clients/{id}/sample

Generate samples from the current model state.

**Request:**
```json
{
  "prompts": ["Once upon a time"],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

**Response (200 OK):**
```json
{
  "samples": [
    {
      "prompt": "Once upon a time",
      "completion": "...",
      "tokens_generated": 64,
      "finish_reason": "stop"
    }
  ],
  "model_step": 100,
  "sampling_config": { ... }
}
```

#### POST /v1/training_clients/{id}/save_state

Save current training state as an encrypted artifact.

**Request:**
```json
{
  "include_optimizer": true,
  "metadata": {
    "checkpoint_name": "step-1000",
    "notes": "After first epoch"
  }
}
```

**Response (200 OK):**
```json
{
  "artifact_id": "art-uuid-xxxx",
  "artifact_type": "checkpoint",
  "size_bytes": 1234567,
  "encryption": {
    "algorithm": "AES-256-GCM",
    "key_id": "dek-uuid-xxxx"
  },
  "content_hash": "sha256:xxxx",
  "metadata": { ... },
  "created_at": "2026-01-28T10:05:00Z"
}
```

#### POST /v1/training_clients/{id}/load_state

Load training state from an encrypted artifact.

**Request:**
```json
{
  "artifact_id": "art-uuid-xxxx"
}
```

**Response (200 OK):**
```json
{
  "training_client_id": "tc-uuid-xxxx",
  "loaded_artifact_id": "art-uuid-xxxx",
  "step": 1000,
  "status": "ready"
}
```

#### GET /v1/futures/{id}

Get future status.

**Response (200 OK):**
```json
{
  "future_id": "fut-uuid-xxxx",
  "status": "completed",
  "created_at": "2026-01-28T10:00:01Z",
  "started_at": "2026-01-28T10:00:02Z",
  "completed_at": "2026-01-28T10:00:05Z",
  "training_client_id": "tc-uuid-xxxx",
  "operation": "forward_backward"
}
```

#### GET /v1/futures/{id}/result

Get future result (blocks if not complete, or returns immediately with result).

**Response (200 OK):**
```json
{
  "future_id": "fut-uuid-xxxx",
  "status": "completed",
  "result": {
    "loss": 2.34,
    "grad_norm": 1.5,
    "tokens_processed": 4096,
    "dp_metrics": null
  }
}
```

**Response (202 Accepted):** (If still pending/running)
```json
{
  "future_id": "fut-uuid-xxxx",
  "status": "running",
  "message": "Operation still in progress"
}
```

---

## 3. Privacy Model

### 3.1 Data Classification

| Data Type | Storage | Encryption | Audit |
|-----------|---------|------------|-------|
| Model Weights | Artifact Store | AES-256-GCM | Yes |
| LoRA Adapters | Artifact Store | AES-256-GCM | Yes |
| Optimizer State | Artifact Store | AES-256-GCM | Yes |
| Checkpoints | Artifact Store | AES-256-GCM | Yes |
| Training Batches | Memory only | Not stored | Hash only |
| Prompts/Completions | Memory only | Not stored | Hash only |
| API Keys | Database | Argon2id hash | Yes |
| Audit Logs | Database | Plaintext (integrity protected) | Self |

### 3.2 Encryption Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Key Hierarchy             │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
              ┌─────▼─────┐                    ┌──────▼──────┐
              │    KEK    │                    │   Signing   │
              │  (Vault)  │                    │     Key     │
              └─────┬─────┘                    └─────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
  ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
  │  DEK-T1   │ │DEK-T2 │ │  DEK-Tn   │
  │ (Tenant1) │ │       │ │           │
  └─────┬─────┘ └───────┘ └───────────┘
        │
  ┌─────▼────────────────────────┐
  │  Encrypted Artifact Store    │
  │  - Weights (AES-256-GCM)     │
  │  - Optimizer state           │
  │  - Checkpoints               │
  └──────────────────────────────┘
```

#### Key Hierarchy

1. **Master KEK (Key Encryption Key)**
   - Stored in secure vault (HashiCorp Vault, AWS KMS, or local HSM)
   - Never leaves the vault
   - Used only to wrap/unwrap DEKs

2. **Per-Tenant DEK (Data Encryption Key)**
   - Generated per tenant
   - Wrapped by KEK, stored wrapped in database
   - Unwrapped in memory only when needed
   - Rotatable without re-encrypting all data (envelope encryption)

3. **Per-Artifact IV/Nonce**
   - Unique 96-bit nonce per artifact
   - Stored with artifact metadata

#### Encryption Algorithm

```
Algorithm: AES-256-GCM
Key Size: 256 bits
Nonce Size: 96 bits (12 bytes)
Tag Size: 128 bits (16 bytes)
AAD: artifact_id || tenant_id || created_at
```

### 3.3 Audit Log

The audit log provides an immutable, tamper-evident record of all operations.

#### Log Entry Schema

```json
{
  "entry_id": "uuid",
  "tenant_id": "uuid",
  "training_client_id": "uuid",
  "operation": "forward_backward",
  "request_hash": "sha256:xxxx",
  "request_size_bytes": 4096,
  "artifact_ids_produced": ["art-uuid-1"],
  "artifact_ids_consumed": [],
  "started_at": "2026-01-28T10:00:00Z",
  "completed_at": "2026-01-28T10:00:05Z",
  "duration_ms": 5000,
  "success": true,
  "error_code": null,
  "error_message": null,
  "prev_hash": "sha256:yyyy",
  "record_hash": "sha256:zzzz",
  "dp_metrics": null
}
```

#### Hash Chaining

Each log entry includes a hash of the previous entry, creating an append-only chain:

```
record_hash = SHA256(
    entry_id ||
    tenant_id ||
    training_client_id ||
    operation ||
    request_hash ||
    artifact_ids_produced ||
    started_at ||
    completed_at ||
    success ||
    prev_hash
)
```

Tampering with any entry will break the hash chain for all subsequent entries.

### 3.4 Data Retention

| Data Type | Retention | Deletion |
|-----------|-----------|----------|
| Audit Logs | 7 years (configurable) | Tenant request only |
| Artifacts | Until explicit deletion | Tenant request |
| Training Clients | Until explicit deletion | Cascade deletes artifacts |
| API Keys | Until revoked | Immediate soft-delete |

### 3.5 Privacy-Preserving Logging

The audit log explicitly **does not** contain:

- Raw training data (prompts, completions, batches)
- Model weights or gradients
- Decrypted artifact contents
- PII from training data

Only hashes and sizes are logged to enable verification without exposing content.

---

## 4. Threat Model

### 4.1 Actors

| Actor | Trust Level | Capabilities |
|-------|-------------|--------------|
| **Tenant** | Trusted | Owns data, legitimate API access |
| **Service Operator** | Partially Trusted | Infrastructure access, no key access |
| **Network Attacker** | Untrusted | Passive/active network attacks |
| **Other Tenants** | Untrusted | Separate tenant, legitimate API access |

### 4.2 Security Properties

#### Confidentiality

| Threat | Mitigation |
|--------|------------|
| Network eavesdropping | TLS 1.3 for all connections |
| Server-side data exposure | Encryption-at-rest with per-tenant keys |
| Cross-tenant data access | Tenant isolation at API, DB, and storage layers |
| Operator data access | Operator cannot access DEKs (vault-protected KEK) |

#### Integrity

| Threat | Mitigation |
|--------|------------|
| Artifact tampering | AES-GCM authentication tag |
| Audit log tampering | Hash chaining |
| Request replay | Request nonces + timestamp validation |
| API key theft | Key rotation, short expiry, IP allowlisting (optional) |

#### Availability

| Threat | Mitigation |
|--------|------------|
| DoS on training | Rate limiting, queue limits per tenant |
| Resource exhaustion | Quotas, monitoring, auto-scaling |

### 4.3 Assumptions

1. **Vault Security**: The key vault (KEK storage) is assumed secure and correctly configured.
2. **TLS Termination**: TLS termination happens at a trusted boundary.
3. **Database Integrity**: The database is protected by standard access controls.
4. **Tenant API Key Protection**: Tenants are responsible for protecting their API keys.

### 4.4 Out of Scope

The following are explicitly out of scope for v1:

- Hardware-based attestation (TEE/SGX)
- End-to-end encryption where server cannot compute
- Byzantine fault tolerance
- Client-side gradient encryption
- Zero-knowledge proofs of training

### 4.5 Differential Privacy Threat Model

When DP mode is enabled:

| Property | Guarantee |
|----------|-----------|
| Training data privacy | (epsilon, delta)-DP guarantee against gradient attacks |
| Membership inference | Bounded by DP parameters |
| Model inversion | Bounded by DP parameters + sampling constraints |

**Limitations:**
- DP accounting is best-effort (based on theoretical analysis)
- Side-channel attacks not covered
- DP does not protect against architectural biases

---

## 5. Configuration

### 5.1 Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `TG_TINKER_API_KEY` | API key for authentication | Yes | - |
| `TG_TINKER_BASE_URL` | Base URL for TG-Tinker API | No | `https://api.tensorguard.io` |
| `TG_TINKER_TENANT_ID` | Tenant ID (if not in key) | No | Derived from key |
| `TG_TINKER_TIMEOUT` | Default request timeout (seconds) | No | `300` |
| `TG_TINKER_RETRY_COUNT` | Max retries for failed requests | No | `3` |

### 5.2 Server Configuration

```yaml
# config/tg_tinker.yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

queue:
  type: "redis"  # "memory" | "redis" | "sqs"
  max_pending_per_tenant: 100
  worker_concurrency: 4

storage:
  type: "s3"  # "local" | "s3" | "gcs"
  bucket: "tg-tinker-artifacts"
  encryption:
    enabled: true
    key_vault: "hashicorp"  # "hashicorp" | "aws_kms" | "local"
    kek_id: "tg-tinker-kek"

database:
  url: "postgresql://user:pass@host:5432/tg_tinker"
  pool_size: 10

audit:
  enabled: true
  retention_days: 2555  # 7 years

dp:
  enabled: true
  default_accountant: "rdp"
  max_epsilon: 10.0
```

---

## 6. Error Handling

### 6.1 Error Response Format

```json
{
  "error": {
    "code": "TRAINING_CLIENT_NOT_FOUND",
    "message": "Training client with ID 'tc-xxx' not found",
    "details": {
      "training_client_id": "tc-xxx"
    },
    "request_id": "req-uuid"
  }
}
```

### 6.2 Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid API key |
| `PERMISSION_DENIED` | 403 | Key lacks required permission |
| `TRAINING_CLIENT_NOT_FOUND` | 404 | Training client does not exist |
| `FUTURE_NOT_FOUND` | 404 | Future does not exist |
| `ARTIFACT_NOT_FOUND` | 404 | Artifact does not exist |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `RATE_LIMITED` | 429 | Too many requests |
| `QUEUE_FULL` | 503 | Operation queue is full |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `FUTURE_TIMEOUT` | 408 | Future did not complete in time |
| `FUTURE_CANCELLED` | 409 | Future was cancelled |
| `DP_BUDGET_EXCEEDED` | 400 | Differential privacy budget exhausted |

---

## Appendix A: SDK Quick Reference

```python
from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig, DPConfig

# Initialize client
service = ServiceClient()

# Create training client with DP
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    dp_config=DPConfig(
        enabled=True,
        noise_multiplier=1.0,
        max_grad_norm=1.0
    )
)
tc = service.create_training_client(config)

# Training loop with async execution
for batch in dataloader:
    fb_future = tc.forward_backward(batch)
    opt_future = tc.optim_step()

    # Overlap: submit next batch while waiting
    result = fb_future.result()
    opt_result = opt_future.result()

    print(f"Loss: {result.loss}, Epsilon: {opt_result.dp_metrics.total_epsilon}")

# Save checkpoint
checkpoint = tc.save_state()
print(f"Saved: {checkpoint.artifact_id}")

# Sample from model
samples = tc.sample(["Once upon a time"], max_tokens=100)
print(samples[0].completion)
```

---

## Appendix B: Comparison with Tinker API

| Feature | Tinker API | TG-Tinker |
|---------|------------|-----------|
| Core Primitives | forward_backward, optim_step, sample, save_state | Same |
| Async Execution | Futures | Same |
| Encryption at Rest | Optional | **Default** (per-tenant keys) |
| Audit Logging | Basic | **Hash-chained, tamper-evident** |
| DP Support | None | **Built-in with accounting** |
| Batch Storage | Server-side | **Memory only, hash logged** |
| Multi-tenancy | Yes | Yes (with strict isolation) |
| Open Source | No | Yes |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-28 | Initial specification |
