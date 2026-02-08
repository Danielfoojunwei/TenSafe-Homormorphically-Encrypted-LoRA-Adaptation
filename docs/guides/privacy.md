# Privacy Guide

TenSafe provides multiple layers of privacy protection for ML training and inference.

## Privacy Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   Privacy Layers                           │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  Differential   │  │  Homomorphic    │                 │
│  │  Privacy (DP)   │  │  Encryption     │                 │
│  │                 │  │  (N2HE)         │                 │
│  │  • DP-SGD       │  │  • Zero-Rotation │                 │
│  │  • Noise        │  │  • MOAI Contract │                 │
│  │  • Clipping     │  │  • TEE Evidence  │                 │
│  │  • Accounting   │  │  • CKKS/TFHE     │                 │
│  └─────────────────┘  └─────────────────┘                 │
│                                                            │
│  ┌─────────────────────────────────────────────────┐      │
│  │           Encryption at Rest                     │      │
│  │  • AES-256-GCM artifacts                        │      │
│  │  • KEK/DEK key hierarchy                        │      │
│  │  • Tenant-isolated keys                         │      │
│  └─────────────────────────────────────────────────┘      │
│                                                            │
│  ┌─────────────────────────────────────────────────┐      │
│  │           Immutable Audit Log                    │      │
│  │  • Hash-chained records                         │      │
│  │  • Non-repudiation                              │      │
│  │  • Compliance reporting                         │      │
│  └─────────────────────────────────────────────────┘      │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

## Differential Privacy

### What is DP?

Differential Privacy provides mathematical guarantees that model outputs don't reveal information about individual training examples.

### DP-SGD Training

```python
from tg_tinker import DPConfig, TrainingConfig, LoRAConfig

dp_config = DPConfig(
    enabled=True,
    noise_multiplier=1.0,    # Noise scale
    max_grad_norm=1.0,       # Gradient clipping bound
    target_epsilon=8.0,      # Privacy budget
    target_delta=1e-5,       # Failure probability
    accountant_type="rdp",   # Accounting method
)

config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
    dp_config=dp_config,
)

tc = service.create_training_client(config)
```

### Privacy Budget

The (epsilon, delta) privacy budget determines the privacy guarantee:

| Epsilon | Privacy Level | Use Case |
|---------|---------------|----------|
| 1-3 | Strong | Sensitive medical/financial data |
| 3-8 | Moderate | Most production use cases |
| 8-15 | Relaxed | Less sensitive data |
| 15+ | Weak | Research/experimentation |

### Monitoring DP Metrics

```python
for batch in dataloader:
    fb_result = tc.forward_backward(batch).result()
    opt_result = tc.optim_step(apply_dp_noise=True).result()

    if opt_result.dp_metrics:
        metrics = opt_result.dp_metrics
        print(f"Noise applied: {metrics.noise_applied}")
        print(f"Epsilon this step: {metrics.epsilon_spent:.4f}")
        print(f"Total epsilon: {metrics.total_epsilon:.4f}")
        print(f"Gradient norm before clip: {metrics.grad_norm_before_clip:.4f}")
        print(f"Gradient norm after clip: {metrics.grad_norm_after_clip:.4f}")
        print(f"Samples clipped: {metrics.num_clipped}")
```

### Budget Exhaustion

```python
from tg_tinker import DPBudgetExceededError

try:
    for batch in dataloader:
        tc.forward_backward(batch).result()
        tc.optim_step(apply_dp_noise=True).result()
except DPBudgetExceededError as e:
    print(f"Budget exceeded: epsilon={e.details['current_epsilon']}")
    print(f"Max allowed: {e.details['max_epsilon']}")
    # Save checkpoint before stopping
    tc.save_state(metadata={"reason": "dp_budget_exceeded"})
```

### DP Best Practices

1. **Start with higher epsilon, reduce gradually**
   ```python
   # Experiment phase
   dp_config = DPConfig(target_epsilon=15.0)

   # Production
   dp_config = DPConfig(target_epsilon=8.0)
   ```

2. **Balance noise and utility**
   ```python
   # Higher noise = more privacy, lower utility
   DPConfig(noise_multiplier=1.5, max_grad_norm=0.5)

   # Lower noise = less privacy, higher utility
   DPConfig(noise_multiplier=0.5, max_grad_norm=1.0)
   ```

3. **Use gradient accumulation wisely**
   ```python
   # Larger effective batch size = better privacy/utility
   config = TrainingConfig(
       batch_size=8,
       gradient_accumulation_steps=8,  # Effective: 64
       dp_config=DPConfig(noise_multiplier=1.0),
   )
   ```

## Homomorphic Encryption (N2HE)

### What is HE?

Homomorphic Encryption allows computation on encrypted data without decryption. TenSafe integrates N2HE for:

- **Zero-Rotation (MOAI)**: Eliminates side-channel leakage from HE rotations.
- **Evidence Fabric**: Hardware-backed proof of secure computation environments (TEE).
- **Encrypted LoRA**: Keeps adapter deltas private during multi-tenant sharing.
- **Secure Aggregation**: MPC protocol for federated updates.

### N2HE Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    N2HE Module                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │   Core     │  │   Keys     │  │   Adapter  │        │
│  │  N2HEScheme│  │ HEKeyMgr   │  │ Encrypted  │        │
│  │  N2HEContext│ │ HEKeyBundle│  │ LoRARuntime│        │
│  │  LWE/RLWE  │  │ PK/SK/EK   │  │            │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│                                                          │
│  ┌────────────┐  ┌────────────────────────────┐        │
│  │ Inference  │  │      Serialization         │        │
│  │ Private    │  │  CiphertextFormat          │        │
│  │ Mode       │  │  Binary/JSON/Base64/CBOR   │        │
│  └────────────┘  └────────────────────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### HE Key Management

```python
from tensorguard.n2he import (
    HEKeyManager,
    HESchemeParams,
    N2HEScheme,
)

# Create key manager
key_manager = HEKeyManager(tenant_id="tenant-123")

# Generate key bundle
params = HESchemeParams(
    scheme=N2HEScheme.CKKS,
    poly_modulus_degree=8192,
    security_level=128,
)

bundle = key_manager.generate_key_bundle(params=params)
print(f"Public key size: {len(bundle.public_key.data)} bytes")
print(f"Total bundle size: {bundle.get_total_size()} bytes")
```

### Encrypted LoRA Adapter

```python
from tensorguard.n2he import (
    EncryptedLoRARuntime,
    AdapterEncryptionConfig,
    create_encrypted_runtime,
)

# Create encrypted runtime
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
from tensorguard.n2he import (
    PrivateInferenceMode,
    create_private_inference_mode,
)

# Create private inference mode
inference = create_private_inference_mode(
    key_bundle=bundle,
    params=params,
)

# Encrypt input
import numpy as np
embedding = np.random.randn(512).astype(np.float32)
encrypted_input = inference.encrypt_input(embedding)

# Process (encrypted computation happens server-side)
result = inference.process(encrypted_input)

# Verify TEE Attestation (Evidence Fabric)
if result.evidence:
    inference.verify_attestation(result.evidence)

# Decrypt output
output = inference.decrypt_output(result.encrypted_output)
```

### Ciphertext Serialization

```python
from tensorguard.n2he import (
    CiphertextFormat,
    serialize_ciphertext,
    deserialize_ciphertext,
    create_ciphertext_bundle,
)

# Encrypt data
context = N2HEContext(params=params)
data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
ciphertext = context.encrypt(data)

# Serialize for storage/transmission
serialized = serialize_ciphertext(
    ciphertext,
    format=CiphertextFormat.BINARY,
)
print(f"Serialized size: {len(serialized.data)} bytes")

# Deserialize
restored = deserialize_ciphertext(serialized, params)
```

### N2HE Performance

Benchmark results from E2E tests:

| Operation | Time (ms) | Throughput |
|-----------|-----------|------------|
| Key Generation | 0.114 | 8,751 ops/sec |
| Encryption | 0.022 | 46,014 ops/sec |
| LoRA Delta | 0.676 | 1,479 ops/sec |
| Decryption | 0.019 | 52,000 ops/sec |

## Encrypted Artifacts

All TenSafe artifacts are encrypted at rest:

```python
# Save encrypted checkpoint
checkpoint = tc.save_state()

print(f"Encryption: {checkpoint.encryption.algorithm}")  # AES-256-GCM
print(f"Key ID: {checkpoint.encryption.key_id}")

# Artifacts remain encrypted when downloaded
encrypted_data = service.pull_artifact(checkpoint.artifact_id)
# Decryption requires tenant key
```

### Key Hierarchy

```
┌─────────────────────────────────┐
│    Master Key (HSM-backed)      │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│    KEK (Key Encryption Key)     │
│    Per-tenant                   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│    DEK (Data Encryption Key)    │
│    Per-artifact                 │
└─────────────────────────────────┘
```

## Audit Logging

All operations are logged with cryptographic integrity:

```python
# Get audit logs
logs = service.get_audit_logs(
    training_client_id=tc.id,
    operation="forward_backward",
    limit=100,
)

for entry in logs:
    print(f"Operation: {entry.operation}")
    print(f"Request hash: {entry.request_hash}")
    print(f"Record hash: {entry.record_hash}")
    print(f"Previous hash: {entry.prev_hash}")  # Chain integrity
    print(f"Success: {entry.success}")
```

### Audit Log Fields

| Field | Description |
|-------|-------------|
| `entry_id` | Unique log entry ID |
| `request_hash` | Hash of the request |
| `record_hash` | Hash of this record |
| `prev_hash` | Hash of previous record (chain) |
| `artifact_ids_produced` | Artifacts created |
| `artifact_ids_consumed` | Artifacts used |
| `dp_metrics` | DP metrics if applicable |

## Compliance

TenSafe helps meet regulatory requirements:

### GDPR
- Right to erasure via tenant key rotation
- Data minimization through DP
- Audit trail for accountability

### HIPAA
- Encryption at rest and in transit
- Access controls and audit logs
- DP for de-identification

### SOC 2
- Immutable audit logs
- Cryptographic integrity
- Access monitoring

## Best Practices

### 1. Layer Privacy Protections

```python
# Combine DP + Encryption for defense in depth
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16),
    dp_config=DPConfig(
        enabled=True,
        target_epsilon=8.0,
    ),
)
# Artifacts automatically encrypted
```

### 2. Monitor Privacy Budget

```python
# Track epsilon across training
epsilon_history = []

for batch in dataloader:
    result = tc.optim_step(apply_dp_noise=True).result()
    if result.dp_metrics:
        epsilon_history.append(result.dp_metrics.total_epsilon)

        # Alert at 80% budget
        if result.dp_metrics.total_epsilon > 0.8 * target_epsilon:
            print("Warning: Approaching privacy budget limit")
```

### 3. Regular Key Rotation

```python
# Rotate keys periodically
key_manager.rotate_keys(tenant_id)
```

### 4. Verify Audit Chain

```python
# Verify chain integrity
logs = service.get_audit_logs(training_client_id=tc.id)

for i in range(1, len(logs)):
    assert logs[i].prev_hash == logs[i-1].record_hash
```

## Next Steps

- [Training Guide](training.md) - Training workflows
- [Cookbook: Privacy Budget](../cookbook/privacy-budget.md) - Advanced DP usage
- [Cookbook: Encrypted Inference](../cookbook/encrypted-inference.md) - N2HE tutorial
