# TenSafe Distributed Training

This component handles distributed, privacy-preserving training across multi-GPU and multi-node clusters using the Ray Train framework.

## Component Overview

- **`TenSafeRayTrainer`**: The primary entry point for distributed training. It orchestrates worker setup, data sharding, and coordinated privacy accounting.
- **Secure Aggregation**: A cryptographic layer that aggregates gradients using pairwise masking to ensure no single worker or aggregator sees individual client updates.
- **Production RDP Accountant**: A centralized privacy accountant using the Rényi Differential Privacy (RDP) mechanism to track the global privacy budget across all shards.

## Distributed Flow

1. **Worker Boot**: Ray spawns training workers.
2. **TEE Verification**: Each worker performs a TEE attestation check via Evidence Fabric.
3. **Local Compute**: Workers compute per-sample clipped gradients.
4. **Secure Aggregation**: Gradients are aggregated via the secure protocol.
5. **DP Noise Injection**: Calibrated noise is added to the aggregated gradient.
6. **Unified Update**: All workers apply the same privacy-preserving update.

## Configuration

Standard configuration via `TenSafeRayConfig`:

```python
config = TenSafeRayConfig(
    num_workers=8,
    use_gpu=True,
    dp_config=DPConfig(
        target_epsilon=8.0,
        target_delta=1e-5
    ),
    secure_aggregation=True
)
```

## Internal Dependencies

- `tensorguard.privacy.accountants`: Production-grade DP implementations.
- `ray.train`: Foundation for distributed orchestration.
- `tensorguard.observability`: For multi-node span and metric collection.
