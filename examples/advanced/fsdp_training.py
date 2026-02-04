"""
Fully Sharded Data Parallel (FSDP) Training

Demonstrates memory-efficient training using FSDP for large models
that don't fit on a single GPU.

Requirements:
- TenSafe account with multi-GPU access
- pip install tensafe[distributed]

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python fsdp_training.py
"""

import os
from tensafe import TenSafeClient
from tensafe.distributed import FSDPConfig, ShardingStrategy


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Configure FSDP
    fsdp_config = FSDPConfig(
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Shard everything
        cpu_offload=False,  # Keep on GPU for speed
        backward_prefetch="BACKWARD_PRE",  # Prefetch for efficiency
        mixed_precision="bf16",  # Use BF16 for memory savings
        limit_all_gathers=True,  # Reduce peak memory
    )

    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-70B",  # Large model
        lora_config={
            "rank": 16,
            "alpha": 32.0,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
        dp_config={
            "enabled": True,
            "noise_multiplier": 1.0,
            "target_epsilon": 8.0,
        },
        fsdp_config=fsdp_config,
    )

    print("FSDP Configuration:")
    print(f"  Sharding: {fsdp_config.sharding_strategy}")
    print(f"  Mixed precision: {fsdp_config.mixed_precision}")
    print(f"  CPU offload: {fsdp_config.cpu_offload}")
    print()

    # Training with FSDP
    sample_batch = {
        "input_ids": [[1, 2, 3, 4, 5]] * 4,  # Smaller batch for large model
        "attention_mask": [[1, 1, 1, 1, 1]] * 4,
        "labels": [[2, 3, 4, 5, 6]] * 4,
    }

    for step in range(50):
        fb = tc.forward_backward(batch=sample_batch).result()
        opt = tc.optim_step(apply_dp_noise=True).result()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: loss={fb.get('loss', 0):.4f}")

    print("\nFSDP training complete!")


if __name__ == "__main__":
    main()
