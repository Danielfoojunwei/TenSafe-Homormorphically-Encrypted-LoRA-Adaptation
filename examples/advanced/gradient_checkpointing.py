"""
Gradient Checkpointing Example

Demonstrates memory-efficient training using gradient checkpointing
to trade compute for memory.

This allows training larger batch sizes or longer sequences
while maintaining privacy guarantees.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python gradient_checkpointing.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Enable gradient checkpointing for memory efficiency
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
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
        training_config={
            "gradient_checkpointing": True,
            # Checkpoint every N layers
            "checkpoint_every_n_layers": 4,
            # Use reentrant checkpointing for compatibility
            "use_reentrant": False,
        },
    )

    print("Gradient checkpointing enabled")
    print("Memory savings: ~60% activation memory reduction")
    print("Compute overhead: ~20% additional forward passes")
    print()

    # Now we can use larger batches
    large_batch = {
        "input_ids": [[1, 2, 3, 4, 5] * 100] * 16,  # Larger batch
        "attention_mask": [[1, 1, 1, 1, 1] * 100] * 16,
        "labels": [[2, 3, 4, 5, 6] * 100] * 16,
    }

    print(f"Batch size: 16")
    print(f"Sequence length: 500")
    print()

    for step in range(50):
        fb = tc.forward_backward(batch=large_batch).result()
        opt = tc.optim_step(apply_dp_noise=True).result()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: loss={fb.get('loss', 0):.4f}")

    print("\nTraining complete with gradient checkpointing!")


if __name__ == "__main__":
    main()
