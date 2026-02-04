"""
Multi-GPU Training Example

Demonstrates data-parallel training across multiple GPUs for
faster training with privacy guarantees.

Requirements:
- TenSafe account with GPU access
- Multiple GPUs available
- pip install tensafe[distributed]

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python multi_gpu_training.py
"""

import os
from tensafe import TenSafeClient
from tensafe.distributed import DataParallelConfig


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Configure multi-GPU training
    dp_config = DataParallelConfig(
        num_gpus=4,  # Use 4 GPUs
        gradient_accumulation_steps=2,
        # Each GPU processes batch_size samples
        # Effective batch size = num_gpus * batch_size * gradient_accumulation_steps
    )

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
            "max_grad_norm": 1.0,
            "target_epsilon": 8.0,
        },
        batch_size=8,
        data_parallel_config=dp_config,
    )

    print(f"Training with {dp_config.num_gpus} GPUs")
    print(f"Per-GPU batch size: 8")
    print(f"Gradient accumulation: {dp_config.gradient_accumulation_steps}")
    print(f"Effective batch size: {dp_config.num_gpus * 8 * dp_config.gradient_accumulation_steps}")
    print()

    # Training loop
    sample_batch = {
        "input_ids": [[1, 2, 3, 4, 5]] * 8,
        "attention_mask": [[1, 1, 1, 1, 1]] * 8,
        "labels": [[2, 3, 4, 5, 6]] * 8,
    }

    for step in range(100):
        # Forward-backward (distributed across GPUs)
        fb = tc.forward_backward(batch=sample_batch).result()

        # Optimizer step (gradients synchronized across GPUs)
        opt = tc.optim_step(apply_dp_noise=True).result()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: loss={fb.get('loss', 0):.4f}")

    print("\nTraining complete!")
    print(f"Final DP epsilon: {tc.get_dp_metrics()['total_epsilon']:.4f}")


if __name__ == "__main__":
    main()
