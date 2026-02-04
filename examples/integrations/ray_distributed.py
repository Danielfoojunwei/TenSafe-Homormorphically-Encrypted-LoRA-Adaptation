"""
Ray Train Distributed Training

Demonstrates distributed training across multiple nodes using
Ray Train with TenSafe privacy guarantees.

Requirements:
- TenSafe account and API key
- Ray cluster access
- pip install tensafe[distributed] ray

Usage:
    export TENSAFE_API_KEY="your-api-key"
    export RAY_ADDRESS="ray://cluster-head:10001"
    python ray_distributed.py
"""

import os
import ray
from ray import train
from ray.train import ScalingConfig
from tensafe import TenSafeClient
from tensafe.distributed.ray import TenSafeTrainer


def train_func():
    """Training function that runs on each worker."""
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Get worker info
    worker_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    print(f"Worker {worker_rank}/{world_size} starting...")

    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={
            "rank": 16,
            "alpha": 32.0,
        },
        dp_config={
            "enabled": True,
            "noise_multiplier": 1.0,
            "target_epsilon": 8.0,
        },
        distributed_config={
            "world_size": world_size,
            "rank": worker_rank,
            "backend": "ray",
        },
    )

    # Training loop
    sample_batch = {
        "input_ids": [[1, 2, 3, 4, 5]] * 8,
        "attention_mask": [[1, 1, 1, 1, 1]] * 8,
        "labels": [[2, 3, 4, 5, 6]] * 8,
    }

    for step in range(100):
        fb = tc.forward_backward(batch=sample_batch).result()
        opt = tc.optim_step(apply_dp_noise=True).result()

        # Report metrics to Ray
        train.report({
            "loss": fb.get("loss", 0.0),
            "step": step + 1,
            "dp_epsilon": opt.get("dp_metrics", {}).get("total_epsilon", 0.0),
        })

    # Save checkpoint
    checkpoint = tc.save_state()
    return {"checkpoint_id": checkpoint.artifact_id}


def main():
    # Initialize Ray
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    # Configure distributed training
    scaling_config = ScalingConfig(
        num_workers=4,
        use_gpu=True,
        resources_per_worker={"GPU": 1},
    )

    # Create TenSafe Ray trainer
    trainer = TenSafeTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        run_config=train.RunConfig(
            name="tensafe-distributed",
            storage_path="/tmp/ray_results",
        ),
    )

    # Run training
    print("Starting distributed training...")
    result = trainer.fit()

    print("\nTraining complete!")
    print(f"Best checkpoint: {result.checkpoint}")
    print(f"Final metrics: {result.metrics}")


if __name__ == "__main__":
    main()
