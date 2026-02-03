"""
Weights & Biases Integration Example

Demonstrates logging TenSafe training metrics to Weights & Biases
for experiment tracking and visualization.

Requirements:
- TenSafe account and API key
- Weights & Biases account and API key
- pip install wandb

Usage:
    export TENSAFE_API_KEY="your-api-key"
    export WANDB_API_KEY="your-wandb-key"
    python wandb_tracking.py
"""

import os
import wandb
from tensafe import TenSafeClient


def main():
    # Initialize W&B
    wandb.init(
        project="tensafe-training",
        name="lora-finetuning-dp",
        config={
            "model": "meta-llama/Llama-3-8B",
            "lora_rank": 16,
            "lora_alpha": 32.0,
            "batch_size": 8,
            "dp_epsilon": 8.0,
            "dp_delta": 1e-5,
            "noise_multiplier": 1.0,
        },
    )

    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Create training client
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
            "target_delta": 1e-5,
        },
        batch_size=8,
    )

    # Log training client ID
    wandb.config.update({"training_client_id": tc.id})

    # Training loop
    num_steps = 1000
    sample_batch = {
        "input_ids": [[1, 2, 3, 4, 5]] * 8,
        "attention_mask": [[1, 1, 1, 1, 1]] * 8,
        "labels": [[2, 3, 4, 5, 6]] * 8,
    }

    for step in range(num_steps):
        # Forward-backward
        fb_future = tc.forward_backward(batch=sample_batch)
        fb_result = fb_future.result()

        # Optimizer step
        opt_future = tc.optim_step(apply_dp_noise=True)
        opt_result = opt_future.result()

        # Log metrics to W&B
        log_dict = {
            "step": step + 1,
            "loss": fb_result.get("loss", 0.0),
            "gradient_norm": fb_result.get("gradient_norm", 0.0),
        }

        # Add DP metrics
        if "dp_metrics" in opt_result:
            dp = opt_result["dp_metrics"]
            log_dict.update({
                "dp/epsilon": dp.get("total_epsilon", 0.0),
                "dp/delta": dp.get("delta", 0.0),
                "dp/budget_remaining": 8.0 - dp.get("total_epsilon", 0.0),
            })

        wandb.log(log_dict)

        # Save checkpoint periodically
        if (step + 1) % 100 == 0:
            checkpoint = tc.save_state(
                include_optimizer=True,
                metadata={"step": step + 1, "wandb_run_id": wandb.run.id},
            )

            # Log artifact to W&B
            artifact = wandb.Artifact(
                name=f"checkpoint-{step+1}",
                type="model",
                metadata={
                    "step": step + 1,
                    "artifact_id": checkpoint.artifact_id,
                    "dp_epsilon": tc.get_dp_metrics()["total_epsilon"],
                },
            )
            wandb.log_artifact(artifact)
            print(f"Step {step+1}: Saved checkpoint and logged to W&B")

    # Final summary
    final_metrics = tc.get_dp_metrics()
    wandb.summary["final_epsilon"] = final_metrics["total_epsilon"]
    wandb.summary["final_delta"] = final_metrics["delta"]
    wandb.summary["total_steps"] = num_steps

    wandb.finish()
    print("Training complete! View results at:", wandb.run.url)


if __name__ == "__main__":
    main()
