"""
MLflow Model Registry Integration

Demonstrates registering TenSafe models with MLflow for
versioning, staging, and deployment workflows.

Requirements:
- TenSafe account and API key
- MLflow tracking server
- pip install mlflow

Usage:
    export TENSAFE_API_KEY="your-api-key"
    export MLFLOW_TRACKING_URI="http://localhost:5000"
    python mlflow_registry.py
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from tensafe import TenSafeClient


def main():
    # Configure MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("tensafe-experiments")

    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    with mlflow.start_run(run_name="lora-training-dp"):
        # Log parameters
        mlflow.log_params({
            "model": "meta-llama/Llama-3-8B",
            "lora_rank": 16,
            "lora_alpha": 32.0,
            "dp_epsilon_target": 8.0,
            "dp_delta": 1e-5,
            "batch_size": 8,
        })

        # Create training client
        tc = client.create_training_client(
            model_ref="meta-llama/Llama-3-8B",
            lora_config={"rank": 16, "alpha": 32.0},
            dp_config={"enabled": True, "target_epsilon": 8.0},
        )

        mlflow.log_param("training_client_id", tc.id)

        # Training loop (simplified)
        sample_batch = {
            "input_ids": [[1, 2, 3, 4, 5]] * 8,
            "attention_mask": [[1, 1, 1, 1, 1]] * 8,
            "labels": [[2, 3, 4, 5, 6]] * 8,
        }

        for step in range(100):
            fb = tc.forward_backward(batch=sample_batch).result()
            opt = tc.optim_step(apply_dp_noise=True).result()

            # Log metrics
            mlflow.log_metrics({
                "loss": fb.get("loss", 0.0),
                "dp_epsilon": opt.get("dp_metrics", {}).get("total_epsilon", 0.0),
            }, step=step)

        # Save final checkpoint
        checkpoint = tc.save_state(include_optimizer=False)

        # Log model artifact to MLflow
        mlflow.log_artifact(
            local_path=f"/tmp/checkpoint_{checkpoint.artifact_id}",
            artifact_path="model",
        )

        # Log final metrics
        final_dp = tc.get_dp_metrics()
        mlflow.log_metrics({
            "final_epsilon": final_dp["total_epsilon"],
            "final_delta": final_dp["delta"],
        })

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_name = "tensafe-llama3-lora"

        result = mlflow.register_model(model_uri, model_name)
        print(f"Model registered: {model_name} version {result.version}")

        # Transition to staging
        mlflow_client = MlflowClient()
        mlflow_client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging",
        )
        print(f"Model transitioned to Staging")

    print("\nMLflow UI:", os.environ.get("MLFLOW_TRACKING_URI"))


if __name__ == "__main__":
    main()
