"""
Continual Learning Example

Demonstrates incremental learning without catastrophic forgetting.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python continual_learning.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={"rank": 16, "alpha": 32.0},
        dp_config={"enabled": True, "target_epsilon": 8.0},
        continual_learning={
            "enabled": True,
            "method": "ewc",  # Elastic Weight Consolidation
            "ewc_lambda": 1000,  # Regularization strength
            "replay_buffer_size": 1000,  # Experience replay
        },
    )

    print("Continual Learning Configuration:")
    print("  Method: EWC (Elastic Weight Consolidation)")
    print("  Replay buffer: 1000 samples")
    print()

    # Train on Task 1
    print("=== Training Task 1: Code Generation ===")
    task1_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[2, 3, 4, 5, 6]] * 8}
    for step in range(50):
        tc.forward_backward(batch=task1_batch).result()
        tc.optim_step(apply_dp_noise=True).result()
    print("Task 1 complete")

    # Compute Fisher information for EWC
    tc.compute_fisher_information()

    # Train on Task 2 (with EWC regularization)
    print("\n=== Training Task 2: Translation ===")
    task2_batch = {"input_ids": [[6, 7, 8, 9, 10]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[7, 8, 9, 10, 11]] * 8}
    for step in range(50):
        tc.forward_backward(batch=task2_batch).result()
        tc.optim_step(apply_dp_noise=True, apply_ewc=True).result()
    print("Task 2 complete")

    # Evaluate on both tasks
    print("\n=== Evaluation ===")
    task1_loss = tc.evaluate(task1_batch)
    task2_loss = tc.evaluate(task2_batch)
    print(f"Task 1 loss: {task1_loss:.4f}")
    print(f"Task 2 loss: {task2_loss:.4f}")
    print(f"DP budget used: ε={tc.get_dp_metrics()['total_epsilon']:.4f}")


if __name__ == "__main__":
    main()
