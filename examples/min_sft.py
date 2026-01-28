#!/usr/bin/env python3
"""
Minimal Supervised Fine-Tuning (SFT) Example

This example demonstrates the core TG-Tinker SDK workflow:
1. Create a training client
2. Run forward-backward and optimizer steps asynchronously
3. Save checkpoints
4. Sample from the fine-tuned model

Prerequisites:
    pip install -e .
    export TG_TINKER_API_KEY="your-api-key"
    export TG_TINKER_BASE_URL="http://localhost:8080"  # For local server

Usage:
    python examples/min_sft.py
"""

import os
import sys

from tg_tinker import (
    ServiceClient,
    TrainingConfig,
    LoRAConfig,
    OptimizerConfig,
    DPConfig,
)


def create_dummy_batch(batch_size: int = 4, seq_len: int = 128):
    """Create a dummy training batch for demonstration."""
    return {
        "input_ids": [[1] * seq_len for _ in range(batch_size)],
        "attention_mask": [[1] * seq_len for _ in range(batch_size)],
        "labels": [[1] * seq_len for _ in range(batch_size)],
    }


def main():
    print("=" * 60)
    print("TG-Tinker Minimal SFT Example")
    print("=" * 60)

    # =========================================================================
    # Step 1: Initialize ServiceClient
    # =========================================================================
    print("\n[1] Initializing ServiceClient...")

    try:
        service = ServiceClient()
        print(f"    Connected to: {service._base_url}")
    except ValueError as e:
        print(f"    Error: {e}")
        print("    Please set TG_TINKER_API_KEY environment variable")
        sys.exit(1)

    # =========================================================================
    # Step 2: Configure and Create TrainingClient
    # =========================================================================
    print("\n[2] Creating TrainingClient...")

    # Configure LoRA for parameter-efficient fine-tuning
    lora_config = LoRAConfig(
        rank=16,
        alpha=32.0,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # Configure optimizer
    optim_config = OptimizerConfig(
        name="adamw",
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    # Optional: Configure differential privacy
    # Uncomment to enable DP mode
    dp_config = DPConfig(
        enabled=True,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        target_epsilon=8.0,
        target_delta=1e-5,
    )

    # Full training configuration
    config = TrainingConfig(
        model_ref="meta-llama/Llama-3-8B",
        lora_config=lora_config,
        optimizer=optim_config,
        dp_config=dp_config,  # Set to None to disable DP
        batch_size=4,
        gradient_accumulation_steps=4,
        metadata={"experiment": "min_sft_demo"},
    )

    tc = service.create_training_client(config)
    print(f"    Created: {tc}")
    print(f"    Training Client ID: {tc.id}")
    print(f"    DP Enabled: {tc.dp_enabled}")

    # =========================================================================
    # Step 3: Training Loop with Async Execution
    # =========================================================================
    print("\n[3] Running training loop...")

    num_steps = 5
    for step in range(num_steps):
        print(f"\n    Step {step + 1}/{num_steps}")

        # Create a batch (in real usage, this would come from a DataLoader)
        batch = create_dummy_batch(batch_size=4, seq_len=128)

        # Queue forward-backward pass (returns immediately with Future)
        fb_future = tc.forward_backward(batch)
        print(f"        Queued forward_backward: {fb_future.future_id}")

        # Queue optimizer step (can be called before waiting on fb_future)
        opt_future = tc.optim_step()
        print(f"        Queued optim_step: {opt_future.future_id}")

        # Now wait for results
        print("        Waiting for forward_backward result...")
        fb_result = fb_future.result(timeout=300)
        print(f"        Loss: {fb_result.loss:.4f}")
        print(f"        Grad Norm: {fb_result.grad_norm:.4f}")

        print("        Waiting for optim_step result...")
        opt_result = opt_future.result(timeout=300)
        print(f"        Step: {opt_result.step}")
        print(f"        Learning Rate: {opt_result.learning_rate:.2e}")

        # Report DP metrics if available
        if opt_result.dp_metrics:
            print(f"        DP Epsilon Spent: {opt_result.dp_metrics.epsilon_spent:.4f}")
            print(f"        DP Total Epsilon: {opt_result.dp_metrics.total_epsilon:.4f}")

    # =========================================================================
    # Step 4: Save Checkpoint
    # =========================================================================
    print("\n[4] Saving checkpoint...")

    checkpoint = tc.save_state(
        include_optimizer=True,
        metadata={
            "checkpoint_name": "final",
            "steps_completed": num_steps,
        },
    )
    print(f"    Artifact ID: {checkpoint.artifact_id}")
    print(f"    Size: {checkpoint.size_bytes:,} bytes")
    print(f"    Encryption: {checkpoint.encryption.algorithm}")
    print(f"    Content Hash: {checkpoint.content_hash[:16]}...")

    # Report final DP metrics
    if checkpoint.dp_metrics:
        print(f"    Final DP Epsilon: {checkpoint.dp_metrics.total_epsilon:.4f}")

    # =========================================================================
    # Step 5: Sample from Model
    # =========================================================================
    print("\n[5] Sampling from fine-tuned model...")

    sample_result = tc.sample(
        prompts=["Once upon a time in a land far away,"],
        max_tokens=50,
        temperature=0.7,
        top_p=0.9,
    )

    for sample in sample_result.samples:
        print(f"\n    Prompt: {sample.prompt}")
        print(f"    Completion: {sample.completion}")
        print(f"    Tokens Generated: {sample.tokens_generated}")
        print(f"    Finish Reason: {sample.finish_reason}")

    print(f"    Model Step: {sample_result.model_step}")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n[6] Done!")
    print(f"    Final training client state: {tc}")

    service.close()
    print("    ServiceClient closed")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
