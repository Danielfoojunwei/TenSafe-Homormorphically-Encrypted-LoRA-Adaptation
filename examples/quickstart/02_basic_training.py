#!/usr/bin/env python3
"""
Basic LoRA Training with TenSafe

This example demonstrates how to perform basic LoRA (Low-Rank Adaptation)
fine-tuning using TenSafe. LoRA is a parameter-efficient fine-tuning method
that adds small trainable matrices to the model while keeping the base
model frozen.

What this example demonstrates:
- Creating a TrainingClient with LoRA configuration
- Running forward and backward passes
- Performing optimizer steps
- Monitoring training progress
- Saving checkpoints

Prerequisites:
- TenSafe server running
- API key configured

Expected Output:
    Training configuration created
    Step 1: loss=2.3456, lr=0.0001
    Step 2: loss=2.1234, lr=0.0001
    ...
    Training complete! Final loss: 1.5678
    Checkpoint saved: my-lora-adapter
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def create_sample_dataset(num_samples: int = 100):
    """Create a simple sample dataset for demonstration."""
    # In production, you would load your actual dataset here
    samples = []
    for i in range(num_samples):
        samples.append({
            "input_ids": list(range(i % 50, (i % 50) + 128)),  # Simulated tokens
            "attention_mask": [1] * 128,
            "labels": list(range((i % 50) + 1, (i % 50) + 129)),
        })
    return samples


def main():
    """Demonstrate basic LoRA training with TenSafe."""

    from tg_tinker import ServiceClient
    from tg_tinker.schemas import TrainingConfig, LoRAConfig, OptimizerConfig

    # =========================================================================
    # Step 1: Initialize the client
    # =========================================================================
    print("Initializing TenSafe client...")

    client = ServiceClient(
        base_url=os.environ.get("TG_TINKER_BASE_URL", "http://localhost:8000"),
        api_key=os.environ.get("TG_TINKER_API_KEY", "demo-api-key"),
    )

    # =========================================================================
    # Step 2: Configure LoRA training
    # =========================================================================
    print("\nConfiguring LoRA training...")

    # LoRA configuration
    lora_config = LoRAConfig(
        rank=16,              # Low-rank dimension (smaller = fewer parameters)
        alpha=32,             # Scaling factor (typically 2x rank)
        target_modules=[      # Which layers to add LoRA adapters to
            "q_proj",         # Query projection
            "v_proj",         # Value projection
            "k_proj",         # Key projection (optional)
            "o_proj",         # Output projection (optional)
        ],
        dropout=0.05,         # Dropout for regularization
        bias="none",          # Don't train biases
    )

    # Optimizer configuration
    optimizer_config = OptimizerConfig(
        name="adamw",
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # Full training configuration
    training_config = TrainingConfig(
        model_ref="meta-llama/Llama-3-8B",  # Base model to fine-tune
        lora_config=lora_config,
        optimizer=optimizer_config,
        batch_size=8,
        gradient_accumulation_steps=4,      # Effective batch size = 32
        max_steps=1000,
    )

    print(f"  Model: {training_config.model_ref}")
    print(f"  LoRA rank: {lora_config.rank}")
    print(f"  Learning rate: {optimizer_config.lr}")
    print(f"  Batch size: {training_config.batch_size}")

    # =========================================================================
    # Step 3: Create the training client
    # =========================================================================
    print("\nCreating training client...")

    try:
        tc = client.create_training_client(training_config)
        print(f"Training client created: {tc.training_client_id}")
    except Exception as e:
        print(f"Note: Server connection failed ({e})")
        print("Running in demonstration mode with simulated responses.")
        # For demonstration, we'll continue with mock behavior
        return demonstrate_training_loop()

    # =========================================================================
    # Step 4: Training loop
    # =========================================================================
    print("\nStarting training loop...")

    dataset = create_sample_dataset(num_samples=100)
    num_steps = 10  # For demonstration

    for step in range(num_steps):
        # Get batch
        batch_start = (step * training_config.batch_size) % len(dataset)
        batch = dataset[batch_start:batch_start + training_config.batch_size]

        # Forward and backward pass
        future = tc.forward_backward(batch)
        result = future.result()  # Wait for completion

        # Get loss from result
        loss = result.get("loss", 0.0)

        # Optimizer step (every gradient_accumulation_steps)
        if (step + 1) % training_config.gradient_accumulation_steps == 0:
            tc.optim_step()

        # Log progress
        print(f"  Step {step + 1}/{num_steps}: loss={loss:.4f}")

    # =========================================================================
    # Step 5: Save checkpoint
    # =========================================================================
    print("\nSaving checkpoint...")

    save_result = tc.save_state(
        name="my-lora-adapter",
        metadata={
            "description": "My first LoRA adapter",
            "base_model": training_config.model_ref,
            "steps": num_steps,
        }
    )
    print(f"Checkpoint saved: {save_result.artifact_id}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"""
    Your LoRA adapter has been trained and saved.

    Key statistics:
    - Total steps: {num_steps}
    - Final loss: {loss:.4f}
    - Artifact ID: {save_result.artifact_id}

    Next steps:
    - Load and use this adapter for inference
    - Add differential privacy (see 04_dp_training.py)
    - Deploy with HE-LoRA encryption (see 05_encrypted_inference.py)
    """)

    client.close()


def demonstrate_training_loop():
    """Demonstrate training loop without actual server connection."""
    print("\n[Demo Mode] Simulating training loop...")

    # Simulated training progress
    losses = [2.5, 2.3, 2.1, 1.9, 1.8, 1.7, 1.65, 1.6, 1.55, 1.5]

    for step, loss in enumerate(losses):
        print(f"  Step {step + 1}/10: loss={loss:.4f}")

    print("\n[Demo Mode] Training simulation complete!")
    print("To run actual training, ensure the TenSafe server is running.")


if __name__ == "__main__":
    main()
