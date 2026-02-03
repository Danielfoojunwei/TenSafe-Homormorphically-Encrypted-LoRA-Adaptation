"""
HuggingFace Hub Integration

Demonstrates pushing and pulling models from HuggingFace Hub with TenSafe.

Requirements:
- TenSafe account and API key
- HuggingFace account and token
- pip install huggingface_hub

Usage:
    export TENSAFE_API_KEY="your-api-key"
    export HF_TOKEN="your-hf-token"
    python huggingface_hub.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Train a model
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={"rank": 16, "alpha": 32.0},
        dp_config={"enabled": True, "target_epsilon": 8.0},
    )

    # ... training loop ...
    sample_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[2, 3, 4, 5, 6]] * 8}
    for _ in range(10):
        tc.forward_backward(batch=sample_batch).result()
        tc.optim_step(apply_dp_noise=True).result()

    # Save checkpoint
    checkpoint = tc.save_state()

    # Push to HuggingFace Hub
    print("Pushing to HuggingFace Hub...")
    hub_url = client.integrations.push_to_hub(
        artifact_id=checkpoint.artifact_id,
        repo_id="your-username/my-private-lora",
        private=True,
        commit_message="DP-trained LoRA adapter (ε=8.0)",
        model_card={
            "license": "apache-2.0",
            "tags": ["lora", "differential-privacy", "tensafe"],
            "base_model": "meta-llama/Llama-3-8B",
            "privacy_budget": {"epsilon": 8.0, "delta": 1e-5},
        },
    )
    print(f"Pushed to: {hub_url}")

    # Pull from HuggingFace Hub
    print("\nPulling from HuggingFace Hub...")
    loaded_adapter = client.integrations.pull_from_hub(
        repo_id="your-username/my-private-lora",
        revision="main",
    )
    print(f"Loaded adapter: {loaded_adapter.id}")


if __name__ == "__main__":
    main()
