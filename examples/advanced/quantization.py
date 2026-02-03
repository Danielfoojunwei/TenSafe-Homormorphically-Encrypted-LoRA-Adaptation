"""
Model Quantization (QLoRA) Example

Demonstrates 4-bit quantization for memory-efficient training.

Requirements:
- TenSafe account and API key
- GPU with sufficient memory

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python quantization.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # QLoRA: 4-bit quantization + LoRA
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={"rank": 16, "alpha": 32.0},
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        },
        dp_config={"enabled": True, "target_epsilon": 8.0},
    )

    print("QLoRA Configuration:")
    print("  Quantization: 4-bit NF4")
    print("  Double quantization: enabled")
    print("  Estimated memory savings: ~75%")
    print()

    # Training
    sample_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[2, 3, 4, 5, 6]] * 8}

    for step in range(50):
        fb = tc.forward_backward(batch=sample_batch).result()
        tc.optim_step(apply_dp_noise=True).result()
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: loss={fb.get('loss', 0):.4f}")

    print("\nQLoRA training complete!")


if __name__ == "__main__":
    main()
