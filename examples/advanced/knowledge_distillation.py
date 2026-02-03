"""
Knowledge Distillation Example

Demonstrates distilling knowledge from a large teacher model to a smaller student.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python knowledge_distillation.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Create teacher (large model)
    teacher = client.create_inference_client(model_ref="meta-llama/Llama-3-70B")

    # Create student (small model to train)
    student = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        lora_config={"rank": 16, "alpha": 32.0},
        dp_config={"enabled": True, "target_epsilon": 8.0},
    )

    print("Knowledge Distillation Setup:")
    print(f"  Teacher: Llama-3-70B (frozen)")
    print(f"  Student: Llama-3-8B (training)")
    print(f"  Privacy: DP-SGD enabled")
    print()

    # Distillation training
    prompts = ["Explain quantum computing:", "What is machine learning?", "Describe blockchain:"]

    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        for i, prompt in enumerate(prompts):
            # Get teacher logits
            teacher_output = teacher.generate(prompt=prompt, max_tokens=50, return_logits=True)

            # Train student to match teacher
            result = student.distill_step(
                input_text=prompt,
                teacher_logits=teacher_output.logits,
                temperature=2.0,  # Soften distributions
                alpha=0.5,  # Balance KL-div and CE loss
            )
            print(f"  [{i+1}/{len(prompts)}] Loss: {result.loss:.4f}")

    print("\nDistillation complete!")
    print(f"Final DP: ε={student.get_dp_metrics()['total_epsilon']:.4f}")


if __name__ == "__main__":
    main()
