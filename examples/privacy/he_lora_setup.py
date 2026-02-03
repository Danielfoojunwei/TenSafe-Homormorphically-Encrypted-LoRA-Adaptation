"""
HE-LoRA Setup Example

Demonstrates setting up Homomorphically Encrypted LoRA (HE-LoRA) for
private inference where model weights remain encrypted.

This provides cryptographic privacy guarantees - user activations never
see plaintext LoRA weights.

Requirements:
- TenSafe account and API key
- TGSP package with encrypted LoRA adapter

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python he_lora_setup.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Step 1: Load TGSP adapter (encrypted LoRA)
    print("Step 1: Loading TGSP adapter...")
    adapter = client.tgsp.load_adapter(
        tgsp_path="/path/to/adapter.tgsp",
        # Optional: recipient key for multi-recipient encryption
        recipient_key_path=None,
    )
    print(f"  Adapter ID: {adapter.adapter_id}")
    print(f"  Model: {adapter.model_name} v{adapter.model_version}")
    print(f"  Signature verified: {adapter.signature_verified}")
    print(f"  LoRA rank: {adapter.lora_rank}")

    # Step 2: Activate adapter for inference
    print("\nStep 2: Activating adapter...")
    client.tgsp.activate_adapter(adapter.adapter_id)
    print("  Adapter activated!")

    # Step 3: Run encrypted inference
    print("\nStep 3: Running HE-LoRA inference...")

    # Input activations (from base model's attention layer)
    # In practice, these come from the base model forward pass
    sample_activations = [[0.1, 0.2, 0.3, 0.4, 0.5] * 100]  # 500-dim

    result = client.tgsp.inference(
        inputs=sample_activations,
        module_name="q_proj",  # Target module
    )

    print(f"  Inference time: {result.inference_time_ms:.2f}ms")
    print(f"  TGSP compliant: {result.tgsp_compliant}")
    print(f"  Output shape: {len(result.outputs)}x{len(result.outputs[0])}")

    # Step 4: View HE metrics
    if result.he_metrics:
        print("\nHE Performance Metrics:")
        print(f"  Scheme: {result.he_metrics.get('scheme', 'CKKS')}")
        print(f"  Rotations: {result.he_metrics.get('rotations', 0)}")
        print(f"  Multiplications: {result.he_metrics.get('multiplications', 0)}")

    # Step 5: List all loaded adapters
    print("\nLoaded adapters:")
    adapters = client.tgsp.list_adapters()
    for a in adapters:
        status = "ACTIVE" if a.is_active else "loaded"
        print(f"  - {a.adapter_id} ({status}): {a.forward_count} inferences")


if __name__ == "__main__":
    main()
