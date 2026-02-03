"""
TGSP Packaging Example

Demonstrates creating TenSafe Secure Package (TGSP) files from
standard LoRA adapters for encrypted distribution.

TGSP provides:
- Post-quantum hybrid signatures (Ed25519 + Dilithium)
- Hybrid encryption (Kyber + ChaCha20Poly1305)
- Manifest integrity verification
- Audit trail support

Requirements:
- TenSafe account and API key
- LoRA adapter files (safetensors or PyTorch format)

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python tgsp_packaging.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Convert a standard LoRA adapter to TGSP format
    print("Converting LoRA adapter to TGSP format...")
    print("=" * 50)

    result = client.tgsp.convert(
        input_path="/path/to/lora_adapter.safetensors",
        output_path="/path/to/output/adapter.tgsp",
        model_name="my-custom-model",
        model_version="1.0.0",
        validate_weights=True,  # Validate LoRA weights
        auto_generate_keys=True,  # Generate signing keys if needed
    )

    if result.success:
        print(f"Conversion successful!")
        print(f"  Output: {result.output_path}")
        print(f"  Adapter ID: {result.adapter_id}")
        print(f"  Input format: {result.input_format}")
        print(f"  Input size: {result.input_size_bytes / 1024:.1f} KB")
        print(f"  Output size: {result.output_size_bytes / 1024:.1f} KB")
        print(f"  Conversion time: {result.conversion_time_ms:.1f} ms")
        print()
        print("Cryptographic details:")
        print(f"  Manifest hash: {result.manifest_hash[:32]}...")
        print(f"  Payload hash: {result.payload_hash[:32]}...")
        print(f"  Signature key ID: {result.signature_key_id}")
        print()
        print("LoRA configuration:")
        print(f"  Rank: {result.lora_rank}")
        print(f"  Alpha: {result.lora_alpha}")
        print(f"  Target modules: {', '.join(result.target_modules)}")
    else:
        print(f"Conversion failed: {result.error}")

    # Batch conversion example
    print("\n" + "=" * 50)
    print("Batch converting multiple adapters...")

    batch_result = client.tgsp.batch_convert(
        input_paths=[
            "/path/to/adapter1.safetensors",
            "/path/to/adapter2.safetensors",
            "/path/to/adapter3/",  # HuggingFace directory
        ],
        output_dir="/path/to/output/tgsp/",
        model_version="1.0.0",
    )

    print(f"Batch conversion complete:")
    print(f"  Total: {batch_result.total}")
    print(f"  Successful: {batch_result.successful}")
    print(f"  Failed: {batch_result.failed}")


if __name__ == "__main__":
    main()
