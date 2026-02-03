"""
Post-Quantum Cryptography Example

Demonstrates using quantum-resistant cryptographic algorithms.

Requirements:
- TenSafe account and API key
- liboqs library installed

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python pqc_encryption.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Enable post-quantum cryptography
    tc = client.create_training_client(
        model_ref="meta-llama/Llama-3-8B",
        crypto_config={
            "signature_algorithm": "dilithium3",  # PQC signature
            "key_encapsulation": "kyber768",  # PQC key exchange
            "hybrid_mode": True,  # Ed25519 + Dilithium
        },
    )

    print("Post-Quantum Cryptography Enabled:")
    print("  Signatures: Dilithium3 (NIST Level 3)")
    print("  Key encapsulation: Kyber768 (NIST Level 3)")
    print("  Hybrid mode: Classical + PQC")
    print()

    # Training (encrypted with PQC)
    sample_batch = {"input_ids": [[1, 2, 3, 4, 5]] * 8, "attention_mask": [[1] * 5] * 8, "labels": [[2, 3, 4, 5, 6]] * 8}

    for step in range(20):
        tc.forward_backward(batch=sample_batch).result()
        tc.optim_step().result()

    # Save with PQC signatures
    checkpoint = tc.save_state(metadata={"pqc_enabled": True})
    print(f"Checkpoint saved with PQC signatures: {checkpoint.artifact_id}")

    # Create TGSP with PQC
    tgsp_result = client.tgsp.convert(
        input_path=f"/tmp/{checkpoint.artifact_id}",
        output_path="/tmp/pqc_adapter.tgsp",
        crypto_config={
            "signature_algorithm": "dilithium3",
            "encryption_algorithm": "kyber768_chacha20",
        },
    )
    print(f"TGSP created with PQC: {tgsp_result.output_path}")


if __name__ == "__main__":
    main()
