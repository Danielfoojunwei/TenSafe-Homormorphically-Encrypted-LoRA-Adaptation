"""
TGSP Verification Example

Demonstrates verifying TGSP package integrity and signatures.

Requirements:
- TenSafe account and API key
- TGSP package to verify

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python tgsp_verification.py
"""

import os
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    tgsp_path = "/path/to/adapter.tgsp"

    # Verify TGSP package
    print(f"Verifying TGSP: {tgsp_path}")
    print("=" * 50)

    result = client.tgsp.verify(
        tgsp_path=tgsp_path,
        verify_signatures=True,
        verify_hashes=True,
        check_revocation=True,
    )

    print(f"Verification Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print()
    print("Details:")
    print(f"  Manifest hash valid: {result.manifest_hash_valid}")
    print(f"  Payload hash valid: {result.payload_hash_valid}")
    print(f"  Classical signature valid: {result.classical_signature_valid}")
    print(f"  PQC signature valid: {result.pqc_signature_valid}")
    print(f"  Certificate not revoked: {not result.is_revoked}")
    print()
    print("Package Info:")
    print(f"  Model: {result.model_name} v{result.model_version}")
    print(f"  Created: {result.created_at}")
    print(f"  Signer: {result.signer_id}")

    if not result.is_valid:
        print()
        print("ERRORS:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
