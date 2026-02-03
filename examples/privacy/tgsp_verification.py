#!/usr/bin/env python3
"""
TGSP Verification with TenSafe

This example demonstrates how to verify TenSafe Secure Package (TGSP)
files before loading them. Verification ensures package integrity and
authenticity, protecting against tampering and unauthorized modifications.

What this example demonstrates:
- Verifying TGSP package signatures
- Checking manifest integrity
- Validating payload checksums
- Inspecting package metadata

Key concepts:
- Signature verification: Confirm package authenticity
- Hash verification: Ensure package hasn't been modified
- Trust chain: Verify signer identity
- Certificate validation: Check signer credentials

Prerequisites:
- TenSafe CLI or SDK
- TGSP package file

Expected Output:
    Verifying TGSP package...

    Signature Verification:
    - Ed25519: VALID
    - Dilithium: VALID

    Integrity Checks:
    - Manifest hash: VALID
    - Payload hash: VALID

    Package verification: PASSED
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class SignatureResult:
    """Result of signature verification."""
    algorithm: str
    valid: bool
    signer_id: Optional[str] = None
    signed_at: Optional[str] = None


@dataclass
class IntegrityResult:
    """Result of integrity check."""
    component: str
    expected_hash: str
    actual_hash: str
    valid: bool


@dataclass
class TGSPVerificationResult:
    """Complete verification result for a TGSP package."""
    package_path: str
    adapter_id: str
    model_name: str
    model_version: str
    signatures: List[SignatureResult]
    integrity_checks: List[IntegrityResult]
    verified: bool
    created_at: str
    creator_id: Optional[str]


class TGSPVerifier:
    """Verify TGSP package integrity and authenticity."""

    def __init__(self, trusted_keys: Optional[Dict[str, str]] = None):
        self.trusted_keys = trusted_keys or {}

    def verify(self, package_path: str) -> TGSPVerificationResult:
        """Verify a TGSP package."""
        # Simulated verification
        manifest_hash = hashlib.sha256(b"manifest_data").hexdigest()
        payload_hash = hashlib.sha256(b"payload_data").hexdigest()

        signatures = [
            SignatureResult("Ed25519", True, "signer-abc123", "2024-01-15T10:30:00Z"),
            SignatureResult("Dilithium3", True, "signer-abc123", "2024-01-15T10:30:00Z"),
        ]

        integrity_checks = [
            IntegrityResult("manifest", manifest_hash, manifest_hash, True),
            IntegrityResult("payload", payload_hash, payload_hash, True),
        ]

        all_valid = all(s.valid for s in signatures) and all(i.valid for i in integrity_checks)

        return TGSPVerificationResult(
            package_path=package_path,
            adapter_id="tgsp-abc12345",
            model_name="custom-model",
            model_version="1.0.0",
            signatures=signatures,
            integrity_checks=integrity_checks,
            verified=all_valid,
            created_at="2024-01-15T10:30:00Z",
            creator_id="creator-xyz",
        )


def main():
    """Demonstrate TGSP verification."""

    print("=" * 60)
    print("TGSP VERIFICATION")
    print("=" * 60)
    print("""
    TGSP verification ensures package security:

    1. SIGNATURE VERIFICATION
       - Ed25519: Fast classical signature
       - Dilithium: Post-quantum signature
       - Both must be valid for security

    2. INTEGRITY VERIFICATION
       - Manifest hash: Verifies metadata unchanged
       - Payload hash: Verifies adapter weights unchanged

    3. TRUST VERIFICATION
       - Check signer identity
       - Verify signing key is trusted
    """)

    print("\nInitializing TGSP verifier...")
    verifier = TGSPVerifier(trusted_keys={"signer-abc123": "trusted_public_key"})

    package_path = "/path/to/adapter.tgsp"
    print(f"  Package: {package_path}")

    print("\nVerifying TGSP package...")
    print("-" * 50)

    result = verifier.verify(package_path)

    print("\nSignature Verification:")
    for sig in result.signatures:
        status = "VALID" if sig.valid else "INVALID"
        print(f"  - {sig.algorithm}: {status}")

    print("\nIntegrity Verification:")
    for check in result.integrity_checks:
        status = "VALID" if check.valid else "INVALID"
        print(f"  - {check.component}: {status}")

    print("-" * 50)

    if result.verified:
        print(f"\nVERIFICATION: PASSED")
        print(f"""
    Package details:
    - Adapter ID: {result.adapter_id}
    - Model: {result.model_name} v{result.model_version}
    - Created: {result.created_at}
    - Creator: {result.creator_id}

    Package is safe to load!
    """)
    else:
        print("\nVERIFICATION: FAILED - Do NOT load this package!")

    print("=" * 60)
    print("VERIFICATION BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for secure TGSP handling:

    1. Always verify before loading
    2. Manage trusted keys carefully
    3. Handle verification failures by alerting security
    4. Maintain audit trail of all verifications
    """)


if __name__ == "__main__":
    main()
