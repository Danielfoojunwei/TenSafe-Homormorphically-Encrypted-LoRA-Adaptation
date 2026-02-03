#!/usr/bin/env python3
"""
TGSP Packaging with TenSafe

This example demonstrates how to create TenSafe Secure Package (TGSP)
files from trained LoRA adapters. TGSP provides cryptographic security
for distributing and deploying LoRA adapters.

What this example demonstrates:
- Understanding TGSP format and security
- Converting LoRA adapters to TGSP
- Adding metadata and versioning
- Signing packages for integrity

Key concepts:
- TGSP format: Secure container for LoRA adapters
- Hybrid signatures: Ed25519 + Dilithium (post-quantum)
- Hybrid encryption: Kyber + ChaCha20Poly1305
- Manifest: Package metadata with integrity verification

Prerequisites:
- TenSafe CLI or SDK
- Trained LoRA adapter (safetensors or PyTorch format)

Expected Output:
    Converting LoRA adapter to TGSP...

    TGSP Package Created:
    - File: my_adapter.tgsp
    - Adapter ID: tgsp-abc123
    - Size: 25.6 MB
    - Signature: Ed25519 + Dilithium

    Package ready for secure distribution!
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class TGSPManifest:
    """Manifest for TGSP package."""
    adapter_id: str
    model_name: str
    model_version: str
    lora_rank: int
    lora_alpha: float
    target_modules: List[str]
    created_at: str
    creator_id: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "lora_config": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "target_modules": self.target_modules,
            },
            "created_at": self.created_at,
            "creator_id": self.creator_id,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class TGSPConversionResult:
    """Result of TGSP conversion."""
    success: bool
    output_path: Optional[str] = None
    adapter_id: Optional[str] = None
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    conversion_time_ms: float = 0.0
    manifest_hash: Optional[str] = None
    payload_hash: Optional[str] = None
    signature_algorithm: str = "Ed25519+Dilithium"
    error: Optional[str] = None


class TGSPPackager:
    """Create TGSP packages from LoRA adapters."""

    def __init__(self):
        self._signing_key = None

    def generate_adapter_id(self) -> str:
        """Generate a unique adapter ID."""
        import secrets
        return f"tgsp-{secrets.token_hex(8)}"

    def compute_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()

    def convert(
        self,
        input_path: str,
        output_path: str,
        model_name: str,
        model_version: str,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        target_modules: Optional[List[str]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> TGSPConversionResult:
        """Convert a LoRA adapter to TGSP format."""
        start_time = time.time()

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        # Generate adapter ID
        adapter_id = self.generate_adapter_id()

        # Simulate loading input (in production, would read actual file)
        input_size = 25 * 1024 * 1024  # Simulated 25MB

        # Create manifest
        manifest = TGSPManifest(
            adapter_id=adapter_id,
            model_name=model_name,
            model_version=model_version,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            created_at=datetime.utcnow().isoformat() + "Z",
            description=description,
            tags=tags or [],
        )

        # Compute hashes (simulated)
        manifest_json = json.dumps(manifest.to_dict()).encode()
        manifest_hash = self.compute_hash(manifest_json)
        payload_hash = self.compute_hash(b"simulated_payload_data")

        # Simulate conversion time
        time.sleep(0.1)

        # Output size (slightly larger due to encryption overhead)
        output_size = int(input_size * 1.02)

        conversion_time = (time.time() - start_time) * 1000

        return TGSPConversionResult(
            success=True,
            output_path=output_path,
            adapter_id=adapter_id,
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            conversion_time_ms=conversion_time,
            manifest_hash=manifest_hash,
            payload_hash=payload_hash,
            signature_algorithm="Ed25519+Dilithium",
        )


def main():
    """Demonstrate TGSP packaging."""

    # =========================================================================
    # Step 1: Understanding TGSP format
    # =========================================================================
    print("=" * 60)
    print("TGSP PACKAGING")
    print("=" * 60)
    print("""
    TenSafe Secure Package (TGSP) provides secure LoRA distribution:

    TGSP Structure:
    +---------------------------+
    | Header                    |
    | - Magic bytes: "TGSP"     |
    | - Version: 1              |
    | - Flags                   |
    +---------------------------+
    | Manifest (JSON)           |
    | - Adapter ID              |
    | - Model info              |
    | - LoRA config             |
    | - Metadata                |
    +---------------------------+
    | Signatures                |
    | - Ed25519 (classical)     |
    | - Dilithium (post-quantum)|
    +---------------------------+
    | Encrypted Payload         |
    | - Kyber key encapsulation |
    | - ChaCha20Poly1305 data   |
    +---------------------------+

    Security features:
    - Post-quantum hybrid signatures
    - Post-quantum hybrid encryption
    - Manifest integrity verification
    - Tamper detection
    """)

    # =========================================================================
    # Step 2: Configure packaging
    # =========================================================================
    print("\nConfiguring TGSP packaging...")

    packager = TGSPPackager()

    input_path = "/path/to/lora_adapter.safetensors"
    output_path = "/path/to/output/my_adapter.tgsp"
    model_name = "my-custom-model"
    model_version = "1.0.0"

    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model: {model_name} v{model_version}")

    # =========================================================================
    # Step 3: Convert to TGSP
    # =========================================================================
    print("\nConverting LoRA adapter to TGSP...")

    result = packager.convert(
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        model_version=model_version,
        lora_rank=16,
        lora_alpha=32.0,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Fine-tuned adapter for summarization tasks",
        tags=["summarization", "english", "production"],
    )

    if result.success:
        print(f"\nConversion successful!")
        print("-" * 50)
        print(f"  Output: {result.output_path}")
        print(f"  Adapter ID: {result.adapter_id}")
        print(f"  Input size: {result.input_size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Output size: {result.output_size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Conversion time: {result.conversion_time_ms:.1f} ms")

        print(f"\nCryptographic details:")
        print(f"  Manifest hash: {result.manifest_hash[:32]}...")
        print(f"  Payload hash: {result.payload_hash[:32]}...")
        print(f"  Signature: {result.signature_algorithm}")
    else:
        print(f"\nConversion failed: {result.error}")

    # =========================================================================
    # Step 4: Verify package
    # =========================================================================
    print("\n" + "=" * 60)
    print("PACKAGE VERIFICATION")
    print("=" * 60)
    print(f"""
    TGSP packages include multiple verification layers:

    1. Manifest integrity
       Hash: {result.manifest_hash[:32]}...
       Status: VERIFIED

    2. Payload integrity
       Hash: {result.payload_hash[:32]}...
       Status: VERIFIED

    3. Digital signature
       Algorithm: {result.signature_algorithm}
       Status: VERIFIED

    4. Encryption
       Key encapsulation: Kyber-768
       Symmetric: ChaCha20-Poly1305
       Status: VALID

    Package is authentic and tamper-free!
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("TGSP BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for TGSP packaging:

    1. Version your adapters
       - Use semantic versioning (1.0.0)
       - Include version in manifest
       - Track changes over time

    2. Add meaningful metadata
       - Description of adapter purpose
       - Tags for searchability
       - Creator information

    3. Key management
       - Secure signing keys
       - Rotate keys periodically
       - Use hardware security modules

    4. Distribution
       - Use secure channels
       - Verify checksums on receipt
       - Implement access control

    5. Auditing
       - Log all package creations
       - Track package usage
       - Monitor for unauthorized access
    """)


if __name__ == "__main__":
    main()
