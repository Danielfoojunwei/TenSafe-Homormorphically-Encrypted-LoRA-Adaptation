"""
Verification: Unified Key Management Fabric
Verifies that Identity, MOAI, and N2HE keys are all correctly using the unified vault.
"""

import os
import sys
import numpy as np
from pathlib import Path
import pytest

# Skip entire module if tenseal is not available
tenseal = pytest.importorskip("tenseal", reason="tenseal (FHE library) not installed")

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from tensorguard.core.keys import vault, KeyScope
from tensorguard.core.crypto import N2HEContext
from tensorguard.moai.keys import MoaiKeyManager, MoaiConfig
from tensorguard.agent.identity.csr_generator import CSRGenerator


def verify_unified_fabric():
    print("=== Unified Key Fabric Verification ===")

    # 1. Verify N2HE (Aggregation)
    print("\n[Scope: AGGREGATION] Testing N2HE Key")
    ctx = N2HEContext()
    ctx.generate_keys()
    ctx.save_key("test_n2he")
    print(f"  -> Path Check: {os.path.exists('keys/aggregation/test_n2he.npy.bin')}")
    print(f"  -> Meta Check: {os.path.exists('keys/aggregation/test_n2he.meta.json')}")

    # 2. Verify MOAI (Inference)
    print("\n[Scope: INFERENCE] Testing MOAI Key")
    moai_mgr = MoaiKeyManager()
    cfg = MoaiConfig(poly_modulus_degree=8192)
    key_id, _, _, _ = moai_mgr.generate_keypair("test-tenant", cfg)
    print(f"  -> Created Key ID: {key_id}")
    print(f"  -> Path Check: {os.path.exists(f'keys/inference/{key_id}.pub')}")

    # 3. Verify Identity (Identity)
    print("\n[Scope: IDENTITY] Testing RSA Key")
    # CSRGenerator handles its own key storage
    csr_gen = CSRGenerator()
    key_pair = csr_gen.generate_key(key_type="RSA", key_size=2048)
    print(f"  -> Path Check: {os.path.exists(f'keys/identity/{key_pair.key_id}.key')}")

    # 4. Global Discovery
    print("\n[Global] Testing Discovery")
    all_keys = vault.list_keys()
    print(f"  -> Total Keys in Vault: {len(all_keys)}")
    for k in all_keys:
        print(f"     - [{k['scope']}] {k['key_id']} (Alg: {k['algorithm']})")

    # Final logic check
    assert len([k for k in all_keys if k['scope'] == 'aggregation']) >= 1
    assert len([k for k in all_keys if k['scope'] == 'inference']) >= 1
    assert len([k for k in all_keys if k['scope'] == 'identity']) >= 1

    print("\nUnified Fabric Status: OPERATIONAL")


def test_unified_fabric():
    """Pytest entry point for the unified fabric verification."""
    verify_unified_fabric()


if __name__ == "__main__":
    try:
        verify_unified_fabric()
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        sys.exit(1)
