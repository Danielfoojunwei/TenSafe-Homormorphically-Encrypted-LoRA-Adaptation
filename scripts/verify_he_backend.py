#!/usr/bin/env python3
"""
Verify the N2HE-HEXL HE backend is properly installed and functional.

This script:
1. Checks if the native N2HE-HEXL library is available
2. Reports CKKS parameters (ring degree, modulus chain, etc.)
3. Verifies Galois keys exist for rotation operations
4. Tests basic encrypt/decrypt roundtrip

Usage:
    python scripts/verify_he_backend.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def main():
    print("=" * 60)
    print("N2HE-HEXL Backend Verification")
    print("=" * 60)
    print()

    try:
        from crypto_backend.n2he_hexl import HEBackendNotAvailableError, verify_backend

        result = verify_backend()

        print(f"Backend: {result['backend']}")
        print(f"Available: {result['available']}")
        print()

        print("CKKS Parameters:")
        params = result['params']
        print(f"  Ring Degree (N): {params['ring_degree']}")
        print(f"  Scale: 2^{params['scale_bits']} = {params['scale']:.2e}")
        print(f"  Slot Count: {params['slot_count']}")
        print(f"  Coeff Modulus Chain Length: {params['coeff_modulus_chain_length']}")
        print(f"  Coeff Modulus Sizes (bits): {params['coeff_modulus_sizes']}")
        print(f"  Has Galois Keys: {params['has_galois_keys']}")
        print(f"  Security Level: {params['security_level']} bits")
        print()

        print("Encrypt/Decrypt Test:")
        test = result['test_encrypt_decrypt']
        print(f"  Input:  {test['input']}")
        print(f"  Output: {[f'{x:.6f}' for x in test['output']]}")
        print(f"  Max Error: {test['max_error']:.2e}")
        print(f"  Passed: {test['passed']}")
        print()

        if test['passed']:
            print("SUCCESS: N2HE-HEXL backend is properly installed and functional!")
            print()
            print("JSON Output:")
            print(json.dumps(result, indent=2, default=str))
            return 0
        else:
            print("WARNING: Encrypt/decrypt error exceeds threshold")
            return 1

    except ImportError as e:
        print(f"ERROR: Failed to import N2HE-HEXL backend: {e}")
        print()
        print("The N2HE-HEXL native library is not installed.")
        print("To build it, run:")
        print()
        print("    ./scripts/build_n2he_hexl.sh")
        print()
        print("Prerequisites:")
        print("  - CMake 3.16+")
        print("  - GCC 9+ with C++17 support")
        print("  - Python 3.9+ with pybind11")
        print()
        return 1

    except Exception as e:
        print(f"ERROR: Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
