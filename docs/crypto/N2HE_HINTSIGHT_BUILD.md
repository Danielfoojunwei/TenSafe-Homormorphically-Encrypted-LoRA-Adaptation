# HintSight N2HE Build Guide

## Overview

HintSight N2HE is an alternative homomorphic encryption backend for TenSafe that uses FasterNTT for polynomial operations instead of Intel HEXL. This makes it portable across different CPU architectures (not Intel-specific).

**Repository:** https://github.com/HintSight-Technology/N2HE

## Key Features

- **LWE-based encryption** for weighted sums and convolutions
- **FHEW ciphertexts** for non-polynomial activation functions
- **FasterNTT** for fast polynomial multiplication
- **Cross-platform** - works on Intel, AMD, ARM CPUs
- **No Intel HEXL dependency**

## Prerequisites

- **CMake** 3.16 or later
- **GCC** 9+ or Clang 10+ with C++17 support
- **Python** 3.9+ with pip
- **OpenSSL** 3.2.1+ (with development headers)
- **Git** for cloning dependencies

### Installing Prerequisites

```bash
# Ubuntu/Debian
apt-get install cmake build-essential python3-dev libssl-dev git

# macOS
brew install cmake openssl python3
```

## Quick Start

```bash
# Build HintSight N2HE
./scripts/build_n2he_hintsight.sh

# Verify installation
python scripts/verify_hintsight_backend.py
```

## Build Options

```bash
# Clean rebuild
./scripts/build_n2he_hintsight.sh --clean

# Build and run N2HE tests
./scripts/build_n2he_hintsight.sh --test
```

## What Gets Installed

### Dependencies (third_party/)

```
third_party/
└── N2HE-HintSight/          # Cloned from GitHub
    ├── include/             # Header files
    ├── build/               # Build artifacts
    │   ├── test             # Test executable
    │   └── LUT_test         # LUT test executable
    └── ...
```

### Build Artifacts

```
crypto_backend/
└── n2he_hintsight/
    ├── __init__.py              # Python wrapper
    └── lib/
        └── n2he_hintsight_native.*.so  # Native module
```

## LWE Parameters

Default parameters optimized for LoRA computation:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 1024 | Lattice dimension |
| q | 2^32 | Ciphertext modulus |
| t | 2^16 | Plaintext modulus |
| std_dev | 3.2 | Gaussian noise |
| Security | 128-bit | NIST level |

Higher security parameters available:

| Parameter | Value | Description |
|-----------|-------|-------------|
| n | 2048 | Lattice dimension |
| q | 2^54 | Ciphertext modulus |
| t | 2^20 | Plaintext modulus |
| Security | 192-bit | Higher security |

## Verification

After building, verify the backend:

```bash
$ python scripts/verify_hintsight_backend.py

============================================================
HintSight N2HE Backend Verification
============================================================

Backend: HintSight-N2HE
Available: True

LWE Parameters:
  Lattice Dimension (n): 1024
  Ciphertext Modulus (q): 2^32
  Plaintext Modulus (t): 2^16
  Noise Std Dev: 3.2
  Security Level: 128 bits

Encrypt/Decrypt Test:
  Input:  [1.0, 2.0, 3.0, 4.0]
  Output: ['1.002', '1.998', '3.001', '4.003']
  Max Error: 0.003
  Passed: True

LoRA Delta Test:
  Input Dim: 4
  Rank: 8
  Scaling: 0.5
  Max Error: 0.02
  Passed: True

SUCCESS: HintSight N2HE backend is properly installed and functional!
```

## Comparison with N2HE-HEXL

| Feature | HintSight N2HE | N2HE-HEXL |
|---------|---------------|-----------|
| **Polynomial Math** | FasterNTT | Intel HEXL |
| **CPU Support** | Any (Intel, AMD, ARM) | Intel only |
| **HE Scheme** | LWE/FHEW | CKKS |
| **Precision** | Lower (integer) | Higher (approximate) |
| **Speed** | Good | Faster on Intel |
| **Memory** | Lower | Higher |

## Troubleshooting

### "HintSight N2HE native module not available"

The native library wasn't built or isn't in the Python path.

```bash
# Rebuild
./scripts/build_n2he_hintsight.sh --clean

# Check the library exists
ls crypto_backend/n2he_hintsight/lib/n2he_hintsight_native*.so
```

### CMake can't find OpenSSL

```bash
# Ubuntu/Debian
apt-get install libssl-dev

# macOS - set OpenSSL path
export OPENSSL_ROOT_DIR=$(brew --prefix openssl)
./scripts/build_n2he_hintsight.sh --clean
```

### Missing pybind11

```bash
pip install pybind11[global]
```

## Integration with TenSafe

Once built, the backend is available via:

```python
from crypto_backend.n2he_hintsight import N2HEHintSightBackend, verify_backend

# Verify backend
result = verify_backend()
print(result)

# Use backend
backend = N2HEHintSightBackend()
backend.setup_context()
backend.generate_keys()

# Encrypt and compute
ct = backend.encrypt(np.array([1.0, 2.0, 3.0, 4.0]))
ct_delta = backend.lora_delta(ct, lora_a, lora_b, scaling=0.5)
result = backend.decrypt(ct_delta)
```

## References

- [HintSight N2HE Repository](https://github.com/HintSight-Technology/N2HE)
- [N2HE Paper](https://ieeexplore.ieee.org/document/...) - IEEE TDSC
- [FasterNTT](https://github.com/...) - Number-Theoretic Transform library
- [pybind11](https://pybind11.readthedocs.io/)
