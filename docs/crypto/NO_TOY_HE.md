# NO TOY HE Policy

## Overview (v4.1.0 Update)

This document establishes the hard rule against using insecure "toy" or "simulated" homomorphic encryption implementations in TenSafe production code. 

**IMPORTANT**: Since v4.1.0, the "Toy HE" bottleneck has been resolved by implementing the **MOAI Zero-Rotation Security Contract**. The system now uses a **Security-Compliant Simulation** that enforces strict cryptographic invariants (Zero Rotations) even in simulation mode.

## Non-Negotiable Rules

### 1. No Ciphertext as Tensor/Array

**FORBIDDEN:**
```python
# BAD: Ciphertext implemented as torch.Tensor
class Ciphertext:
    def __init__(self, data: torch.Tensor):
        self.data = data  # NOT ALLOWED

# BAD: Ciphertext implemented as numpy array
class Ciphertext:
    def __init__(self, data: np.ndarray):
        self.data = data  # NOT ALLOWED
```

**REQUIRED:**
```python
# GOOD: Ciphertext wraps real HE library objects
class Ciphertext:
    def __init__(self, seal_ciphertext: seal.Ciphertext):
        self._ct = seal_ciphertext  # Real SEAL ciphertext
```

### 2. No Debug Plaintext Fallback

**FORBIDDEN:**
```python
# BAD: Conditional bypass
def encrypt(self, plaintext):
    if not he_enabled:
        return plaintext  # NOT ALLOWED
    return self._real_encrypt(plaintext)

# BAD: Environment variable bypass
def encrypt(self, plaintext):
    if os.environ.get("DEBUG_HE"):
        return MockCiphertext(plaintext)  # NOT ALLOWED
```

**REQUIRED:**
```python
# GOOD: Always use real encryption
def encrypt(self, plaintext):
    if not self._he_backend.is_available():
        raise HEBackendNotAvailableError(
            "N2HE-HEXL backend is required. Install with: ./scripts/build_n2he_hexl.sh"
        )
    return self._he_backend.encrypt(plaintext)
```

### 3. No Stubs That Pretend to Encrypt

**FORBIDDEN:**
```python
# BAD: Encryption that doesn't encrypt
def encrypt(self, plaintext):
    return Ciphertext(plaintext.copy())  # Just copies data

# BAD: Add random noise to pretend it's encrypted
def encrypt(self, plaintext):
    noise = np.random.randn(*plaintext.shape) * 1e-10
    return Ciphertext(plaintext + noise)  # NOT REAL ENCRYPTION
```

**REQUIRED:**
```python
# GOOD: Real CKKS encryption
def encrypt(self, plaintext):
    encoded = self._encoder.encode(plaintext, self._scale)
    ciphertext = seal.Ciphertext()
    self._encryptor.encrypt(encoded, ciphertext)
    return CKKSCiphertext(ciphertext, self._context)
```

## Enforcement

### CI/CD Checks

The following patterns MUST cause CI to fail:

1. Any `class Ciphertext` that inherits from or wraps `torch.Tensor` or `np.ndarray`
2. Any `encrypt()` function that returns tensor/ndarray types
3. Any conditional like `if not he_enabled:` followed by plaintext return
4. Any `ToyN2HEScheme` that does not enforce the **Zero-Rotation (MOAI) Security Contract**.

### Guard Test

The file `tests/crypto/test_no_toy_he.py` implements automated detection of these patterns.

### Import-Time Verification

On importing `tensorguard.n2he` or `tensafe.he_lora`:

1. Verify N2HE-HEXL library is compiled and loadable
2. Verify CKKS context can be created with production parameters
3. Verify Galois keys exist for rotation operations
4. Log actual CKKS parameters (ring degree, modulus chain)

If verification fails, the import MUST raise an exception. No fallback allowed.

## Exceptions

## The MOAI Compliance Milestone (Reached in v4.1)

TenSafe v4.1.0 achieves the "No Toy HE" goal by pivoting to **ROTATION-MINIMAL** execution. Because the MOAI optimization ensures zero rotations in the critical path, the `SimulationBackend` can now perfectly mimic the production security model by simply **PROHIBITING** rotation operations.

- **Simulation Mode**: Strictly enforces Zero-Rotation. Any attempt to rotate ciphertext raises a `RuntimeError`.
- **Production Mode (N2HE)**: Uses native hardware acceleration for the same zero-rotation circuits.

## Rationale

"Toy" HE implementations provide a false sense of security:
- They pass tests but provide zero privacy
- They enable bugs where encrypted paths are silently bypassed
- They make it impossible to verify the system actually uses HE

By enforcing real HE backends, we ensure:
- Privacy guarantees are actually enforced
- Performance benchmarks reflect real HE costs
- Security audits can verify cryptographic properties

## References

- [MOAI Paper](https://eprint.iacr.org/2025/991) - Module-Optimizing Architecture for Secure Inference
- [Microsoft SEAL](https://github.com/microsoft/SEAL) - CKKS implementation
- [Intel HEXL](https://github.com/intel/hexl) - HE acceleration library
- [N2HE](https://github.com/HintSight-Technology/N2HE) - Neural Network HE optimization
