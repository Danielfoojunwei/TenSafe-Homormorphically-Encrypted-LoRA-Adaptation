# Crypto Hygiene and Tamper Resistance Review

**Date**: 2026-02-08
**Status**: PASS (v4.1.0 Security Compliance)

## Threat Model

### Assets Protected
1. **Model weights** - Encrypted in TGSP packages
2. **LoRA adapters** - Encrypted during training and storage
3. **Training data** - Never stored, processed via DP-SGD
4. **Audit logs** - Hash-chained for integrity

### Attackers
1. **Network attacker** - Can observe/modify traffic
2. **Storage attacker** - Has access to encrypted artifacts
3. **Insider threat** - Has legitimate access to some systems

- **Confidentiality**: Only authorized recipients can decrypt payloads
- **Integrity**: Any modification to ciphertext/AAD is detected
- **Authenticity**: Payloads are bound to their manifest and recipients
- **Zero-Rotation (MOAI)**: No sensitive information leaks via rotations (Enforced)
- **Remote Attestation**: Hardware-backed TEE evidence for all HAS operations

## Cryptographic Primitives

### Symmetric Encryption
| Use Case | Algorithm | Key Size | Mode |
|----------|-----------|----------|------|
| TGSP payloads | ChaCha20-Poly1305 | 256-bit | AEAD |
| TG-Tinker artifacts | AES-256-GCM | 256-bit | AEAD |
| N2HE key bundles | AES-256-GCM | 256-bit | AEAD |
| Evidence Bundles | AES-256-GCM | 256-bit | AEAD |

### Asymmetric Encryption
| Use Case | Algorithm | Security Level |
|----------|-----------|----------------|
| TGSP recipients | X25519 + HPKE | 128-bit (classical) |
| PQC hybrid | ML-KEM-768 | 128-bit (post-quantum) |

### Digital Signatures
| Use Case | Algorithm | Security Level |
|----------|-----------|----------------|
| TGSP manifests | Ed25519 | 128-bit (classical) |
| PQC hybrid | ML-DSA-65 | 128-bit (post-quantum) |

## Randomness Sources

### Verified Secure Usage
```python
# Cryptographic keys
key = secrets.token_bytes(32)  # ✅ CSPRNG

# Nonces
nonce = os.urandom(12)  # ✅ CSPRNG

# IDs
package_id = secrets.token_hex(8)  # ✅ CSPRNG
```

### Non-Cryptographic Usage (OK)
```python
# Benchmark data generation - gated by TENSAFE_TOY_HE
# Simulation Backend - Security Compliant
plaintext = microkernel.compute(...)  # ✅ Contract enforced
rotations = microkernel.rotate(...)   # ❌ Raises RuntimeError
```

## AEAD Binding Invariants

### TGSP Payload Encryption
```python
# AAD binds ciphertext to:
aad = manifest_hash + recipients_hash + chunk_index

# This ensures:
# 1. Payload cannot be moved to different manifest
# 2. Payload cannot be redirected to different recipients
# 3. Payload chunks cannot be reordered
```

### TG-Tinker Artifact Encryption
```python
# AAD binds ciphertext to:
aad = f"{artifact_id}|{tenant_id}|{training_client_id}"

# This ensures:
# 1. Artifacts cannot be swapped between tenants
# 2. Artifacts cannot be moved between training clients
```

## Tamper Resistance Tests

### Test Coverage
| Attack Vector | Test | Result |
|---------------|------|--------|
| Bit flip in ciphertext | `test_bitflip_in_ciphertext_fails` | PASS |
| Truncated ciphertext | `test_truncated_ciphertext_fails` | PASS |
| Wrong manifest hash | `test_wrong_manifest_hash_fails` | PASS |
| Wrong recipients hash | `test_wrong_recipients_hash_fails` | PASS |
| Wrong decryption key | `test_wrong_key_fails` | PASS |
| AES-GCM bit flip | `test_aesgcm_bitflip_fails` | PASS |
| AES-GCM wrong AAD | `test_aesgcm_wrong_aad_fails` | PASS |
| ChaCha20-Poly1305 bit flip | `test_chacha20poly1305_bitflip_fails` | PASS |
| Nonce uniqueness | `test_nonce_uniqueness` | PASS |

### Run Tests
```bash
pytest tests/security/test_crypto_tamper.py -v
```

## Version Compatibility

### TGSP Format
- Current version: `1.0`
- Version field in manifest: `tgsp_version`
- Unsupported versions should fail with clear error

### Serialization Formats
| Format | Version Field | Compatibility |
|--------|---------------|---------------|
| TGSP manifest | `tgsp_version` | Strict |
| N2HE keys | `provider_version` | Strict |
| Artifacts | `artifact_type` | Backward compatible |

## Security Invariants

### MUST Hold
1. **All randomness for crypto uses `secrets` or `os.urandom`**
2. **All AEAD operations include meaningful AAD**
3. **Nonces are never reused with the same key**
4. **Version mismatches fail closed**
5. **Tampered ciphertext raises clear exceptions**
6. **Zero-Rotation contract is strictly enforced in microkernel**
7. **TEE Evidence is attached to every Compute step**

### MAY Be Relaxed (Test/Dev Only)
1. `np.random` usage in toy HE mode (gated by `TENSAFE_TOY_HE=1`)
2. Deterministic test fixtures for reproducibility

## Recommendations

### Immediate
- ✅ All tamper tests pass
- ✅ Proper randomness in all crypto paths
- ✅ AAD binding in all AEAD operations

### Future Work
1. Add version negotiation protocol for TGSP
2. Implement key rotation for long-lived bundles
3. Add certificate pinning for production deployments
4. Consider hardware security module (HSM) integration
