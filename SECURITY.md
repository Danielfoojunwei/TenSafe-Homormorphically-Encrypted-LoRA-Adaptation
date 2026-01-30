# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in TenSafe, please report it responsibly:

1. **DO NOT** create a public GitHub issue
2. Email security concerns to the maintainers directly
3. Include detailed steps to reproduce the issue
4. Allow reasonable time for a fix before public disclosure

## Key Management

### NEVER Commit Secrets

The following should **NEVER** be committed to version control:

- Private keys (`*.key`, `*.pem`, `*.p12`)
- API keys and tokens
- Database credentials
- Environment files with secrets (`.env`)
- Any file in the `keys/` directory

### Development Key Generation

For development and testing, generate keys locally:

```bash
# Generate identity keys for testing
python -c "
from tensafe.identity.keys.provider import FileKeyProvider
provider = FileKeyProvider('./keys/identity')
key_id = provider.generate_key()
print(f'Generated test key: {key_id}')
"

# Generate N2HE keys for aggregation testing
python -c "
from tensafe.core.crypto import N2HEContext
ctx = N2HEContext()
print('N2HE context initialized with fresh keys')
"
```

### Production Key Management

For production deployments:

1. Use a proper Key Management System (KMS):
   - AWS KMS
   - Google Cloud KMS
   - HashiCorp Vault
   - Hardware Security Modules (HSM)

2. Rotate keys regularly
3. Use separate keys for each environment
4. Enable audit logging for all key operations

## Cryptographic Notices

### Post-Quantum Cryptography (PQC)

**IMPORTANT**: The PQC implementations in `src/tensafe/crypto/pqc/` are
**SIMULATORS ONLY** and provide **NO ACTUAL SECURITY**. They are included for:

- API compatibility testing
- Performance benchmarking
- Integration development

For production PQC, integrate with:
- liboqs (Open Quantum Safe)
- NIST PQC finalist implementations
- Hardware-backed PQC when available

### Custom Cryptography

The N2HE (Noise-Tolerant Homomorphic Encryption) implementation in
`src/tensafe/core/crypto.py` is a research prototype. For production:

1. Obtain third-party cryptographic audit
2. Use constant-time implementations
3. Implement proper side-channel protections
4. Consider using established libraries (SEAL, OpenFHE)

## Serialization Security

### Avoid Pickle

Never use `pickle` for untrusted data. TenSafe uses:
- `msgpack` for safe binary serialization
- `json` for human-readable formats
- Protocol Buffers for structured data (where applicable)

If you must handle legacy pickle files, use `safetensors` or similar
restricted unpicklers.

## Network Security

### TLS Requirements

All production deployments should:
- Use TLS 1.3 (minimum TLS 1.2)
- Enable certificate verification
- Use strong cipher suites
- Implement certificate pinning for critical paths

### API Authentication

- Use short-lived JWT tokens
- Implement proper token rotation
- Rate limit authentication endpoints
- Log all authentication failures

## Compliance

TenSafe's evidence fabric supports compliance documentation for:
- SOC 2 Type II
- ISO 27001
- NIST CSF

However, achieving compliance requires:
- Proper operational controls
- Security monitoring
- Incident response procedures
- Regular audits

The software alone does not guarantee compliance.
