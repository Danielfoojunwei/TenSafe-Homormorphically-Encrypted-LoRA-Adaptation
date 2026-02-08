# Security Policy

## Supported Versions

Only the latest major version of TenSafe (v4.x) is currently supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 4.x     | :white_check_mark: |
| 3.x     | :x:                |
| < 3.x   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you find a security vulnerability, please report it privately.

**Do not open a GitHub issue for security vulnerabilities.**

Please send an email to security@tensafe.dev with:
- A description of the vulnerability
- Reproduction steps or a proof-of-concept
- The potential impact

We will acknowledge your report and provide a timeline for a fix.

## Security Controls

TenSafe employs several core security layers:
- **Zero-Rotation (MOAI)**: Cryptographic enforcement of the rotation-free HE contract.
- **Evidence Fabric**: Hardware-based TEE attestation.
- **Post-Quantum Signatures**: Hybrid Dilithium3 + classical signature schemes.
- **Audit Trails**: Hash-chained, tamper-evident logs.
