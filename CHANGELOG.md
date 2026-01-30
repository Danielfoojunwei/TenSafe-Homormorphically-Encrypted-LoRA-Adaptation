# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-01-30

### Added

- **Error Taxonomy**: Unified error hierarchy with machine-readable codes (`TG_*` prefix)
- **Structured Logging**: JSON output for production, human-readable for development
- **Sensitive Data Filtering**: Automatic redaction of passwords, tokens, keys in logs
- **Feature Maturity Matrix**: Clear documentation of production-ready vs experimental features
- **Platform API Spec**: Comprehensive OpenAPI documentation
- **Server Smoke Tests**: 11 integration tests for platform endpoints
- **Crypto Tamper Tests**: 13 tests verifying AEAD tamper resistance
- **Pre-release Verification Script**: Automated quality gates before release

### Changed

- **README**: Added feature maturity warnings and package name clarifications
- **N2HE**: Clear marking of ToyN2HEScheme as non-production (NO security)
- **Dependencies**: Updated to use version ranges for flexibility

### Security

- **Error Messages**: No longer leak sensitive paths, hashes, or key material
- **AEAD Binding**: All encrypted payloads bind manifest/recipients hash to AAD
- **Nonce Uniqueness**: Verified through automated tests
- **Input Validation**: Strict validation on all API endpoints

### Fixed

- **Pyright Errors**: Resolved all type checking errors
- **Ruff Warnings**: Fixed 65+ linting issues
- **Test Isolation**: Server smoke tests properly isolate database

## [2.0.0] - 2025-12-01

### Added

- N2HE homomorphic encryption integration
- Post-quantum cryptography (ML-KEM, ML-DSA)
- TGSP secure packaging format
- Compliance evidence framework

### Changed

- Migrated from Flask to FastAPI
- Updated to Pydantic v2
- Restructured package layout

## [1.0.0] - 2025-06-01

### Added

- Initial release
- Differential privacy (DP-SGD) support
- Encrypted artifact storage
- Hash-chain audit logging
- Training client SDK
