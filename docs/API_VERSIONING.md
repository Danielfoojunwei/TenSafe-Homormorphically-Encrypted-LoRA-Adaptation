# TenSafe API Versioning and Deprecation Policy

**Effective Date:** 2026-02-03
**Last Updated:** 2026-02-03

---

## Overview

TenSafe is committed to providing a stable, reliable API while continuously improving our platform. This document outlines our API versioning strategy and deprecation policies.

## Versioning Strategy

### URL Path Versioning

TenSafe uses URL path versioning for all API endpoints:

```
https://api.tensafe.io/api/v1/training_clients
https://api.tensafe.io/api/v2/training_clients  (future)
```

**Current Version:** `v1`

### Version Lifecycle

| Status | Description | Support Level |
|--------|-------------|---------------|
| **Current** | Latest stable version | Full support, active development |
| **Supported** | Previous stable version | Bug fixes and security patches |
| **Deprecated** | Scheduled for removal | Security patches only, migration warnings |
| **Sunset** | No longer available | Requests return 410 Gone |

### Semantic Versioning (SDK)

Our SDKs follow Semantic Versioning 2.0.0:

```
MAJOR.MINOR.PATCH (e.g., 4.0.0)
```

- **MAJOR**: Breaking changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

---

## Backwards Compatibility Guarantees

### What We Guarantee

For any supported API version, we guarantee:

1. **Existing endpoints will continue to work** at the same URL
2. **Required request parameters will not be added** to existing endpoints
3. **Response field types will not change** (string stays string)
4. **Existing response fields will not be removed**
5. **Error codes will remain consistent**

### What May Change (Non-Breaking)

The following changes are considered non-breaking and may occur without a version bump:

1. **New optional request parameters** with sensible defaults
2. **New response fields** (clients should ignore unknown fields)
3. **New endpoints** added to the API
4. **New error codes** for new error conditions
5. **Documentation improvements**
6. **Performance improvements**

### Breaking Changes (Require New Version)

The following require a new major API version:

1. Removing or renaming endpoints
2. Removing or renaming request/response fields
3. Changing field types
4. Adding required request parameters
5. Changing authentication mechanisms
6. Changing error response structure

---

## Deprecation Process

### Timeline

| Phase | Duration | Actions |
|-------|----------|---------|
| **Announcement** | Day 0 | Deprecation notice published |
| **Warning Period** | 6 months | Deprecation headers sent, docs updated |
| **Migration Period** | 6 months | Reduced support, urgent migration recommended |
| **Sunset** | Day 365+ | Endpoint returns 410 Gone |

**Minimum Notice:** 12 months before sunset for any breaking change.

### Deprecation Headers

Deprecated endpoints return the following headers:

```http
Deprecation: true
Sunset: Sat, 01 Feb 2027 00:00:00 GMT
Link: <https://docs.tensafe.io/migration/v1-to-v2>; rel="deprecation"
X-Deprecation-Notice: This endpoint is deprecated. Please migrate to /api/v2/training_clients by 2027-02-01.
```

### Notification Channels

Deprecation announcements are published through:

1. **API Response Headers** (automatic)
2. **Developer Portal** (https://developers.tensafe.io)
3. **Changelog** (https://docs.tensafe.io/changelog)
4. **Email** (to account owners and API key owners)
5. **Status Page** (https://status.tensafe.io)
6. **GitHub Releases** (for SDK deprecations)

---

## Migration Support

### Migration Guides

For each breaking change, we provide:

1. **Detailed migration guide** with step-by-step instructions
2. **Code examples** in all supported languages
3. **Automated migration tools** where possible
4. **Side-by-side comparison** of old vs new API

### Migration Assistance

| Tier | Support Level |
|------|---------------|
| **Free** | Documentation only |
| **Pro** | Email support, migration guide |
| **Business** | Dedicated migration call, priority support |
| **Enterprise** | Dedicated engineer, custom migration timeline |

### Compatibility Mode

Where technically feasible, we offer compatibility shims:

```python
# Example: Compatibility layer for v1 -> v2 migration
from tensafe import TenSafeClient

# Enable v1 compatibility mode
client = TenSafeClient(
    api_key="...",
    compatibility_mode="v1"  # Translates v1 calls to v2
)
```

---

## Current API Versions

### v1 (Current)

**Status:** Current
**Released:** 2025-06-01
**Supported Until:** At least 2027-06-01

**Base URL:** `https://api.tensafe.io/api/v1/`

**Features:**
- Training client management
- Differential privacy (DP-SGD)
- Encrypted inference (HE-LoRA)
- TGSP secure packaging
- Audit logging

### v2 (Planned)

**Status:** In Development
**Expected Release:** Q3 2026
**Features:**
- GraphQL API option
- Enhanced streaming
- Batch operations
- Improved error responses

---

## API Changelog

### v1.4.0 (2026-02-03)

**Added:**
- `/api/v1/webhooks` - Webhook management endpoints
- `/api/v1/usage` - Usage metering endpoints
- `/api/v1/status` - System status endpoints
- `/playground` - Interactive API playground
- SSO/OIDC authentication support

**Changed:**
- Enhanced OpenAPI documentation with examples
- Improved error messages with request IDs

### v1.3.0 (2026-01-30)

**Added:**
- vLLM backend integration
- Ray Train distributed training
- KEDA auto-scaling support

### v1.2.0 (2025-12-01)

**Added:**
- Post-quantum cryptography support
- TGSP secure packaging

### v1.1.0 (2025-09-01)

**Added:**
- HE-LoRA encrypted inference
- Kubernetes deployment support

### v1.0.0 (2025-06-01)

**Initial Release:**
- Training client API
- Differential privacy (DP-SGD)
- Encrypted checkpoints
- Audit logging

---

## SDK Versions

### Python SDK (`tensafe`)

| Version | Python | API | Status |
|---------|--------|-----|--------|
| 4.x | 3.9+ | v1 | Current |
| 3.x | 3.8+ | v1 | Supported |
| 2.x | 3.7+ | v1 | Deprecated |

### TypeScript SDK (`@tensafe/sdk`)

| Version | Node.js | API | Status |
|---------|---------|-----|--------|
| 1.x | 18+ | v1 | Current |

---

## Best Practices for API Consumers

### Recommended Practices

1. **Always specify API version explicitly**
   ```python
   client = TenSafeClient(api_version="v1")
   ```

2. **Handle unknown response fields gracefully**
   ```python
   # Good: Ignore unknown fields
   name = response.get("name", "unknown")

   # Bad: Strict parsing that fails on new fields
   ```

3. **Subscribe to deprecation notifications**
   - Add your email to the developer portal
   - Monitor the `Deprecation` response header

4. **Test against staging before production**
   ```python
   client = TenSafeClient(
       base_url="https://api.staging.tensafe.io"
   )
   ```

5. **Pin SDK versions in production**
   ```
   # requirements.txt
   tensafe==4.0.0
   ```

### Monitoring for Deprecations

```python
import tensafe

client = tensafe.TenSafeClient(api_key="...")

# Check for deprecation warnings
response = client.training_clients.list()
if response.deprecation_warning:
    logger.warning(
        f"API deprecation: {response.deprecation_warning}"
    )
```

---

## Contact

**Questions about versioning?**
- Email: api-support@tensafe.io
- Developer Portal: https://developers.tensafe.io
- GitHub Issues: https://github.com/tensafe/tensafe/issues

**Report breaking changes:**
- Email: api-breaking@tensafe.io (monitored 24/7)
