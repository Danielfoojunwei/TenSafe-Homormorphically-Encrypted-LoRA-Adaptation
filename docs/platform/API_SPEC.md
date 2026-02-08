# TenSafe Platform API Specification

**Version**: 4.1.0
**API Version**: v1

## Overview

The TG-Tinker Platform provides a privacy-first ML training API built on FastAPI.

## Base URL

```
http://localhost:8000
```

## Health Endpoints

### GET /health

Health check with detailed status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-08T09:00:00.000000",
  "version": "4.1.0",
  "environment": "development",
  "checks": {
    "database": {
      "status": "healthy"
    }
  }
}
```

### GET /ready

Kubernetes readiness probe.

**Response (200)**:
```json
{"ready": true}
```

**Response (503)**:
```json
{"ready": false, "reason": "database unavailable"}
```

### GET /live

Kubernetes liveness probe.

**Response**:
```json
{"alive": true}
```

### GET /version

Service version information.

**Response**:
```json
{
  "service": "TenSafe",
  "version": "4.1.0",
  "api_version": "v1",
  "python_version": "3.9+",
  "environment": "development"
}
```

## Training API

Base path: `/api/v1`

### Training Clients

#### POST /api/v1/training_clients

Create a new training client.

#### GET /api/v1/training_clients/{client_id}

Get training client details.

#### DELETE /api/v1/training_clients/{client_id}

Terminate a training client.

### Training Operations

#### POST /api/v1/training_clients/{client_id}/forward_backward

Execute forward-backward pass.

#### POST /api/v1/training_clients/{client_id}/optim_step

Execute optimizer step with optional DP noise.

#### POST /api/v1/training_clients/{client_id}/sample

Sample from the model.

### State Management

#### POST /api/v1/training_clients/{client_id}/save_state

Save training checkpoint.

#### POST /api/v1/training_clients/{client_id}/load_state

Load training checkpoint.

### Attestation & Hardware Security (NEW)

#### GET /api/v1/attestation/quote
Get a TEE hardware quote for the current server instance.

#### POST /api/v1/attestation/verify
Verify a cryptographic evidence bundle.

## Security

### Headers

All responses include security headers:

| Header | Value |
|--------|-------|
| X-Content-Type-Options | nosniff |
| X-Frame-Options | DENY |
| X-XSS-Protection | 1; mode=block |
| Referrer-Policy | strict-origin-when-cross-origin |

In production, also includes:
| Strict-Transport-Security | max-age=31536000; includeSubDomains |

### CORS

CORS is configured via environment variables:

- `TG_ALLOWED_ORIGINS`: Comma-separated list of allowed origins
- `TG_ALLOW_CREDENTIALS`: Set to "true" to allow credentials

Default (development): localhost:3000, localhost:5173

### Authentication

API authentication is enforced via API keys:

```bash
curl -H "Authorization: Bearer <api-key>" http://localhost:8000/api/v1/...
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| TG_ENVIRONMENT | Environment (development/production) | development |
| TG_ALLOWED_ORIGINS | CORS allowed origins | localhost:3000 |
| TG_ALLOW_CREDENTIALS | Allow CORS credentials | false |
| TG_ENABLE_SECURITY_HEADERS | Enable security headers | true |
| DATABASE_URL | Database connection URL | sqlite:///./tgsp.db |
| PORT | Server port | 8000 |

## Running the Server

```bash
# Development
uvicorn tensorguard.platform.main:app --reload

# Production
uvicorn tensorguard.platform.main:app --host 0.0.0.0 --port 8000

# With environment variables
TG_ENVIRONMENT=production \
TG_ALLOWED_ORIGINS="https://app.example.com" \
uvicorn tensorguard.platform.main:app
```

## OpenAPI Documentation

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
