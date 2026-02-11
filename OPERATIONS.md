# TenSafe Operations Guide

## Environment Resolution

TenSafe uses a single environment variable `TENSAFE_ENV` to determine
runtime behavior. Accepted values:

| Value        | Behavior |
|-------------|----------|
| `local`     | SQLite fallback OK, playground enabled, auto table creation, demo mode allowed |
| `dev`       | Same as local (intended for shared dev servers) |
| `staging`   | Production-like: DATABASE_URL required, no SQLite, no demo mode, no playground |
| `production`| Full production: all safeguards enforced |
| *(missing)* | **Defaults to `production`** (fail-closed) |

Legacy `TG_ENVIRONMENT` is still read if `TENSAFE_ENV` is not set.
`development` maps to `dev`, `testing`/`test` map to `dev`.

## Required Environment Variables

### Always Required in Production/Staging

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/tensafe` |
| `TG_SECRET_KEY` | JWT signing key (min 32 bytes hex) | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `TG_ADMIN_SECRET_KEY` | Admin JWT signing key | Same generation method |

### Optional (with safe defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `TENSAFE_ENV` | `production` | Runtime environment |
| `PORT` | `8000` | API server port |
| `WORKERS` | `4` | Gunicorn worker count |
| `TG_ALLOWED_ORIGINS` | `""` (none in prod) | CORS origins, comma-separated |
| `TG_ENABLE_RATE_LIMITING` | `true` | Enable rate limiting |
| `TG_ENABLE_CSP` | `true` | Content Security Policy |
| `TG_ENABLE_INPUT_VALIDATION` | `true` | Input validation middleware |
| `TG_ENABLE_SECURITY_HEADERS` | `true` | Security response headers |
| `TG_DB_POOL_SIZE` | `10` | DB connection pool size |
| `TG_DB_MAX_OVERFLOW` | `20` | Max overflow connections |
| `TG_DB_POOL_RECYCLE` | `3600` | Connection recycle (seconds) |
| `TG_CRYPTO_MODE` | `testing` | Crypto mode: `production`, `testing`, `hybrid` |
| `TG_HE_BACKEND` | `toy_simulation` | HE backend: `n2he_native`, `tenseal`, `toy_simulation` |

### HE Backend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TG_HE_BACKEND` | `toy_simulation` | HE backend selection |
| `TENSAFE_TOY_HE` | `0` | Enable toy HE in dev (explicit opt-in: set to `1`) |
| `TENSAFE_SKIP_HE_CHECK` | `0` | Skip Docker HE self-check (set to `1` to run without HE) |

### Demo Mode (NEVER in production)

| Variable | Default | Notes |
|----------|---------|-------|
| `TG_DEMO_MODE` | `false` | Hard-fails if `true` in production/staging |
| `TG_ADMIN_DEMO_MODE` | `false` | Hard-fails if `true` in production/staging |

## Health Endpoints

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `GET /healthz` | Liveness probe (process alive) | No |
| `GET /readyz` | Readiness probe (DB connectivity) | No |
| `GET /version` | Version and feature flags | No |
| `GET /health` | Legacy full health (backward compat) | No |

## Building

```bash
# Build wheel
python -m pip install -U pip build
python -m build --wheel

# Install from wheel
pip install dist/tensafe-4.1.0-py3-none-any.whl

# Install with HE support
pip install "dist/tensafe-4.1.0-py3-none-any.whl[he]"

# Install all extras
pip install "dist/tensafe-4.1.0-py3-none-any.whl[all]"
```

## Docker

```bash
# Build
docker build -t tensafe:latest .

# Run (production)
docker run -d \
  -e DATABASE_URL=postgresql://... \
  -e TG_SECRET_KEY=... \
  -e TG_ADMIN_SECRET_KEY=... \
  -p 8000:8000 \
  tensafe:latest

# Run (development, with SQLite)
docker run -d \
  -e TENSAFE_ENV=dev \
  -p 8000:8000 \
  tensafe:latest

# Run without HE backend (explicitly disabled)
docker run -d \
  -e TENSAFE_SKIP_HE_CHECK=1 \
  -e TENSAFE_ENV=dev \
  -p 8000:8000 \
  tensafe:latest
```

## Safe Defaults Summary

- **Missing TENSAFE_ENV** → treated as `production` (fail-closed)
- **Missing DATABASE_URL in prod/staging** → hard fail at startup
- **Demo mode in prod/staging** → hard fail at startup
- **SQLite in prod/staging** → hard fail at startup
- **Playground routes** → only mounted in `local`/`dev`
- **OpenAPI/Swagger** → only exposed in `local`/`dev`
- **Auto table creation** → only in `local`/`dev`; use Alembic in staging/prod
