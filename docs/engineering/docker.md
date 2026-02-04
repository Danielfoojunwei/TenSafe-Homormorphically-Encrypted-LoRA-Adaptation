# TenSafe Docker Build Guide

## Overview

TenSafe uses a multi-stage Docker build to produce optimized production images.
The build process creates a Python wheel and installs it properly (no editable installs).

## Build Process

### Build the image

```bash
docker build -t tensafe:latest .
```

### Build with specific tag

```bash
docker build -t tensafe:v4.0.0 .
```

### Build for specific platform

```bash
docker build --platform linux/amd64 -t tensafe:latest .
```

## Running the Container

### Basic run

```bash
docker run -p 8000:8000 tensafe:latest
```

### With environment configuration

```bash
docker run -p 8000:8000 \
  -e TG_SECRET_KEY="your-secret-key-here" \
  -e DATABASE_URL="postgresql://user:pass@host:5432/tensafe" \
  -e TG_ENVIRONMENT="production" \
  tensafe:latest
```

### With volume mounts for data persistence

```bash
docker run -p 8000:8000 \
  -v /path/to/data:/app/data \
  -v /path/to/cache:/app/.cache \
  tensafe:latest
```

## Container Smoke Test

After building, verify the container works:

```bash
# Test that imports resolve correctly
docker run --rm tensafe:latest python -c "
from tensorguard.platform.main import app
print('tensorguard.platform.main: OK')

import tensafe
print(f'tensafe: OK')

import tg_tinker
print(f'tg_tinker: OK')
"
```

Expected output:
```
tensorguard.platform.main: OK
tensafe: OK
tg_tinker: OK
```

## Health Checks

The container includes a health check that polls `/health` every 30 seconds:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' <container_id>
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TG_ENVIRONMENT` | Runtime environment | `production` |
| `TG_SECRET_KEY` | JWT signing key (required in production) | - |
| `DATABASE_URL` | Database connection string | SQLite fallback |
| `PORT` | API server port | `8000` |
| `WORKERS` | Gunicorn worker count | `4` |
| `TIMEOUT` | Request timeout (seconds) | `120` |
| `KEEPALIVE` | Keep-alive timeout (seconds) | `5` |
| `MAX_REQUESTS` | Max requests per worker before restart | `10000` |
| `MAX_REQUESTS_JITTER` | Jitter for worker restart | `1000` |

## Exposed Ports

- `8000`: API server (FastAPI/Uvicorn)
- `9090`: Prometheus metrics endpoint

## Security Notes

1. The container runs as non-root user `tensafe`
2. `TG_SECRET_KEY` must be set in production (container will warn otherwise)
3. Use `DATABASE_URL` for production database (default SQLite is NOT for production)
4. Mount secrets via Kubernetes secrets or Docker secrets, not environment variables in production

## Troubleshooting

### ModuleNotFoundError

If you see import errors, the wheel may not have been built correctly:

```bash
# Rebuild with no cache
docker build --no-cache -t tensafe:latest .
```

### Permission errors

The container runs as non-root. Ensure mounted volumes are writable:

```bash
# Fix permissions on host
chown -R 1000:1000 /path/to/data
```

### Health check failing

Check container logs:

```bash
docker logs <container_id>
```

Common causes:
- `TG_SECRET_KEY` not set (warnings but should still start)
- Database connection issues
- Port binding conflicts
