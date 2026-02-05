# TenSafe Operational Runbook

This document provides procedures for operating and troubleshooting TenSafe in production.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Health Monitoring](#health-monitoring)
3. [Common Issues and Resolutions](#common-issues-and-resolutions)
4. [Emergency Procedures](#emergency-procedures)
5. [Maintenance Tasks](#maintenance-tasks)

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    FastAPI Service                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Auth Layer  │  │ Rate Limit  │  │ Observability       │  │
│  │ (API Keys)  │  │             │  │ (Metrics, Logs)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ TG-Tinker API Routes                                    ││
│  │ - Training Clients  - Futures  - Artifacts  - DLQ       ││
│  └─────────────────────────────────────────────────────────┘│
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
    ┌──────────▼──────────┐        ┌─────────▼─────────┐
    │   PostgreSQL/SQLite │        │   Storage Backend │
    │   (State, Queue)    │        │   (Artifacts)     │
    └─────────────────────┘        └───────────────────┘
```

### Key Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TG_ENVIRONMENT` | Yes | `development` or `production` |
| `TG_SECRET_KEY` | Prod | JWT/session signing key |
| `DATABASE_URL` | Prod | PostgreSQL connection string |
| `TG_MASTER_KEY` | Prod | Artifact encryption master key |
| `TG_AUTH_MODE` | No | Override auth mode |
| `TG_ALLOW_MOCK_BACKEND` | No | Allow mock vLLM in production |

---

## Health Monitoring

### Health Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/internal/health` | Liveness probe | `200 {"status": "ok"}` |
| `/internal/ready` | Readiness probe | `200 {checks: [...]}` |
| `/internal/metrics` | Prometheus metrics | Prometheus text format |

### Key Metrics

**Request Metrics:**
- `tg_tinker_requests_total{method,endpoint,status}` - Request count
- `tg_tinker_request_duration_seconds{method,endpoint}` - Latency histogram

**Business Metrics:**
- `tg_tinker_training_clients_active{tenant_id}` - Active clients
- `tg_tinker_queue_depth{status}` - Job queue depth
- `tg_tinker_dp_epsilon_total{tenant_id,training_client_id}` - DP budget spent

**Storage Metrics:**
- `tg_tinker_artifact_storage_bytes{tenant_id}` - Storage usage

### Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Request latency p99 | > 1s | > 5s |
| Error rate | > 1% | > 5% |
| Queue depth | > 100 | > 1000 |
| Health check failures | 1 | 3 consecutive |

---

## Common Issues and Resolutions

### Issue: High Request Latency

**Symptoms:**
- `tg_tinker_request_duration_seconds` p99 > 1s
- Slow API responses

**Investigation:**
```bash
# Check database connections
docker exec <container> psql -c "SELECT count(*) FROM pg_stat_activity"

# Check queue depth
curl http://localhost:8000/internal/metrics | grep queue_depth

# Check for long-running queries
docker exec <container> psql -c "SELECT query, now() - query_start as duration FROM pg_stat_activity WHERE state = 'active'"
```

**Resolution:**
1. Scale up database connections if at limit
2. Check for missing indexes on frequently queried columns
3. Consider adding read replicas for heavy read workloads

---

### Issue: Job Queue Backlog

**Symptoms:**
- `tg_tinker_queue_depth{status="pending"}` continuously increasing
- Clients timing out waiting for futures

**Investigation:**
```bash
# Check worker status
docker logs <worker_container> --tail=100

# Check job failure rate
curl http://localhost:8000/internal/metrics | grep -E "(completed|failed)_total"

# Check DLQ depth
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/admin/dlq/stats
```

**Resolution:**
1. Scale up workers if processing is healthy but slow
2. Check for recurring job failures (may indicate backend issue)
3. If jobs are failing, check dead letter queue for error patterns:
   ```bash
   curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/v1/admin/dlq
   ```

---

### Issue: Authentication Failures

**Symptoms:**
- 401 responses increasing
- `INVALID_API_KEY` or `API_KEY_EXPIRED` errors in logs

**Investigation:**
```bash
# Check recent auth failures
docker logs <container> 2>&1 | grep -i "auth\|401" | tail -50

# Verify API key status (admin endpoint)
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://localhost:8000/v1/admin/keys?prefix=<key_prefix>
```

**Resolution:**
1. Verify client is using correct API key
2. Check if key has expired (`expires_at` field)
3. Check if tenant is suspended
4. Rotate key if compromised

---

### Issue: Artifact Storage Full

**Symptoms:**
- `tg_tinker_artifact_storage_bytes` approaching quota
- Storage write failures

**Investigation:**
```bash
# Check storage usage by tenant
curl http://localhost:8000/internal/metrics | grep storage_bytes

# Check disk space
df -h /app/data
```

**Resolution:**
1. Identify tenants approaching quota
2. Contact tenants to clean up old artifacts
3. If urgent, temporarily increase quota
4. Consider implementing retention policies

---

### Issue: HE Backend Unavailable

**Symptoms:**
- `HENotAvailableError` in logs
- All HE operations failing

**Investigation:**
```bash
# Check HE backend status
docker logs <container> 2>&1 | grep -i "tenseal\|he\|homomorphic"

# Verify execution policy
echo $TG_ENVIRONMENT
echo $TG_ALLOW_MOCK_BACKEND
```

**Resolution:**
1. Verify TenSEAL is installed: `pip show tenseal`
2. Check for library conflicts
3. If development, consider enabling mock mode (NOT for production)
4. Restart container to reinitialize HE backend

---

### Issue: Database Connection Failures

**Symptoms:**
- 500 errors on all endpoints
- `OperationalError: could not connect to server`

**Investigation:**
```bash
# Test database connectivity
docker exec <container> python -c "from sqlmodel import create_engine; e=create_engine('$DATABASE_URL'); e.connect()"

# Check database logs
docker logs <db_container> --tail=100
```

**Resolution:**
1. Verify `DATABASE_URL` is correct
2. Check database is running and accepting connections
3. Verify network connectivity between app and database
4. Check connection pool exhaustion

---

## Emergency Procedures

### Procedure: Emergency Tenant Suspension

**When:** Suspected abuse, security breach, or compliance violation

```bash
# 1. Suspend tenant via admin API
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"reason": "Security investigation"}' \
     http://localhost:8000/v1/admin/tenants/<tenant_id>/suspend

# 2. Revoke all API keys
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://localhost:8000/v1/admin/tenants/<tenant_id>/revoke-all-keys

# 3. Cancel pending jobs
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://localhost:8000/v1/admin/tenants/<tenant_id>/cancel-jobs

# 4. Document incident
```

---

### Procedure: Master Key Rotation

**When:** Scheduled rotation or suspected compromise

```bash
# 1. Generate new master key
NEW_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# 2. Deploy with both keys (old for decrypt, new for encrypt)
export TG_MASTER_KEY=$NEW_KEY
export TG_MASTER_KEY_PREVIOUS=$OLD_KEY

# 3. Re-encrypt existing artifacts (background task)
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
     http://localhost:8000/v1/admin/artifacts/reencrypt

# 4. After reencryption completes, remove old key
unset TG_MASTER_KEY_PREVIOUS

# 5. Restart services
```

---

### Procedure: Database Recovery

**When:** Database corruption or data loss

```bash
# 1. Stop all services
docker-compose stop api worker

# 2. Restore from backup
pg_restore -d tensafe /backups/latest.dump

# 3. Verify data integrity
docker exec <db_container> psql -d tensafe -c "SELECT count(*) FROM tinker_training_clients"

# 4. Restart services
docker-compose start api worker

# 5. Verify health
curl http://localhost:8000/internal/ready
```

---

## Maintenance Tasks

### Daily Tasks

1. **Check alerts** - Review monitoring dashboards
2. **Check DLQ** - Process or acknowledge dead letter entries
3. **Check logs** - Review error patterns

### Weekly Tasks

1. **Review metrics trends** - Look for degradation patterns
2. **Check storage growth** - Plan for capacity
3. **Test backups** - Verify backup restoration works

### Monthly Tasks

1. **Key rotation** - Rotate master keys and admin credentials
2. **Security review** - Review access logs for anomalies
3. **Capacity planning** - Review usage trends

### Quarterly Tasks

1. **Disaster recovery test** - Full restore test
2. **Performance baseline** - Update performance benchmarks
3. **Dependency updates** - Security patches

---

## Contact Information

| Role | Contact |
|------|---------|
| On-call Engineer | <on-call-rotation> |
| Security Team | security@example.com |
| Database Admin | dba@example.com |

---

## Change Log

| Date | Author | Description |
|------|--------|-------------|
| 2024-01-XX | Initial | Created runbook |
