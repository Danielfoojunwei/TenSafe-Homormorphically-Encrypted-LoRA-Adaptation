# Database Operations Runbook

**Last Updated:** 2026-02-03
**Owner:** Platform Engineering
**Review Cycle:** Quarterly

---

## Overview

This runbook covers database operations for TenSafe's PostgreSQL database, including backup, restore, migration, and troubleshooting procedures.

## Database Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  FastAPI → SQLModel/SQLAlchemy → Connection Pool (asyncpg)  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────────────────────┐
│                   PostgreSQL Cluster                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │   Primary   │──▶│  Replica 1  │   │  Replica 2  │       │
│  │  (writes)   │   │  (reads)    │   │  (reads)    │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Backup Procedures

### 1.1 Automated Backups

Backups run automatically via cron:
- **Full backup:** Daily at 02:00 UTC
- **WAL archiving:** Continuous (for point-in-time recovery)
- **Retention:** 30 days

**Verify backup status:**
```bash
# Check last backup
kubectl exec -n tensafe postgres-0 -- \
  cat /var/lib/postgresql/backup/last_backup.log

# List available backups
aws s3 ls s3://tensafe-backups/postgres/ --recursive | tail -20
```

### 1.2 Manual Backup

**Create immediate backup:**
```bash
# Full backup using pg_dump
kubectl exec -n tensafe postgres-0 -- \
  pg_dump -Fc -f /tmp/backup.dump tensafe

# Copy to local machine
kubectl cp tensafe/postgres-0:/tmp/backup.dump ./backup.dump

# Upload to S3
aws s3 cp ./backup.dump s3://tensafe-backups/postgres/manual/$(date +%Y%m%d_%H%M%S).dump
```

**Table-specific backup:**
```bash
kubectl exec -n tensafe postgres-0 -- \
  pg_dump -Fc -t training_clients -t audit_logs tensafe > tables.dump
```

### 1.3 Backup Verification

**Weekly verification procedure:**
```bash
# 1. Download latest backup
aws s3 cp s3://tensafe-backups/postgres/daily/latest.dump ./verify.dump

# 2. Restore to verification database
createdb tensafe_verify
pg_restore -d tensafe_verify ./verify.dump

# 3. Run integrity checks
psql tensafe_verify -c "SELECT COUNT(*) FROM training_clients;"
psql tensafe_verify -c "SELECT COUNT(*) FROM audit_logs;"

# 4. Cleanup
dropdb tensafe_verify
rm ./verify.dump
```

---

## 2. Restore Procedures

### 2.1 Full Database Restore

**Prerequisites:**
- Backup file available
- Sufficient disk space
- Application stopped or in maintenance mode

**Procedure:**
```bash
# 1. Enable maintenance mode
kubectl patch deployment tensafe-api -n tensafe \
  -p '{"spec":{"replicas":0}}'

# 2. Stop connections to database
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='tensafe';"

# 3. Drop and recreate database
dropdb tensafe
createdb tensafe

# 4. Restore from backup
pg_restore -d tensafe /path/to/backup.dump

# 5. Verify restore
psql tensafe -c "\dt"
psql tensafe -c "SELECT COUNT(*) FROM training_clients;"

# 6. Resume application
kubectl patch deployment tensafe-api -n tensafe \
  -p '{"spec":{"replicas":3}}'
```

### 2.2 Point-in-Time Recovery

**Restore to specific timestamp:**
```bash
# 1. Stop PostgreSQL
pg_ctl stop -D /var/lib/postgresql/data

# 2. Restore base backup
rm -rf /var/lib/postgresql/data/*
tar -xzf /backups/base/latest.tar.gz -C /var/lib/postgresql/data/

# 3. Configure recovery
cat > /var/lib/postgresql/data/recovery.conf << EOF
restore_command = 'aws s3 cp s3://tensafe-backups/wal/%f %p'
recovery_target_time = '2026-02-03 14:30:00 UTC'
recovery_target_action = 'promote'
EOF

# 4. Start PostgreSQL
pg_ctl start -D /var/lib/postgresql/data
```

### 2.3 Table-Level Restore

**Restore specific tables:**
```bash
# 1. Extract table from backup
pg_restore -l backup.dump | grep "TABLE DATA training_clients" > restore.list

# 2. Restore to temporary table
pg_restore -d tensafe -t training_clients_restore backup.dump

# 3. Verify data
psql tensafe -c "SELECT COUNT(*) FROM training_clients_restore;"

# 4. Swap tables (if verified)
psql tensafe << EOF
BEGIN;
ALTER TABLE training_clients RENAME TO training_clients_old;
ALTER TABLE training_clients_restore RENAME TO training_clients;
DROP TABLE training_clients_old;
COMMIT;
EOF
```

---

## 3. Migration Procedures

### 3.1 Running Migrations

**Development:**
```bash
# Generate new migration
alembic revision --autogenerate -m "Add usage_events table"

# Review generated migration
cat alembic/versions/xxx_add_usage_events_table.py

# Apply migration
alembic upgrade head
```

**Production:**
```bash
# 1. Create backup before migration
pg_dump -Fc tensafe > pre_migration_$(date +%Y%m%d).dump

# 2. Run migration in maintenance window
alembic upgrade head

# 3. Verify migration
alembic current
psql tensafe -c "\d+ new_table"

# 4. Monitor for errors
kubectl logs -f deployment/tensafe-api -n tensafe | grep -i error
```

### 3.2 Rollback Migration

**Rollback last migration:**
```bash
# Show current version
alembic current

# Rollback one step
alembic downgrade -1

# Rollback to specific version
alembic downgrade abc123def456
```

### 3.3 Zero-Downtime Migrations

**For schema changes that require zero downtime:**

1. **Add new column (nullable):**
```sql
ALTER TABLE training_clients ADD COLUMN new_field VARCHAR(255);
```

2. **Deploy code that writes to both old and new:**
```python
# Dual-write phase
client.old_field = value
client.new_field = value
```

3. **Backfill existing data:**
```sql
UPDATE training_clients SET new_field = old_field WHERE new_field IS NULL;
```

4. **Deploy code that reads from new:**
```python
value = client.new_field or client.old_field
```

5. **Remove old column (after verification):**
```sql
ALTER TABLE training_clients DROP COLUMN old_field;
```

---

## 4. Performance Troubleshooting

### 4.1 Slow Query Investigation

**Find slow queries:**
```sql
-- Top 10 slowest queries
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- Currently running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 seconds';
```

**Explain slow query:**
```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM training_clients WHERE tenant_id = 'xxx';
```

### 4.2 Index Analysis

**Check missing indexes:**
```sql
-- Tables with sequential scans
SELECT schemaname, relname, seq_scan, seq_tup_read,
       idx_scan, idx_tup_fetch
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan
ORDER BY seq_tup_read DESC;

-- Index usage
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

**Create index:**
```sql
-- Create index concurrently (no blocking)
CREATE INDEX CONCURRENTLY idx_training_clients_tenant
ON training_clients (tenant_id);
```

### 4.3 Connection Pool Issues

**Check connection usage:**
```sql
-- Current connections by state
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state;

-- Connections by client
SELECT client_addr, count(*)
FROM pg_stat_activity
GROUP BY client_addr
ORDER BY count DESC;
```

**Kill idle connections:**
```sql
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
AND query_start < now() - interval '30 minutes';
```

### 4.4 Disk Space Issues

**Check table sizes:**
```sql
SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

**Reclaim space:**
```sql
-- Analyze and vacuum
VACUUM ANALYZE training_clients;

-- Full vacuum (locks table)
VACUUM FULL training_clients;
```

---

## 5. Failover Procedures

### 5.1 Automatic Failover

Patroni handles automatic failover. Monitor status:
```bash
# Check cluster status
kubectl exec -n tensafe postgres-0 -- patronictl list

# View Patroni logs
kubectl logs -n tensafe postgres-0 -c patroni
```

### 5.2 Manual Failover

**Promote replica to primary:**
```bash
# 1. Identify target replica
kubectl exec -n tensafe postgres-0 -- patronictl list

# 2. Initiate switchover
kubectl exec -n tensafe postgres-0 -- \
  patronictl switchover --master postgres-0 --candidate postgres-1 --force

# 3. Verify new primary
kubectl exec -n tensafe postgres-0 -- patronictl list
```

### 5.3 Disaster Recovery Failover

**Failover to DR region:**
```bash
# 1. Promote DR cluster
kubectl --context=dr-cluster exec -n tensafe postgres-dr-0 -- \
  patronictl failover --force

# 2. Update DNS
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123 \
  --change-batch file://dns-failover.json

# 3. Verify connectivity
psql -h db.tensafe.io -U tensafe -c "SELECT 1;"
```

---

## 6. Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| DBA On-Call | dba-oncall@tensafe.io | PagerDuty |
| Platform Lead | platform@tensafe.io | Slack #platform |
| Vendor Support | support@postgresql.org | Support Portal |

---

## 7. Maintenance Windows

| Operation | Window | Duration | Impact |
|-----------|--------|----------|--------|
| Minor version upgrade | Sunday 02:00-04:00 UTC | 30 min | Read-only |
| Major version upgrade | Scheduled | 2-4 hours | Downtime |
| Index creation | Anytime | Varies | None (CONCURRENTLY) |
| VACUUM FULL | Sunday 02:00-06:00 UTC | 1-2 hours | Table locked |

---

*Last tested: 2026-01-15*
*Next review: 2026-04-03*
