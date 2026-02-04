# Disaster Recovery Runbook

**Last Updated:** 2026-02-03
**Owner:** Platform Engineering
**Classification:** Confidential
**Review Cycle:** Quarterly

---

## 1. Overview

### 1.1 Recovery Objectives

| Metric | Target | Maximum |
|--------|--------|---------|
| **RTO** (Recovery Time Objective) | 1 hour | 4 hours |
| **RPO** (Recovery Point Objective) | 5 minutes | 15 minutes |

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRIMARY REGION (us-east-1)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   API (3x)   │  │  DB Primary │  │   Redis     │             │
│  │   Pods      │  │  (Patroni)  │  │   Cluster   │             │
│  └─────────────┘  └──────┬──────┘  └─────────────┘             │
│                          │ Streaming Replication                 │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DR REGION (us-west-2)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  API (warm) │  │  DB Replica │  │   Redis     │             │
│  │   Standby   │  │  (Standby)  │  │   Replica   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Disaster Classification

### Level 1: Service Degradation
- **Impact:** Single component failure
- **Examples:** One API pod down, cache miss spike
- **Response:** Automatic remediation via Kubernetes

### Level 2: Partial Outage
- **Impact:** Region capacity reduced
- **Examples:** AZ failure, database failover
- **Response:** Traffic shift, capacity scaling

### Level 3: Full Regional Outage
- **Impact:** Primary region unavailable
- **Examples:** Regional outage, data center fire
- **Response:** DR region activation

### Level 4: Catastrophic Loss
- **Impact:** Both regions affected
- **Examples:** Coordinated attack, provider-wide outage
- **Response:** Backup restoration, manual recovery

---

## 3. DR Procedures

### 3.1 Regional Failover (Level 3)

**Decision Criteria:**
- Primary region unreachable for > 10 minutes
- Multiple availability zone failures
- Provider confirmed regional outage

**Procedure:**

#### Step 1: Declare DR Event (5 min)
```bash
# 1. Notify stakeholders
./scripts/dr/notify_stakeholders.sh "DR_INITIATED"

# 2. Create incident channel
./scripts/dr/create_incident_channel.sh

# 3. Log decision
echo "$(date) - DR initiated by $(whoami)" >> /var/log/dr/events.log
```

#### Step 2: Verify DR Region Health (5 min)
```bash
# Check DR cluster health
kubectl --context=dr-cluster get nodes
kubectl --context=dr-cluster get pods -n tensafe

# Verify database replica status
kubectl --context=dr-cluster exec -n tensafe postgres-dr-0 -- \
  patronictl list

# Check replica lag
psql -h db-dr.tensafe.internal -c \
  "SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;"
```

#### Step 3: Promote DR Database (10 min)
```bash
# 1. Break replication and promote replica
kubectl --context=dr-cluster exec -n tensafe postgres-dr-0 -- \
  patronictl failover tensafe-cluster --force

# 2. Verify promotion
kubectl --context=dr-cluster exec -n tensafe postgres-dr-0 -- \
  psql -c "SELECT pg_is_in_recovery();"  # Should return 'f' (false)

# 3. Update connection strings
kubectl --context=dr-cluster apply -f k8s/dr/db-config.yaml
```

#### Step 4: Scale DR Application (5 min)
```bash
# Scale up API pods
kubectl --context=dr-cluster scale deployment tensafe-api \
  -n tensafe --replicas=10

# Enable inference endpoints
kubectl --context=dr-cluster scale deployment tensafe-inference \
  -n tensafe --replicas=5

# Verify pods are running
kubectl --context=dr-cluster get pods -n tensafe -w
```

#### Step 5: Switch Traffic (5 min)
```bash
# 1. Update DNS to point to DR region
./scripts/dr/dns_failover.sh us-west-2

# 2. Update CDN origin
./scripts/dr/cdn_failover.sh us-west-2

# 3. Verify traffic flow
curl -s https://api.tensafe.io/health | jq .region
# Expected: "us-west-2"
```

#### Step 6: Verify Services (10 min)
```bash
# Run smoke tests
./scripts/dr/smoke_tests.sh

# Check critical endpoints
curl -s https://api.tensafe.io/health
curl -s https://api.tensafe.io/v1/training_clients -H "Authorization: Bearer $TEST_KEY"

# Monitor error rates
kubectl --context=dr-cluster logs -f deployment/tensafe-api -n tensafe | grep ERROR
```

#### Step 7: Notify Recovery Complete
```bash
# Update status page
./scripts/status/update.sh "operational" "Failover complete. Services restored in DR region."

# Notify stakeholders
./scripts/dr/notify_stakeholders.sh "DR_COMPLETE"
```

---

### 3.2 Failback to Primary Region

**Prerequisites:**
- Primary region fully recovered
- Database fully synchronized
- All tests passing in primary

**Procedure:**

#### Step 1: Prepare Primary Region (30 min)
```bash
# 1. Verify primary infrastructure
kubectl --context=primary-cluster get nodes
kubectl --context=primary-cluster get pods -n tensafe

# 2. Set up replication from DR to Primary
kubectl --context=primary-cluster exec -n tensafe postgres-0 -- \
  pg_basebackup -h db-dr.tensafe.internal -D /var/lib/postgresql/data -P -R

# 3. Start primary as replica
kubectl --context=primary-cluster exec -n tensafe postgres-0 -- \
  pg_ctl start
```

#### Step 2: Synchronize Data (Varies)
```bash
# Monitor replication lag
watch -n 5 'kubectl --context=primary-cluster exec -n tensafe postgres-0 -- \
  psql -c "SELECT now() - pg_last_xact_replay_timestamp() AS lag;"'

# Wait until lag < 1 second
```

#### Step 3: Schedule Maintenance Window
```bash
# 1. Notify users of planned maintenance
./scripts/status/update.sh "maintenance" \
  "Scheduled maintenance: Failback to primary region. Expected duration: 15 minutes."

# 2. Wait for low-traffic period (if possible)
```

#### Step 4: Execute Failback
```bash
# 1. Stop writes in DR
kubectl --context=dr-cluster scale deployment tensafe-api -n tensafe --replicas=0

# 2. Wait for final sync
sleep 30

# 3. Promote primary database
kubectl --context=primary-cluster exec -n tensafe postgres-0 -- \
  patronictl failover tensafe-cluster --force

# 4. Scale up primary API
kubectl --context=primary-cluster scale deployment tensafe-api -n tensafe --replicas=10

# 5. Switch DNS back to primary
./scripts/dr/dns_failover.sh us-east-1

# 6. Scale down DR (maintain warm standby)
kubectl --context=dr-cluster scale deployment tensafe-api -n tensafe --replicas=1
```

---

## 4. Data Recovery Procedures

### 4.1 Point-in-Time Recovery

**Use case:** Accidental data deletion, corruption

```bash
# 1. Identify recovery point
psql -c "SELECT * FROM audit_logs WHERE operation = 'delete' ORDER BY created_at DESC LIMIT 10;"

# 2. Create recovery database
createdb tensafe_recovery

# 3. Restore to point in time
pg_restore -d tensafe_recovery --target-time="2026-02-03 12:00:00 UTC" /backups/latest.dump

# 4. Extract needed data
psql tensafe_recovery -c "SELECT * FROM training_clients WHERE id = 'xxx';" > recovery_data.sql

# 5. Restore to production (after review)
psql tensafe < recovery_data.sql
```

### 4.2 Full Database Restore

**Use case:** Complete database loss

```bash
# 1. Download latest backup
aws s3 cp s3://tensafe-backups/postgres/daily/latest.dump ./

# 2. Restore database
dropdb tensafe
createdb tensafe
pg_restore -d tensafe latest.dump

# 3. Apply WAL logs for point-in-time
pg_restore --target-time="2026-02-03 14:00:00 UTC" ...
```

### 4.3 Artifact Recovery

**Use case:** Model checkpoint loss

```bash
# 1. List available artifact backups
aws s3 ls s3://tensafe-artifacts-backup/ --recursive

# 2. Restore specific artifact
aws s3 cp s3://tensafe-artifacts-backup/checkpoints/xxx.enc ./

# 3. Verify integrity
sha256sum xxx.enc
# Compare with database record
```

---

## 5. Communication Templates

### 5.1 Initial Incident Notification

```
Subject: [INCIDENT] TenSafe Service Disruption - Investigation in Progress

TenSafe is currently experiencing service disruption.

Status: Investigating
Impact: [Describe impact]
Start Time: [UTC timestamp]

We are actively working to resolve this issue. Updates will be provided every 15 minutes.

For urgent matters, contact: incidents@tensafe.io
```

### 5.2 DR Activation Notification

```
Subject: [DR ACTIVATED] TenSafe Failover to DR Region

We have activated our disaster recovery procedures.

Status: Failover in Progress
Primary Region: Unavailable
DR Region: Activating
Expected Recovery: [Estimated time]

Services may be degraded during failover. We will notify when services are restored.
```

### 5.3 Recovery Complete Notification

```
Subject: [RESOLVED] TenSafe Service Restored

TenSafe services have been fully restored.

Status: Operational
Resolution Time: [UTC timestamp]
Duration: [Total duration]

Root cause analysis will be shared within 48 hours.

We apologize for any inconvenience caused.
```

---

## 6. DR Testing

### 6.1 Testing Schedule

| Test Type | Frequency | Duration | Impact |
|-----------|-----------|----------|--------|
| Tabletop Exercise | Monthly | 2 hours | None |
| Component Failover | Weekly | 30 min | None |
| Database Failover | Monthly | 1 hour | Degraded |
| Full DR Test | Quarterly | 4 hours | Planned maintenance |

### 6.2 DR Test Procedure

```bash
# 1. Schedule maintenance window
# 2. Notify stakeholders

# 3. Execute failover
./scripts/dr/full_dr_test.sh

# 4. Verify all services
./scripts/dr/smoke_tests.sh

# 5. Run performance tests
./scripts/dr/perf_tests.sh

# 6. Failback to primary
./scripts/dr/failback.sh

# 7. Document results
./scripts/dr/generate_report.sh
```

### 6.3 Test Success Criteria

- [ ] Failover completed within RTO (1 hour)
- [ ] Data loss within RPO (5 minutes)
- [ ] All critical endpoints responding
- [ ] Error rate < 1% after stabilization
- [ ] No data corruption detected
- [ ] Audit logs continuous

---

## 7. Contacts and Escalation

### Primary Contacts

| Role | Name | Contact | Hours |
|------|------|---------|-------|
| DR Lead | On-Call Engineer | dr-oncall@tensafe.io | 24/7 |
| Database Admin | DBA Team | dba@tensafe.io | 24/7 |
| Infrastructure | Platform Team | platform@tensafe.io | 24/7 |

### Escalation Path

1. **L1:** On-Call Engineer (5 min response)
2. **L2:** Team Lead (15 min response)
3. **L3:** Engineering Director (30 min response)
4. **L4:** CTO (Critical incidents)

### External Contacts

| Vendor | Contact | Use Case |
|--------|---------|----------|
| AWS Support | Premium Support Portal | Infrastructure issues |
| Cloudflare | Enterprise Support | CDN/DNS issues |
| PagerDuty | support@pagerduty.com | Alerting issues |

---

## 8. Post-Incident Procedures

### 8.1 Incident Review

Within 48 hours of resolution:
1. Collect timeline of events
2. Gather metrics and logs
3. Conduct blameless post-mortem
4. Identify improvement actions
5. Update runbooks if needed

### 8.2 Post-Mortem Template

```markdown
# Incident Post-Mortem: [Title]

## Summary
[Brief description of the incident]

## Timeline
- HH:MM UTC - [Event]
- HH:MM UTC - [Event]

## Root Cause
[Description of root cause]

## Impact
- Duration: X hours Y minutes
- Users affected: N
- Revenue impact: $X

## Resolution
[How the incident was resolved]

## Action Items
- [ ] [Action item 1] - Owner: @person - Due: YYYY-MM-DD
- [ ] [Action item 2] - Owner: @person - Due: YYYY-MM-DD

## Lessons Learned
[Key takeaways]
```

---

*Last DR Test: 2026-01-15*
*Next DR Test: 2026-04-15*
*Document Review: 2026-04-03*
