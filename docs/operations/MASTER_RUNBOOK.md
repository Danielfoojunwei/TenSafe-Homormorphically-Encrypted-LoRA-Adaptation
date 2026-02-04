# TenSafe Master Operations Runbook

**Version**: 1.0.0
**Last Updated**: 2026-02-04
**Purpose**: Step-by-step procedures for common operational scenarios

---

## Table of Contents

1. [Daily Operations](#1-daily-operations)
2. [Incident Response](#2-incident-response)
3. [Scaling Operations](#3-scaling-operations)
4. [Deployment Procedures](#4-deployment-procedures)
5. [Customer Operations](#5-customer-operations)
6. [Maintenance Procedures](#6-maintenance-procedures)
7. [Disaster Recovery](#7-disaster-recovery)
8. [Security Operations](#8-security-operations)

---

## 1. Daily Operations

### 1.1 Morning Startup Procedure (< 10 minutes)

**When**: Every morning at 08:00

**Steps**:

```bash
# Step 1: Check system status (30 seconds)
curl -s https://api.tensafe.io/health | jq '.'
# Expected: {"status": "healthy", ...}

# Step 2: Check overnight alerts (1 minute)
tensafe-admin alerts list --since "12 hours ago"
# Review any alerts, note P1/P2 for follow-up

# Step 3: Run health check (2 minutes)
tensafe-admin health --quick
# Expected: Overall status GREEN or YELLOW

# Step 4: Check key metrics (2 minutes)
tensafe-admin metrics morning-snapshot
# Review: Uptime, Error Rate, DAU, MRR

# Step 5: Review support queue (2 minutes)
tensafe-admin support summary
# Check for P1/P2 tickets

# Step 6: Post status to Slack (1 minute)
# Post summary to #ops channel
```

**Expected Outcome**: All systems green, no P1/P2 issues

**If Issues Found**: Escalate per incident response procedure (Section 2)

---

### 1.2 Hourly Health Pulse (< 2 minutes)

**When**: Every hour during business hours

**Automated Check** (via cron):
```bash
# /etc/cron.d/tensafe-hourly-pulse
0 * * * * tensafe /opt/tensafe/scripts/hourly_pulse.sh
```

**Script**:
```bash
#!/bin/bash
# hourly_pulse.sh

# Run health check
HEALTH=$(tensafe-admin health --format json)
STATUS=$(echo $HEALTH | jq -r '.overall_status')

# Alert if not healthy
if [ "$STATUS" != "healthy" ]; then
    tensafe-admin alert create \
        --severity warning \
        --message "Hourly pulse: Status is $STATUS"
fi
```

---

### 1.3 End of Day Procedure (< 10 minutes)

**When**: Every evening at 17:00

**Steps**:

```bash
# Step 1: Generate daily summary
tensafe-admin metrics daily-summary > /tmp/daily-summary.txt

# Step 2: Review key metrics
cat /tmp/daily-summary.txt
# Check: Uptime %, Error count, New customers, MRR change

# Step 3: Check pending items
tensafe-admin support list --status pending

# Step 4: Prepare on-call handoff
tensafe-admin oncall handoff --output /tmp/handoff.md

# Step 5: Post handoff to Slack #oncall
cat /tmp/handoff.md | slack-cli post --channel oncall

# Step 6: Archive daily report
cp /tmp/daily-summary.txt /var/log/tensafe/daily/$(date +%Y-%m-%d).txt
```

---

## 2. Incident Response

### 2.1 Incident Severity Classification

| Severity | Impact | Examples | Response Time |
|----------|--------|----------|---------------|
| **P1** | Service down | API unreachable, data loss | < 15 min |
| **P2** | Major degradation | Error rate > 5%, latency > 2s | < 1 hour |
| **P3** | Minor impact | Partial feature outage | < 4 hours |
| **P4** | No user impact | Internal issue, monitoring gap | Next business day |

### 2.2 P1 Incident Response Procedure

**When**: Service is DOWN or data loss suspected

**Time: 0 minutes - Alert Received**

```bash
# 1. Acknowledge the alert
tensafe-admin alert ack <alert-id>

# 2. Start incident
tensafe-admin incident create \
    --severity P1 \
    --title "API Down - $(date)" \
    --commander "$(whoami)"
```

**Time: 0-5 minutes - Initial Assessment**

```bash
# 3. Check what's broken
kubectl get pods -n tensafe
kubectl get events -n tensafe --sort-by='.lastTimestamp' | tail -20

# 4. Check recent changes
kubectl rollout history deployment/tensafe-server -n tensafe

# 5. Quick status check
curl -s https://api.tensafe.io/health
```

**Time: 5-15 minutes - Initial Mitigation**

```bash
# Option A: If recent deployment, rollback
kubectl rollout undo deployment/tensafe-server -n tensafe

# Option B: If pods crashing, restart
kubectl rollout restart deployment/tensafe-server -n tensafe

# Option C: If scaling issue, scale up
kubectl scale deployment/tensafe-server -n tensafe --replicas=10
```

**Time: 15-30 minutes - Stabilization**

```bash
# 6. Verify service restored
curl -s https://api.tensafe.io/health
tensafe-admin health

# 7. Update status page
tensafe-admin statuspage update \
    --status "Identified" \
    --message "Issue identified, implementing fix"

# 8. Notify stakeholders
tensafe-admin notify stakeholders \
    --incident-id <id> \
    --message "P1 incident in progress, service restored, investigating root cause"
```

**Time: 30+ minutes - Root Cause Investigation**

```bash
# 9. Collect logs
kubectl logs -l app=tensafe -n tensafe --since=1h > /tmp/incident-logs.txt

# 10. Check metrics
tensafe-admin metrics export --last 2h > /tmp/incident-metrics.json

# 11. Document in incident
tensafe-admin incident update <id> \
    --notes "Root cause: <description>"
```

**Resolution**

```bash
# 12. Close incident
tensafe-admin incident close <id> \
    --resolution "Rolled back to v1.2.3, fixed in v1.2.4"

# 13. Update status page
tensafe-admin statuspage update --status "Resolved"

# 14. Schedule post-mortem
tensafe-admin incident postmortem schedule <id> \
    --date "tomorrow 10:00"
```

### 2.3 P2 Incident Response Procedure

**When**: Major degradation but service still operational

**Steps**:

```bash
# 1. Acknowledge and assess (< 5 min)
tensafe-admin alert ack <alert-id>
tensafe-admin health

# 2. Check error patterns
kubectl logs -l app=tensafe -n tensafe --tail=100 | grep -i error

# 3. Check metrics
tensafe-admin metrics get error_rate --last 30m
tensafe-admin metrics get p95_latency --last 30m

# 4. Common fixes:
# - High error rate: Check recent deploys, external dependencies
# - High latency: Scale up pods
# - Database issues: Check connections, replicas

# 5. Scale if needed
kubectl scale deployment/tensafe-server -n tensafe --replicas=+2

# 6. Monitor for 15 minutes
watch -n 10 tensafe-admin health

# 7. Document and close if resolved
tensafe-admin alert resolve <alert-id> --notes "Scaled up, error rate normalized"
```

---

## 3. Scaling Operations

### 3.1 Manual Scale Up Procedure

**When**: Auto-scaling insufficient or disabled

```bash
# Step 1: Check current state
kubectl get hpa -n tensafe
kubectl get pods -n tensafe

# Step 2: Calculate target replicas
CURRENT=$(kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.replicas}')
TARGET=$((CURRENT + 2))  # Or multiply by 1.5, etc.

# Step 3: Scale up
kubectl scale deployment/tensafe-server -n tensafe --replicas=$TARGET

# Step 4: Wait for pods to be ready
kubectl rollout status deployment/tensafe-server -n tensafe

# Step 5: Verify scaling
kubectl get pods -n tensafe
tensafe-admin health
```

### 3.2 Emergency Scale Up (Double Capacity)

**When**: System under heavy load, immediate action needed

```bash
#!/bin/bash
# emergency-scale-up.sh

echo "Emergency scale up initiated at $(date)"

# Pause auto-scaler to prevent interference
kubectl annotate scaledobject tensafe-scaledobject -n tensafe \
    autoscaling.keda.sh/paused="true"

# Get current replicas
CURRENT=$(kubectl get deployment tensafe-server -n tensafe -o jsonpath='{.spec.replicas}')
TARGET=$((CURRENT * 2))

echo "Scaling from $CURRENT to $TARGET replicas"

# Scale up
kubectl scale deployment/tensafe-server -n tensafe --replicas=$TARGET

# Wait for ready
kubectl rollout status deployment/tensafe-server -n tensafe --timeout=300s

# Verify
kubectl get pods -n tensafe
echo "Scale up complete at $(date)"

# Resume auto-scaler after 30 minutes
echo "Auto-scaler will resume in 30 minutes"
sleep 1800
kubectl annotate scaledobject tensafe-scaledobject -n tensafe \
    autoscaling.keda.sh/paused-
```

### 3.3 Add GPU Node

**When**: GPU utilization > 85% sustained

```bash
# AWS EKS Example
# Step 1: Check current GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true

# Step 2: Scale node group
aws eks update-nodegroup-config \
    --cluster-name tensafe-prod \
    --nodegroup-name gpu-workers \
    --scaling-config minSize=2,maxSize=10,desiredSize=$((CURRENT_GPUS + 1))

# Step 3: Wait for node to join
watch kubectl get nodes

# Step 4: Verify GPU available
kubectl describe nodes -l nvidia.com/gpu.present=true | grep nvidia.com/gpu
```

---

## 4. Deployment Procedures

### 4.1 Standard Deployment

**When**: Deploying new version to production

```bash
# Step 1: Pre-deployment checks
tensafe-admin predeploy check
# Verifies: Tests pass, no P1/P2 incidents, off-peak hours

# Step 2: Create deployment record
tensafe-admin deploy create \
    --version v1.2.4 \
    --deployer "$(whoami)" \
    --change-log "Fixed bug #123, added feature #456"

# Step 3: Deploy with Helm
helm upgrade tensafe ./deploy/helm/tensafe \
    --namespace tensafe \
    --set image.tag=v1.2.4 \
    --wait \
    --timeout 10m

# Step 4: Verify deployment
kubectl rollout status deployment/tensafe-server -n tensafe
tensafe-admin health

# Step 5: Run smoke tests
tensafe-admin test smoke --env production

# Step 6: Monitor for 15 minutes
# Watch error rate and latency

# Step 7: Mark deployment complete
tensafe-admin deploy complete --version v1.2.4
```

### 4.2 Rollback Procedure

**When**: Deployment caused issues

```bash
# Step 1: Identify the problem
tensafe-admin health
kubectl logs -l app=tensafe -n tensafe --tail=50

# Step 2: Rollback to previous version
kubectl rollout undo deployment/tensafe-server -n tensafe

# Or rollback to specific version
kubectl rollout undo deployment/tensafe-server -n tensafe --to-revision=3

# Step 3: Verify rollback
kubectl rollout status deployment/tensafe-server -n tensafe
tensafe-admin health

# Step 4: Document
tensafe-admin deploy rollback \
    --from-version v1.2.4 \
    --to-version v1.2.3 \
    --reason "Error rate spiked to 5%"
```

---

## 5. Customer Operations

### 5.1 New Enterprise Customer Onboarding

**When**: New enterprise customer signs contract

**Checklist**:

```markdown
## Enterprise Onboarding - [CUSTOMER NAME]

### Day 0: Contract Signed
- [ ] Create tenant in system: `tensafe-admin tenant create --name "CustomerName" --tier enterprise`
- [ ] Generate API keys: `tensafe-admin apikey create --tenant <id>`
- [ ] Create admin user account
- [ ] Send welcome email with credentials
- [ ] Schedule kickoff call

### Day 1-3: Technical Kickoff
- [ ] Kickoff call completed
- [ ] Architecture review done
- [ ] Integration timeline agreed
- [ ] Success criteria defined
- [ ] Assign CSM and SE

### Day 4-14: Implementation
- [ ] Customer integrated SDK
- [ ] First API call made
- [ ] First training job run
- [ ] Test environment verified
- [ ] Security review passed

### Day 15-30: Go-Live
- [ ] Production deployment complete
- [ ] Load testing passed
- [ ] Monitoring configured
- [ ] Documentation delivered
- [ ] Training completed

### Post-Launch
- [ ] 30-day check-in scheduled
- [ ] Success metrics tracking
- [ ] QBR scheduled
```

### 5.2 Customer Issue Escalation

**When**: Customer reports critical issue

```bash
# Step 1: Get customer details
tensafe-admin customer info --tenant-id <id>

# Step 2: Check customer's recent activity
tensafe-admin customer activity --tenant-id <id> --last 24h

# Step 3: Check for errors
tensafe-admin logs search --tenant-id <id> --level error --last 24h

# Step 4: Create support ticket with high priority
tensafe-admin support create \
    --tenant-id <id> \
    --priority P2 \
    --title "Customer-reported issue: <summary>" \
    --description "<details>"

# Step 5: Assign to engineer
tensafe-admin support assign <ticket-id> --to <engineer>

# Step 6: Update customer
tensafe-admin support comment <ticket-id> \
    --message "We've received your report and are investigating. ETA for update: 1 hour"
```

### 5.3 Customer Churn Prevention

**When**: Customer shows churn signals

```bash
# Step 1: Identify at-risk customers
tensafe-admin customer at-risk

# Step 2: Get churn risk details
tensafe-admin customer health --tenant-id <id>

# Step 3: Review usage patterns
tensafe-admin customer usage --tenant-id <id> --period 90d

# Step 4: Create intervention plan
tensafe-admin customer intervention create \
    --tenant-id <id> \
    --type "engagement_call" \
    --notes "Usage down 50%, schedule check-in"

# Step 5: Schedule CSM outreach
# Manual step: CSM contacts customer
```

---

## 6. Maintenance Procedures

### 6.1 Scheduled Maintenance Window

**When**: Planned maintenance (e.g., database upgrade)

**Pre-Maintenance (24 hours before)**:

```bash
# 1. Create maintenance window
tensafe-admin maintenance create \
    --title "Database upgrade" \
    --start "2026-02-05 02:00 UTC" \
    --end "2026-02-05 04:00 UTC" \
    --impact "Brief service interruption possible"

# 2. Notify customers
tensafe-admin maintenance notify --id <maint-id>

# 3. Update status page
tensafe-admin statuspage maintenance schedule --id <maint-id>
```

**During Maintenance**:

```bash
# 1. Start maintenance
tensafe-admin maintenance start --id <maint-id>

# 2. Update status page
tensafe-admin statuspage update --status "Maintenance in progress"

# 3. Perform maintenance tasks
# <your maintenance steps>

# 4. Verify system health
tensafe-admin health

# 5. End maintenance
tensafe-admin maintenance end --id <maint-id>

# 6. Update status page
tensafe-admin statuspage update --status "Operational"
```

### 6.2 Database Maintenance

**When**: Weekly or as needed

```bash
# 1. Check if maintenance needed
tensafe-admin db health

# 2. Run vacuum (PostgreSQL)
tensafe-admin db vacuum --analyze

# 3. Reindex if needed
tensafe-admin db reindex --table <table-name>

# 4. Check for long-running queries
tensafe-admin db queries --running --min-duration 60s

# 5. Kill problematic queries if needed
tensafe-admin db query kill <query-id>
```

### 6.3 Certificate Renewal

**When**: Certificate expiring within 30 days

```bash
# 1. Check certificate status
tensafe-admin cert status

# 2. Renew certificate
tensafe-admin cert renew --domain api.tensafe.io

# 3. Verify renewal
curl -v https://api.tensafe.io 2>&1 | grep "expire date"

# 4. Restart ingress if needed
kubectl rollout restart deployment/ingress-nginx -n ingress-nginx
```

---

## 7. Disaster Recovery

### 7.1 Database Failover

**When**: Primary database fails

```bash
# 1. Detect failure
tensafe-admin db status
# Shows: Primary UNREACHABLE

# 2. Promote read replica
tensafe-admin db failover --promote replica-1

# 3. Update connection strings
kubectl set env deployment/tensafe-server -n tensafe \
    DATABASE_URL=postgresql://replica-1.tensafe.internal:5432/tensafe

# 4. Verify connectivity
tensafe-admin db health

# 5. Document failover
tensafe-admin incident create \
    --severity P1 \
    --title "Database failover executed"
```

### 7.2 Full Service Recovery

**When**: Major outage requiring full recovery

```bash
# 1. Assess damage
kubectl get all -n tensafe
tensafe-admin health --verbose

# 2. Restore from backup if needed
tensafe-admin backup restore \
    --backup-id latest \
    --confirm

# 3. Restart all services
kubectl rollout restart deployment -n tensafe

# 4. Verify each component
tensafe-admin health --component all

# 5. Run integration tests
tensafe-admin test integration --env production

# 6. Gradually restore traffic
# If using traffic management, gradually increase traffic

# 7. Monitor closely for 24 hours
```

### 7.3 Data Corruption Recovery

**When**: Data integrity issue detected

```bash
# 1. Stop writes immediately
kubectl scale deployment/tensafe-server -n tensafe --replicas=0

# 2. Assess corruption scope
tensafe-admin db integrity-check

# 3. Identify last good backup
tensafe-admin backup list --status verified

# 4. Point-in-time recovery
tensafe-admin backup restore \
    --backup-id <backup-id> \
    --point-in-time "2026-02-04 12:00:00"

# 5. Verify data integrity
tensafe-admin db integrity-check

# 6. Restore service
kubectl scale deployment/tensafe-server -n tensafe --replicas=5

# 7. Notify affected customers
tensafe-admin notify customers \
    --affected-since "2026-02-04 12:00:00" \
    --message "Data recovery completed, please verify your data"
```

---

## 8. Security Operations

### 8.1 Security Incident Response

**When**: Security breach suspected

```bash
# IMMEDIATE (0-5 minutes)
# 1. Isolate affected systems
kubectl cordon <node-name>
kubectl drain <node-name> --ignore-daemonsets

# 2. Preserve evidence
kubectl logs -l app=tensafe -n tensafe --all-containers > /secure/incident-logs-$(date +%s).txt

# 3. Create security incident
tensafe-admin security incident create \
    --severity critical \
    --type "potential-breach" \
    --initial-findings "<description>"

# SHORT-TERM (5-30 minutes)
# 4. Rotate credentials if needed
tensafe-admin secrets rotate --all

# 5. Review access logs
tensafe-admin security audit --last 24h

# 6. Block suspicious IPs
tensafe-admin security block-ip <ip-address>

# INVESTIGATION (30+ minutes)
# 7. Full security audit
tensafe-admin security audit --full --output /secure/audit-report.json

# 8. Notify legal/compliance if required
# Manual step based on findings
```

### 8.2 API Key Rotation

**When**: Scheduled rotation or suspected compromise

```bash
# 1. Generate new keys
tensafe-admin apikey rotate --tenant-id <id>

# 2. Notify customer
tensafe-admin notify customer <id> \
    --subject "API Key Rotation" \
    --message "Your API keys have been rotated. New keys have been sent securely."

# 3. Set grace period for old keys (optional)
tensafe-admin apikey deprecate \
    --key-id <old-key-id> \
    --grace-period 24h

# 4. Monitor for usage of old keys
tensafe-admin apikey usage --key-id <old-key-id>

# 5. Revoke old keys after grace period
tensafe-admin apikey revoke --key-id <old-key-id>
```

### 8.3 Compliance Audit Preparation

**When**: SOC2/HIPAA audit scheduled

```bash
# 1. Generate compliance report
tensafe-admin compliance report \
    --framework soc2 \
    --period "2025-01-01 to 2025-12-31" \
    --output /reports/soc2-evidence.pdf

# 2. Export audit logs
tensafe-admin audit-log export \
    --start "2025-01-01" \
    --end "2025-12-31" \
    --output /reports/audit-logs.json

# 3. Generate access reviews
tensafe-admin security access-review \
    --output /reports/access-review.csv

# 4. Generate encryption evidence
tensafe-admin security encryption-report \
    --output /reports/encryption-status.pdf

# 5. Package all evidence
tensafe-admin compliance package \
    --framework soc2 \
    --output /reports/soc2-evidence-package.zip
```

---

## Quick Reference: Emergency Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| On-Call Primary | Rotation | PagerDuty | oncall@tensafe.io |
| Engineering Lead | ___ | ___ | ___ |
| CTO | ___ | ___ | ___ |
| Security | ___ | ___ | security@tensafe.io |
| Legal | ___ | ___ | legal@tensafe.io |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-04 | Platform Ops | Initial master runbook |
