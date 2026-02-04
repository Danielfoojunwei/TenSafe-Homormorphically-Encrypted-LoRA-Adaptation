# TenSafe Daily Operations Checklist

**Version**: 1.0.0
**Last Updated**: 2026-02-04
**Purpose**: Step-by-step daily checklist for founders and operators to ensure product health and customer readiness

---

## Quick Reference: What to Check and When

| Time | Duration | Task | Priority |
|------|----------|------|----------|
| 08:00 | 5 min | Morning Health Check | Critical |
| 08:15 | 10 min | Customer Impact Review | High |
| 10:00 | 5 min | Mid-Morning Pulse | Medium |
| 14:00 | 5 min | Afternoon Check | Medium |
| 17:00 | 10 min | End-of-Day Review | High |
| 23:00 | Auto | Nightly Reports Generated | Auto |

---

## PART 1: MORNING HEALTH CHECK (08:00 - 5 minutes)

### Step 1: System Status Dashboard

**What to Check**: `https://dashboard.tensafe.io/status` or run:

```bash
# Quick status check
curl -s https://api.tensafe.io/health | jq '.'

# Or via CLI
tensafe-admin status --quick
```

**Dashboard Panels to Review**:

| Panel | Healthy | Warning | Critical | Action if Critical |
|-------|---------|---------|----------|-------------------|
| API Uptime (24h) | > 99.9% | 99.5-99.9% | < 99.5% | Check incident log |
| Error Rate (1h) | < 0.1% | 0.1-1% | > 1% | Review error logs |
| P95 Latency | < 100ms | 100-200ms | > 200ms | Check scaling |
| Active Pods | 3+ | 2 | < 2 | Scale immediately |
| GPU Utilization | < 80% | 80-90% | > 90% | Add GPU nodes |

**Traffic Light Status**:
```
GREEN  = All systems operational
YELLOW = Degraded performance, monitoring required
RED    = Service impacted, immediate action needed
```

### Step 2: Overnight Incidents Review

**Check PagerDuty/Slack #incidents**:

```bash
# List overnight incidents
tensafe-admin incidents list --since "8 hours ago"
```

**For each incident, verify**:
- [ ] Incident resolved or escalated appropriately
- [ ] Root cause identified (or investigation ongoing)
- [ ] Customer communication sent if required
- [ ] Post-mortem scheduled if P1/P2

### Step 3: Critical Metrics Snapshot

```bash
# Run morning metrics snapshot
tensafe-admin metrics morning-snapshot
```

**Key Numbers to Record Daily**:

| Metric | Yesterday | Today | Trend | Concern If |
|--------|-----------|-------|-------|------------|
| Active Users (DAU) | ___ | ___ | | Down > 10% |
| API Calls (24h) | ___ | ___ | | Down > 20% |
| Training Jobs (24h) | ___ | ___ | | Down > 30% |
| Error Count (24h) | ___ | ___ | | Up > 50% |
| P95 Latency (avg) | ___ ms | ___ ms | | Up > 50ms |

---

## PART 2: CUSTOMER IMPACT REVIEW (08:15 - 10 minutes)

### Step 4: Support Queue Review

**Check**: Zendesk/Intercom Dashboard

```bash
# Get support ticket summary
tensafe-admin support summary
```

**Triage Priority**:

| Ticket Type | SLA Response | Action |
|-------------|--------------|--------|
| P1 (Service Down) | 1 hour | Escalate immediately |
| P2 (Major Bug) | 4 hours | Assign to on-call |
| P3 (Minor Issue) | 24 hours | Queue for sprint |
| P4 (Question) | 48 hours | Route to docs/CSM |

**Questions to Answer**:
- [ ] Any P1/P2 tickets open?
- [ ] Any tickets > 24 hours without response?
- [ ] Any patterns in recent tickets (same issue repeated)?

### Step 5: Customer Health Dashboard

**Check**: `https://dashboard.tensafe.io/customers`

**At-Risk Customers** (Health Score < 40):

| Customer | Health Score | Last Active | MRR | Risk Indicator | Action |
|----------|--------------|-------------|-----|----------------|--------|
| ___ | ___ | ___ | $___ | ___ | ___ |

**Churn Signals to Watch**:
- No API calls in 7+ days
- Support tickets increasing
- Failed payment
- Team member accounts removed

### Step 6: Enterprise Customer Check

For each enterprise customer ($5K+ MRR):

```bash
# Check enterprise customer status
tensafe-admin customers enterprise-status
```

**Daily Enterprise Checklist**:
- [ ] All enterprise environments healthy
- [ ] No SLA breaches (99.9% uptime)
- [ ] No pending security reviews
- [ ] Scheduled maintenance communicated

---

## PART 3: MID-DAY OPERATIONAL CHECKS

### Step 7: Mid-Morning Pulse (10:00 - 5 minutes)

```bash
# Quick pulse check
tensafe-admin pulse
```

**What Changed Since Morning?**:
- [ ] Any new alerts triggered?
- [ ] Traffic patterns normal for time of day?
- [ ] Queue depth within limits?

**Traffic Pattern Reference**:

| Time (UTC) | Expected Traffic | Alert If |
|------------|------------------|----------|
| 00:00-06:00 | 10-20% of peak | > 30% (unusual) |
| 06:00-10:00 | 40-60% of peak | < 20% (outage?) |
| 10:00-14:00 | 80-100% of peak | < 50% (issue) |
| 14:00-18:00 | 70-90% of peak | < 40% (issue) |
| 18:00-24:00 | 30-50% of peak | > 80% (unusual) |

### Step 8: Afternoon Check (14:00 - 5 minutes)

**Capacity Check**:

```bash
# Check resource utilization
kubectl top pods -n tensafe
kubectl top nodes
```

| Resource | Current | Threshold | Action |
|----------|---------|-----------|--------|
| CPU Avg | ___% | > 70% | Scale up |
| Memory Avg | ___% | > 80% | Scale up |
| GPU Util | ___% | > 85% | Add GPU nodes |
| Disk Usage | ___% | > 80% | Clean up / expand |

**Scaling Decision**:
```
If CPU > 70% OR Memory > 80% OR Latency > 100ms:
  → Scale up by 25% (or +2 pods minimum)

If CPU < 30% AND Memory < 40% AND time > 18:00:
  → Scale down by 1 pod (maintain minimum 3)
```

---

## PART 4: END-OF-DAY REVIEW (17:00 - 10 minutes)

### Step 9: Daily Metrics Summary

```bash
# Generate daily summary
tensafe-admin metrics daily-summary --date today
```

**Business Metrics Checklist**:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| New Signups | ___ | ___ | |
| Activated Users | ___ | ___ | |
| Paid Conversions | ___ | ___ | |
| Churned Users | 0 | ___ | |
| MRR Change | +$___ | +$___ | |

**Operational Metrics Checklist**:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Uptime | 99.9% | ___% | |
| Error Rate | < 0.1% | ___% | |
| P95 Latency | < 100ms | ___ ms | |
| Training Jobs Completed | N/A | ___ | |
| Support Tickets Resolved | All P1/P2 | ___ | |

### Step 10: Tomorrow Preparation

**Check Tomorrow's Calendar**:
- [ ] Any scheduled maintenance?
- [ ] Customer demos or onboarding calls?
- [ ] Marketing campaigns launching?
- [ ] Expected traffic spikes?

**If Yes to Any Above**:
1. Pre-scale infrastructure
2. Alert on-call team
3. Prepare customer communication
4. Test critical paths

### Step 11: Handoff to On-Call

```bash
# Generate on-call handoff
tensafe-admin oncall handoff --shift night
```

**Handoff Communication** (Post to Slack #oncall):

```
## On-Call Handoff - [DATE]

### System Status: [GREEN/YELLOW/RED]

### Open Issues:
- [Issue 1]
- [Issue 2]

### Watch Items:
- [Item 1]
- [Item 2]

### Tomorrow's Events:
- [Event 1]

### Contact:
- Day shift: @[name]
- Night shift: @[name]
```

---

## PART 5: DASHBOARD REFERENCE GUIDE

### Primary Dashboard Panels

Access at: `https://grafana.tensafe.io/d/tensafe-ops`

#### Panel 1: Service Health Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ SERVICE HEALTH                                                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   API Status    │   DB Status     │   Redis Status              │
│   [GREEN DOT]   │   [GREEN DOT]   │   [GREEN DOT]               │
│   99.99% up     │   99.99% up     │   99.99% up                 │
├─────────────────┴─────────────────┴─────────────────────────────┤
│ Active Pods: 5/5    Queue Depth: 12    GPU Nodes: 3/4 active    │
└─────────────────────────────────────────────────────────────────┘
```

**Prometheus Queries**:
```promql
# API Health
up{job="tensafe-api"} == 1

# Pod count
count(kube_pod_status_ready{namespace="tensafe", condition="true"})

# Queue depth
tensafe_request_queue_depth
```

#### Panel 2: Request Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│ REQUEST METRICS (Last Hour)                                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Requests/sec    │ Error Rate      │ P95 Latency                 │
│     245         │    0.02%        │    67 ms                    │
│ [GRAPH ↗]       │ [GRAPH →]       │ [GRAPH →]                   │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

**Prometheus Queries**:
```promql
# Requests per second
sum(rate(tensafe_http_requests_total[5m]))

# Error rate
sum(rate(tensafe_http_requests_total{status=~"5.."}[5m])) /
sum(rate(tensafe_http_requests_total[5m])) * 100

# P95 latency
histogram_quantile(0.95,
  sum(rate(tensafe_inference_latency_seconds_bucket[5m])) by (le)
)
```

#### Panel 3: Privacy & Training Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│ PRIVACY & TRAINING                                               │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ DP Epsilon      │ HE Operations   │ Training Jobs               │
│ Spent: 2.3/10   │   1.2K/hour     │   Active: 3                 │
│ [GAUGE 23%]     │ [COUNTER]       │   Queued: 7                 │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

**Prometheus Queries**:
```promql
# DP Epsilon spent
max(tensafe_dp_epsilon_spent)

# HE operations per hour
sum(rate(tensafe_he_operations_total[1h])) * 3600

# Active training jobs
tensafe_training_jobs_active
```

#### Panel 4: Infrastructure Resources

```
┌─────────────────────────────────────────────────────────────────┐
│ INFRASTRUCTURE                                                   │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ CPU Usage       │ Memory Usage    │ GPU Utilization             │
│    45%          │     62%         │     73%                     │
│ [GAUGE]         │ [GAUGE]         │ [GAUGE]                     │
├─────────────────┴─────────────────┴─────────────────────────────┤
│ Disk: 45% used (450GB/1TB)    Network: 1.2 Gbps in / 0.8 out   │
└─────────────────────────────────────────────────────────────────┘
```

**Prometheus Queries**:
```promql
# CPU usage
avg(rate(container_cpu_usage_seconds_total{namespace="tensafe"}[5m])) * 100

# Memory usage
avg(container_memory_working_set_bytes{namespace="tensafe"}) /
avg(container_spec_memory_limit_bytes{namespace="tensafe"}) * 100

# GPU utilization
avg(DCGM_FI_DEV_GPU_UTIL{namespace="tensafe"})
```

---

## PART 6: ALERT RESPONSE QUICK REFERENCE

### Critical Alerts (Respond < 15 minutes)

| Alert | Meaning | Immediate Action |
|-------|---------|------------------|
| `TenSafeAPIDown` | API not responding | Check pods, restart if needed |
| `HighErrorRate` > 5% | Many requests failing | Check logs, rollback if recent deploy |
| `DatabaseDown` | PostgreSQL unreachable | Failover to replica |
| `PrivacyBudgetExhausted` | Epsilon > 10 | Stop training, alert customer |

### Warning Alerts (Respond < 1 hour)

| Alert | Meaning | Action |
|-------|---------|--------|
| `HighLatencyP95` > 200ms | Slow responses | Scale up pods |
| `HighCPU` > 80% | Resource pressure | Scale up, check for runaway process |
| `HighMemory` > 85% | Memory pressure | Scale up, check for memory leak |
| `QueueBacklog` > 100 | Requests queuing | Scale up workers |
| `DiskSpace` > 80% | Running out of disk | Clean logs, expand volume |

### Informational Alerts (Review within 24 hours)

| Alert | Meaning | Action |
|-------|---------|--------|
| `NewCustomerSignup` | Someone signed up | Welcome email sent automatically |
| `HighTraining` | Unusual training volume | Verify legitimate usage |
| `CertExpiringSoon` | TLS cert expiring | Renew certificate |

---

## PART 7: DAILY CHECKLIST TEMPLATE

Copy and use this template each day:

```markdown
# TenSafe Daily Operations - [DATE]

## Morning Check (08:00) ✅/❌
- [ ] System status: GREEN / YELLOW / RED
- [ ] Overnight incidents reviewed
- [ ] All services responding
- [ ] Error rate < 0.1%
- [ ] P95 latency < 100ms

## Customer Review (08:15) ✅/❌
- [ ] No P1/P2 tickets open
- [ ] At-risk customers identified: ___
- [ ] Enterprise customers healthy

## Mid-Day Check (10:00) ✅/❌
- [ ] Traffic patterns normal
- [ ] No new alerts
- [ ] Resources within limits

## Afternoon Check (14:00) ✅/❌
- [ ] CPU < 70%
- [ ] Memory < 80%
- [ ] Scaling not needed / SCALED to ___

## End of Day (17:00) ✅/❌
- [ ] Daily metrics recorded
- [ ] Tomorrow prep complete
- [ ] On-call handoff posted

## Key Metrics Today
- DAU: ___
- API Calls: ___
- Uptime: ___%
- New Signups: ___
- MRR Change: $___

## Notes/Issues:
___________________________________
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-04 | Platform Ops | Initial checklist |
