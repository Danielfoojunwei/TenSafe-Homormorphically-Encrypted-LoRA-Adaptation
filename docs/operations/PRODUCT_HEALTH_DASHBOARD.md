# TenSafe Product Health Dashboard Specification

**Version**: 1.0.0
**Last Updated**: 2026-02-04
**Purpose**: Complete specification for monitoring product health, customer capacity, and service robustness

---

## Table of Contents

1. [Dashboard Overview](#dashboard-overview)
2. [Tier 1: Service Health (Real-time)](#tier-1-service-health-real-time)
3. [Tier 2: Customer Health (Hourly)](#tier-2-customer-health-hourly)
4. [Tier 3: Business Health (Daily)](#tier-3-business-health-daily)
5. [Tier 4: Capacity Planning (Weekly)](#tier-4-capacity-planning-weekly)
6. [Alert Thresholds Matrix](#alert-thresholds-matrix)
7. [Grafana Dashboard JSON](#grafana-dashboard-json)
8. [CLI Commands Reference](#cli-commands-reference)

---

## Dashboard Overview

### Dashboard Hierarchy

```
Level 0: Executive Summary (Single Page)
    └── Overall Health Score: 0-100
    └── Traffic Light: GREEN / YELLOW / RED
    └── Key Numbers: MRR, DAU, Uptime, Error Rate

Level 1: Service Health (Technical Team)
    └── API Performance
    └── Infrastructure
    └── Security

Level 2: Customer Health (Product/Success Team)
    └── Usage Patterns
    └── Engagement
    └── Churn Risk

Level 3: Business Health (Leadership)
    └── Revenue
    └── Growth
    └── Unit Economics

Level 4: Capacity Planning (Engineering)
    └── Resource Utilization
    └── Scaling Projections
    └── Cost Optimization
```

### Health Score Calculation

```python
# Overall Health Score (0-100)
health_score = (
    service_health * 0.35 +      # API uptime, latency, errors
    customer_health * 0.25 +     # Engagement, satisfaction
    business_health * 0.25 +     # Revenue, growth
    capacity_health * 0.15       # Resource headroom
)

# Traffic Light
if health_score >= 80:
    status = "GREEN"    # Healthy
elif health_score >= 60:
    status = "YELLOW"   # Needs attention
else:
    status = "RED"      # Critical issues
```

---

## Tier 1: Service Health (Real-time)

### 1.1 API Performance Metrics

| Metric | Query | Good | Warning | Critical | Check Frequency |
|--------|-------|------|---------|----------|-----------------|
| **Uptime** | `avg_over_time(up{job="tensafe"}[24h])` | > 99.9% | 99.5-99.9% | < 99.5% | 1 min |
| **Request Rate** | `sum(rate(http_requests_total[5m]))` | > 10/s | 5-10/s | < 5/s | 1 min |
| **Error Rate (5xx)** | `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))` | < 0.1% | 0.1-1% | > 1% | 1 min |
| **Error Rate (4xx)** | `sum(rate(http_requests_total{status=~"4.."}[5m])) / sum(rate(http_requests_total[5m]))` | < 5% | 5-10% | > 10% | 1 min |
| **P50 Latency** | `histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))` | < 50ms | 50-100ms | > 100ms | 1 min |
| **P95 Latency** | `histogram_quantile(0.95, ...)` | < 100ms | 100-200ms | > 200ms | 1 min |
| **P99 Latency** | `histogram_quantile(0.99, ...)` | < 200ms | 200-500ms | > 500ms | 1 min |

### 1.2 Infrastructure Metrics

| Metric | Query | Good | Warning | Critical | Check Frequency |
|--------|-------|------|---------|----------|-----------------|
| **CPU Utilization** | `avg(rate(container_cpu_usage_seconds_total{namespace="tensafe"}[5m])) * 100` | < 60% | 60-80% | > 80% | 1 min |
| **Memory Utilization** | `avg(container_memory_working_set_bytes / container_spec_memory_limit_bytes) * 100` | < 70% | 70-85% | > 85% | 1 min |
| **GPU Utilization** | `avg(DCGM_FI_DEV_GPU_UTIL)` | < 70% | 70-90% | > 90% | 1 min |
| **GPU Memory** | `avg(DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE)` | < 70% | 70-85% | > 85% | 1 min |
| **Disk Usage** | `(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100` | < 70% | 70-85% | > 85% | 5 min |
| **Network In** | `rate(container_network_receive_bytes_total[5m])` | < 1Gbps | 1-5Gbps | > 5Gbps | 1 min |
| **Pod Count** | `count(kube_pod_status_ready{condition="true",namespace="tensafe"})` | >= 3 | 2 | < 2 | 1 min |
| **Pod Restarts** | `sum(increase(kube_pod_container_status_restarts_total{namespace="tensafe"}[1h]))` | 0 | 1-5 | > 5 | 5 min |

### 1.3 Database Metrics

| Metric | Query | Good | Warning | Critical | Check Frequency |
|--------|-------|------|---------|----------|-----------------|
| **DB Connections** | `pg_stat_activity_count / pg_settings_max_connections` | < 70% | 70-85% | > 85% | 1 min |
| **Query Latency (avg)** | `pg_stat_statements_mean_time_seconds` | < 10ms | 10-50ms | > 50ms | 1 min |
| **Slow Queries** | `count(pg_stat_statements_mean_time_seconds > 1)` | 0 | 1-5 | > 5 | 5 min |
| **Replication Lag** | `pg_replication_lag_seconds` | < 1s | 1-10s | > 10s | 1 min |
| **DB Size Growth** | `deriv(pg_database_size_bytes[1h])` | < 100MB/h | 100-500MB/h | > 500MB/h | 1 hour |

### 1.4 Redis Metrics

| Metric | Query | Good | Warning | Critical | Check Frequency |
|--------|-------|------|---------|----------|-----------------|
| **Memory Usage** | `redis_memory_used_bytes / redis_memory_max_bytes` | < 70% | 70-85% | > 85% | 1 min |
| **Hit Rate** | `redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)` | > 95% | 90-95% | < 90% | 1 min |
| **Connected Clients** | `redis_connected_clients` | < 1000 | 1000-5000 | > 5000 | 1 min |
| **Evictions** | `rate(redis_evicted_keys_total[5m])` | 0 | > 0 | > 100/s | 1 min |

### 1.5 Privacy-Specific Metrics

| Metric | Query | Good | Warning | Critical | Check Frequency |
|--------|-------|------|---------|----------|-----------------|
| **DP Epsilon Spent** | `max(tensafe_dp_epsilon_spent)` | < 5 | 5-8 | > 8 (max 10) | 1 min |
| **HE Operations/sec** | `rate(tensafe_he_operations_total[5m])` | < 10K/s | 10-50K/s | > 50K/s | 1 min |
| **HE Latency** | `histogram_quantile(0.95, tensafe_he_operation_latency_bucket)` | < 500us | 500us-1ms | > 1ms | 1 min |
| **Gradient Norm (avg)** | `avg(tensafe_gradient_norm)` | 0.1-10 | < 0.1 or > 10 | < 0.01 or > 100 | 1 min |

---

## Tier 2: Customer Health (Hourly)

### 2.1 Usage Metrics

| Metric | Query | Good | Warning | Critical | Check Frequency |
|--------|-------|------|---------|----------|-----------------|
| **DAU** | SQL: `count(distinct user_id) where activity_date = today` | Growing | Flat | Declining | 1 hour |
| **WAU** | SQL: `count(distinct user_id) where activity_date > today - 7` | Growing | Flat | Declining | 1 hour |
| **MAU** | SQL: `count(distinct user_id) where activity_date > today - 30` | Growing | Flat | Declining | Daily |
| **Stickiness** | `DAU / MAU * 100` | > 25% | 15-25% | < 15% | Daily |
| **API Calls per User** | `sum(api_calls) / count(active_users)` | > 100/day | 50-100/day | < 50/day | 1 hour |
| **Training Jobs per User** | `sum(training_jobs) / count(active_users)` | > 1/week | 0.5-1/week | < 0.5/week | Daily |

### 2.2 Engagement Metrics

| Metric | Definition | Good | Warning | Critical |
|--------|------------|------|---------|----------|
| **Activation Rate** | Users who complete first training / Total signups | > 40% | 20-40% | < 20% |
| **Feature Adoption** | Users using 3+ features / Active users | > 50% | 30-50% | < 30% |
| **Return Rate (D1)** | Users active day after signup / Signups | > 60% | 40-60% | < 40% |
| **Return Rate (D7)** | Users active 7 days after signup / Signups | > 40% | 25-40% | < 25% |
| **Return Rate (D30)** | Users active 30 days after signup / Signups | > 25% | 15-25% | < 15% |

### 2.3 Customer Health Score

```sql
-- Customer Health Score Calculation
WITH customer_metrics AS (
  SELECT
    tenant_id,
    -- Usage score (0-30)
    LEAST(30, (api_calls_30d / 10000.0) * 30) as usage_score,
    -- Engagement score (0-25)
    CASE
      WHEN last_active_date = CURRENT_DATE THEN 25
      WHEN last_active_date > CURRENT_DATE - 7 THEN 20
      WHEN last_active_date > CURRENT_DATE - 14 THEN 15
      WHEN last_active_date > CURRENT_DATE - 30 THEN 10
      ELSE 0
    END as engagement_score,
    -- Feature adoption score (0-25)
    LEAST(25, features_used * 5) as adoption_score,
    -- Support score (0-20)
    CASE
      WHEN open_tickets = 0 THEN 20
      WHEN open_tickets = 1 THEN 15
      WHEN open_tickets <= 3 THEN 10
      ELSE 0
    END as support_score
  FROM customer_summary
)
SELECT
  tenant_id,
  (usage_score + engagement_score + adoption_score + support_score) as health_score,
  CASE
    WHEN (usage_score + engagement_score + adoption_score + support_score) >= 70 THEN 'HEALTHY'
    WHEN (usage_score + engagement_score + adoption_score + support_score) >= 40 THEN 'AT_RISK'
    ELSE 'CRITICAL'
  END as health_status
FROM customer_metrics;
```

### 2.4 Churn Risk Indicators

| Indicator | Weight | How to Detect |
|-----------|--------|---------------|
| No activity 7+ days | 30 | `last_active < now() - 7 days` |
| API calls down 50%+ | 25 | `api_calls_this_week < api_calls_last_week * 0.5` |
| Support tickets up | 20 | `tickets_this_month > tickets_last_month * 2` |
| Team members removed | 15 | `team_size_delta < 0` |
| Failed payment | 10 | `payment_status = 'failed'` |

**Risk Score Calculation**:
```python
risk_score = sum(indicator_weight for indicator in active_indicators)

if risk_score >= 50:
    churn_risk = "HIGH"      # Immediate intervention
elif risk_score >= 25:
    churn_risk = "MEDIUM"    # Schedule check-in
else:
    churn_risk = "LOW"       # Normal monitoring
```

---

## Tier 3: Business Health (Daily)

### 3.1 Revenue Metrics

| Metric | Definition | Good | Warning | Critical |
|--------|------------|------|---------|----------|
| **MRR** | Sum of all active monthly subscriptions | Growing 10%+ MoM | 5-10% MoM | < 5% MoM |
| **ARR** | MRR * 12 | Growing | Flat | Declining |
| **ARPU** | MRR / Active customers | > $100 | $50-100 | < $50 |
| **LTV** | ARPU * Average customer lifetime | > 12x CAC | 3-12x CAC | < 3x CAC |
| **CAC** | Sales & marketing spend / New customers | < $1000 | $1000-3000 | > $3000 |
| **LTV:CAC** | LTV / CAC | > 3:1 | 2-3:1 | < 2:1 |

### 3.2 Growth Metrics

| Metric | Definition | Good | Warning | Critical |
|--------|------------|------|---------|----------|
| **Net Revenue Retention** | (MRR + Expansion - Churn) / MRR | > 120% | 100-120% | < 100% |
| **Gross Revenue Retention** | (MRR - Churn) / MRR | > 95% | 90-95% | < 90% |
| **Churn Rate (Logo)** | Churned customers / Total customers | < 3% | 3-5% | > 5% |
| **Churn Rate (Revenue)** | Churned MRR / Total MRR | < 2% | 2-4% | > 4% |
| **Expansion Rate** | Expansion MRR / Beginning MRR | > 5% | 2-5% | < 2% |

### 3.3 Acquisition Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **New Signups (Daily)** | New accounts created | Track trend |
| **Activation Rate** | Activated / Signups (7 day window) | > 40% |
| **Conversion Rate** | Paid / Activated | > 10% |
| **Time to Value** | Days from signup to first training job | < 3 days |
| **Signup to Paid** | Days from signup to first payment | < 14 days |

### 3.4 Daily Business Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      TENSAFE BUSINESS DASHBOARD                          │
│                         Date: 2026-02-04                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │    MRR      │  │    ARR      │  │    NRR      │  │   Churn     │     │
│  │  $52,400    │  │  $628,800   │  │   118%      │  │   2.1%      │     │
│  │   +12% MoM  │  │   +15% YoY  │  │   [GOOD]    │  │   [GOOD]    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   DAU       │  │   MAU       │  │  Signups    │  │ Conversions │     │
│  │    847      │  │   3,240     │  │     23      │  │      5      │     │
│  │   +5% DoD   │  │   +8% MoM   │  │   +15% WoW  │  │   22% rate  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ REVENUE BREAKDOWN            │ CUSTOMER SEGMENTS                        │
│ ┌────────────────────────┐   │ ┌────────────────────────────────────┐  │
│ │ Free:     1,840 users  │   │ │ Enterprise ($500+): 12 @ $32,400   │  │
│ │ Pro:        289 @ $99  │   │ │ Business ($100-499): 45 @ $14,500  │  │
│ │ Business:    45 @ $499 │   │ │ Pro ($50-99):       289 @ $5,500   │  │
│ │ Enterprise:  12 custom │   │ │ Free:             1,840 @ $0       │  │
│ └────────────────────────┘   │ └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Tier 4: Capacity Planning (Weekly)

### 4.1 Current Capacity Assessment

| Resource | Current | Used | Available | Days Until Full |
|----------|---------|------|-----------|-----------------|
| API Pods | 5 | 3.2 (64%) | 1.8 | N/A (auto-scale) |
| CPU Cores | 20 | 12 (60%) | 8 | N/A |
| Memory | 80 GB | 52 GB (65%) | 28 GB | N/A |
| GPU (A100) | 4 | 2.8 (70%) | 1.2 | ~14 days |
| Database Connections | 100 | 45 (45%) | 55 | ~60 days |
| Storage | 1 TB | 450 GB (45%) | 550 GB | ~45 days |

### 4.2 Growth Projections

```python
# Capacity projection based on growth rate
current_users = 2186
growth_rate = 0.15  # 15% monthly growth

projections = []
for month in range(1, 13):
    projected_users = current_users * (1 + growth_rate) ** month

    # Resource projections (users → resources)
    projected_api_load = projected_users * 50  # API calls/user/day
    projected_gpu_hours = projected_users * 0.5  # GPU hours/user/month
    projected_storage = projected_users * 100  # MB/user

    projections.append({
        'month': month,
        'users': projected_users,
        'api_calls_per_day': projected_api_load,
        'gpu_hours_per_month': projected_gpu_hours,
        'storage_gb': projected_storage / 1000
    })
```

### 4.3 Scaling Thresholds

| Metric | Scale Up Trigger | Scale Up Action | Scale Down Trigger | Scale Down Action |
|--------|------------------|-----------------|-------------------|-------------------|
| CPU | > 70% for 5 min | Add 2 pods | < 30% for 15 min | Remove 1 pod |
| Memory | > 80% for 5 min | Add 2 pods | < 40% for 15 min | Remove 1 pod |
| GPU | > 85% for 10 min | Add GPU node | < 30% for 1 hour | Remove GPU node |
| Latency P95 | > 100ms for 5 min | Add 4 pods | < 50ms for 30 min | Remove 2 pods |
| Queue Depth | > 50 for 2 min | Add 4 pods | < 10 for 15 min | Remove 2 pods |

### 4.4 Cost Projections

| Resource | Current Monthly | Projected (3 mo) | Projected (6 mo) | Projected (12 mo) |
|----------|-----------------|------------------|------------------|-------------------|
| Compute (K8s) | $5,000 | $6,500 | $8,500 | $14,000 |
| GPU | $8,000 | $12,000 | $18,000 | $32,000 |
| Database | $1,500 | $2,000 | $3,000 | $5,000 |
| Storage | $500 | $700 | $1,000 | $1,800 |
| Network | $800 | $1,100 | $1,500 | $2,500 |
| **Total** | **$15,800** | **$22,300** | **$32,000** | **$55,300** |

### 4.5 Capacity Planning Checklist (Weekly)

```markdown
## Weekly Capacity Review - [DATE]

### Current Utilization
- [ ] CPU: ___% (target < 70%)
- [ ] Memory: ___% (target < 80%)
- [ ] GPU: ___% (target < 85%)
- [ ] Storage: ___% (target < 80%)
- [ ] DB Connections: ___% (target < 70%)

### Growth This Week
- [ ] New users: ___
- [ ] New enterprise customers: ___
- [ ] API calls change: ___%
- [ ] Training jobs change: ___%

### Bottlenecks Identified
- [ ] ________________________
- [ ] ________________________

### Scaling Actions Needed
- [ ] ________________________
- [ ] ________________________

### Cost Impact
- [ ] Estimated additional cost: $_____/month
- [ ] Approved: Yes / No / Pending
```

---

## Alert Thresholds Matrix

### Complete Alert Configuration

```yaml
# alertmanager-config.yaml
groups:
  - name: tensafe-critical
    rules:
      # Service Down
      - alert: TenSafeAPIDown
        expr: up{job="tensafe-api"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "TenSafe API is down"
          runbook: "https://runbooks.tensafe.io/api-down"

      # High Error Rate
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{job="tensafe",status=~"5.."}[5m])) /
          sum(rate(http_requests_total{job="tensafe"}[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 1%: {{ $value | humanizePercentage }}"

      # High Latency
      - alert: HighLatencyP95
        expr: |
          histogram_quantile(0.95,
            sum(rate(tensafe_http_request_duration_seconds_bucket[5m])) by (le)
          ) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 200ms: {{ $value | humanizeDuration }}"

      # Privacy Budget
      - alert: PrivacyBudgetHigh
        expr: tensafe_dp_epsilon_spent > 7
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Privacy budget (epsilon) above 7: {{ $value }}"

      - alert: PrivacyBudgetExhausted
        expr: tensafe_dp_epsilon_spent > 9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Privacy budget nearly exhausted: {{ $value }}/10"

      # Resource Alerts
      - alert: HighCPU
        expr: |
          avg(rate(container_cpu_usage_seconds_total{namespace="tensafe"}[5m])) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "CPU usage above 80%"

      - alert: HighMemory
        expr: |
          avg(container_memory_working_set_bytes{namespace="tensafe"} /
              container_spec_memory_limit_bytes{namespace="tensafe"}) > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Memory usage above 85%"

      - alert: HighGPU
        expr: avg(DCGM_FI_DEV_GPU_UTIL) > 90
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU utilization above 90%"

      # Database Alerts
      - alert: DatabaseConnectionsHigh
        expr: pg_stat_activity_count / pg_settings_max_connections > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connections above 85%"

      - alert: DatabaseReplicationLag
        expr: pg_replication_lag_seconds > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database replication lag: {{ $value }}s"

      # Pod Alerts
      - alert: PodCrashLooping
        expr: |
          increase(kube_pod_container_status_restarts_total{namespace="tensafe"}[1h]) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod {{ $labels.pod }} restarting frequently"

      - alert: InsufficientPods
        expr: |
          count(kube_pod_status_ready{namespace="tensafe",condition="true"}) < 3
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Less than 3 pods running"

  - name: tensafe-business
    rules:
      # Churn Alert
      - alert: HighChurnRate
        expr: tensafe_churn_rate_30d > 5
        for: 1h
        labels:
          severity: warning
          team: customer-success
        annotations:
          summary: "Monthly churn rate above 5%"

      # Revenue Alert
      - alert: MRRDecline
        expr: tensafe_mrr < tensafe_mrr offset 1d
        for: 24h
        labels:
          severity: warning
          team: leadership
        annotations:
          summary: "MRR declined from yesterday"
```

---

## CLI Commands Reference

### Quick Health Check Commands

```bash
# Overall system health
tensafe-admin health

# Service health only
tensafe-admin health --service

# Customer health summary
tensafe-admin health --customers

# Business metrics
tensafe-admin health --business

# Full report
tensafe-admin health --full --output json > health-report.json
```

### Monitoring Commands

```bash
# Real-time metrics
tensafe-admin metrics watch

# Specific metric
tensafe-admin metrics get --name "api_latency_p95"

# Export metrics
tensafe-admin metrics export --start "7 days ago" --end "now" --format csv

# Compare periods
tensafe-admin metrics compare --period1 "last week" --period2 "this week"
```

### Customer Commands

```bash
# List at-risk customers
tensafe-admin customers at-risk

# Customer health score
tensafe-admin customers health --tenant-id <id>

# Usage report
tensafe-admin customers usage --tenant-id <id> --period "30 days"

# Churn analysis
tensafe-admin customers churn-analysis --period "90 days"
```

### Capacity Commands

```bash
# Current capacity
tensafe-admin capacity status

# Projection
tensafe-admin capacity project --growth-rate 15 --months 6

# Scaling simulation
tensafe-admin capacity simulate --users 10000

# Cost estimate
tensafe-admin capacity cost --users 10000
```

---

## Grafana Dashboard JSON

The complete Grafana dashboard configuration is available at:

```
deploy/grafana/dashboards/tensafe-product-health.json
```

Import this dashboard to get all panels described in this document.

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-04 | Platform Ops | Initial specification |
