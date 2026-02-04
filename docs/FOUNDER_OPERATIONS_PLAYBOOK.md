# TenSafe Founder Operations Playbook

A comprehensive guide for founders and leadership to track data, understand user patterns, and run operations effectively.

## Table of Contents

1. [Daily Operations Dashboard](#daily-operations-dashboard)
2. [Key Metrics to Monitor](#key-metrics-to-monitor)
3. [User Behavior Analytics](#user-behavior-analytics)
4. [Revenue Operations](#revenue-operations)
5. [Product Health](#product-health)
6. [Alerting & Incident Response](#alerting--incident-response)
7. [SQL Queries for Common Questions](#sql-queries-for-common-questions)
8. [Tool Integrations](#tool-integrations)

---

## Daily Operations Dashboard

### Morning Checklist (5 minutes)

```python
from tensorguard.analytics import FounderDashboard

dashboard = FounderDashboard()
summary = dashboard.get_executive_summary()

print(f"MRR: {summary['summary']['mrr']}")
print(f"MRR Growth: {summary['summary']['mrr_growth']}")
print(f"Active Users: {summary['summary']['active_users']}")
print(f"Churn Rate: {summary['summary']['churn_rate']}")
print(f"Uptime: {summary['summary']['uptime']}")
print(f"Status: {summary['status']}")

if summary['alerts']:
    print("\n⚠️ ALERTS:")
    for alert in summary['alerts']:
        print(f"  - {alert}")
```

### What to Look For

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| MRR Growth | > 10% MoM | 5-10% MoM | < 5% MoM |
| Churn Rate | < 3% | 3-5% | > 5% |
| Uptime | > 99.9% | 99.5-99.9% | < 99.5% |
| Error Rate | < 0.5% | 0.5-1% | > 1% |
| P99 Latency | < 200ms | 200-500ms | > 500ms |

---

## Key Metrics to Monitor

### Business Metrics (Weekly Review)

```python
from tensorguard.analytics import BusinessMetrics
from datetime import datetime, timedelta

biz = BusinessMetrics()

# Revenue metrics
print("=== Revenue Health ===")
print(f"MRR: {biz.get_mrr().formatted_value}")
print(f"ARR: {biz.get_arr().formatted_value}")
print(f"ARPU: {biz.get_arpu().formatted_value}")
print(f"LTV: {biz.get_ltv().formatted_value}")

# Retention metrics
now = datetime.utcnow()
print("\n=== Retention ===")
print(f"Net Revenue Retention: {biz.get_net_revenue_retention(now - timedelta(days=90), now).formatted_value}")
print(f"Churn Rate: {biz.get_churn_rate(now - timedelta(days=30), now).formatted_value}")
```

### User Metrics (Daily Review)

```python
from tensorguard.analytics import UserMetrics

users = UserMetrics()

print("=== Engagement ===")
print(f"DAU: {users.get_active_users('day').value:,}")
print(f"WAU: {users.get_active_users('week').value:,}")
print(f"MAU: {users.get_active_users('month').value:,}")

# Stickiness (DAU/MAU ratio)
dau = users.get_active_users('day').value
mau = users.get_active_users('month').value
stickiness = (dau / mau * 100) if mau > 0 else 0
print(f"Stickiness: {stickiness:.1f}%")

# Feature adoption
print("\n=== Privacy Feature Adoption ===")
features = users.get_privacy_feature_usage()
for name, metric in features.items():
    print(f"  {name}: {metric.formatted_value}")
```

### Metric Definitions

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| **MRR** | Monthly Recurring Revenue | Core revenue health |
| **ARR** | Annual Recurring Revenue (MRR × 12) | Valuation metric |
| **ARPU** | Average Revenue Per User | Monetization efficiency |
| **LTV** | Lifetime Value | Customer worth |
| **NRR** | Net Revenue Retention | Expansion vs churn |
| **DAU/MAU** | Daily/Monthly Active Users | Product engagement |
| **Stickiness** | DAU/MAU ratio | Product habit-forming |

---

## User Behavior Analytics

### Understanding the User Journey

```python
from tensorguard.analytics import UserMetrics

users = UserMetrics()
journey = users.get_user_journey_metrics()

print("=== User Funnel (Last 7 Days) ===")
print(f"Signups: {journey['signups_7d']}")
print(f"Activated: {journey['activated_7d']} ({journey['activation_rate']:.1f}%)")
print(f"Converted: {journey['converted_7d']} ({journey['conversion_rate']:.1f}%)")

# Identify bottlenecks
if journey['activation_rate'] < 30:
    print("\n⚠️ Low activation - review onboarding flow")
if journey['conversion_rate'] < 5:
    print("⚠️ Low conversion - review pricing/value prop")
```

### Feature Adoption Analysis

```python
# Which privacy features are users adopting?
features = users.get_privacy_feature_usage()

print("=== Privacy Feature Adoption ===")
for feature, metric in features.items():
    adoption = metric.value
    status = "✅" if adoption > 50 else "⚠️" if adoption > 20 else "❌"
    print(f"{status} {feature}: {metric.formatted_value}")
```

### Cohort Analysis

```python
from datetime import datetime, timedelta

users = UserMetrics()

# D1, D7, D30 retention by cohort
print("=== Retention by Cohort ===")
for days_ago in [7, 14, 21, 28]:
    cohort_date = datetime.utcnow() - timedelta(days=days_ago)

    d1 = users.get_retention_rate(cohort_date, 1).value
    d7 = users.get_retention_rate(cohort_date, 7).value

    print(f"Cohort {cohort_date.strftime('%Y-%m-%d')}:")
    print(f"  D1: {d1:.1f}%  D7: {d7:.1f}%")
```

---

## Revenue Operations

### Subscription Analytics

```sql
-- Active subscriptions by plan
SELECT
    plan_id,
    COUNT(*) as subscribers,
    SUM(monthly_amount) as mrr
FROM subscriptions
WHERE status = 'active'
GROUP BY plan_id
ORDER BY mrr DESC;

-- Recent upgrades/downgrades
SELECT
    tenant_id,
    old_plan,
    new_plan,
    changed_at
FROM plan_changes
WHERE changed_at > NOW() - INTERVAL '7 days'
ORDER BY changed_at DESC;
```

### Churn Analysis

```sql
-- Churned customers last 30 days
SELECT
    t.name as company,
    t.plan_id as plan,
    t.mrr as lost_mrr,
    t.churned_at,
    t.churn_reason
FROM tenants t
WHERE t.status = 'churned'
  AND t.churned_at > NOW() - INTERVAL '30 days'
ORDER BY t.mrr DESC;

-- Churn by reason
SELECT
    churn_reason,
    COUNT(*) as count,
    SUM(mrr) as lost_mrr
FROM tenants
WHERE status = 'churned'
  AND churned_at > NOW() - INTERVAL '90 days'
GROUP BY churn_reason
ORDER BY lost_mrr DESC;
```

### Revenue Forecasting

```python
from tensorguard.analytics import BusinessMetrics
from datetime import datetime, timedelta

biz = BusinessMetrics()

# Simple MRR projection based on growth rate
current_mrr = biz.get_mrr().value
growth_rate = 0.10  # 10% monthly growth

print("=== 12-Month MRR Projection ===")
projected_mrr = current_mrr
for month in range(1, 13):
    projected_mrr *= (1 + growth_rate)
    print(f"Month {month}: ${projected_mrr:,.0f}")

print(f"\nProjected ARR: ${projected_mrr * 12:,.0f}")
```

---

## Product Health

### API Performance Monitoring

```python
from tensorguard.analytics import OperationalMetrics

ops = OperationalMetrics()
api = ops.get_api_metrics()

print("=== API Health ===")
print(f"Requests/min: {api['requests_per_minute'].value:,.0f}")
print(f"Error Rate: {api['error_rate'].formatted_value}")
print(f"P50 Latency: {api['p50_latency'].value:.0f}ms")
print(f"P99 Latency: {api['p99_latency'].value:.0f}ms")

# Check SLA compliance
if api['error_rate'].value > 0.1:
    print("\n⚠️ Error rate above SLA threshold (0.1%)")
if api['p99_latency'].value > 500:
    print("⚠️ P99 latency above target (500ms)")
```

### Infrastructure Health

```python
infra = ops.get_infrastructure_metrics()

print("=== Infrastructure ===")
print(f"CPU Utilization: {infra['cpu_utilization'].formatted_value}")
print(f"Memory Utilization: {infra['memory_utilization'].formatted_value}")
print(f"GPU Utilization: {infra['gpu_utilization'].formatted_value}")
print(f"Disk Usage: {infra['disk_usage'].formatted_value}")

# Capacity planning alerts
if infra['cpu_utilization'].value > 70:
    print("\n⚠️ Consider scaling - CPU above 70%")
if infra['gpu_utilization'].value > 80:
    print("⚠️ GPU capacity tight - plan for more GPUs")
```

### Training Job Analytics

```sql
-- Training jobs by status (last 24h)
SELECT
    status,
    COUNT(*) as count,
    AVG(duration_seconds) as avg_duration,
    SUM(compute_cost) as total_cost
FROM training_jobs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY status;

-- Top users by compute usage
SELECT
    t.name as tenant,
    COUNT(j.id) as jobs,
    SUM(j.compute_cost) as total_cost
FROM training_jobs j
JOIN tenants t ON j.tenant_id = t.id
WHERE j.created_at > NOW() - INTERVAL '7 days'
GROUP BY t.id, t.name
ORDER BY total_cost DESC
LIMIT 20;
```

---

## Alerting & Incident Response

### Setting Up Alerts

```python
from tensorguard.analytics import FounderDashboard

dashboard = FounderDashboard()

# Define alert thresholds
ALERT_THRESHOLDS = {
    'churn_rate': 5.0,      # Alert if > 5%
    'error_rate': 1.0,      # Alert if > 1%
    'p99_latency': 500,     # Alert if > 500ms
    'cpu_util': 80,         # Alert if > 80%
    'gpu_util': 90,         # Alert if > 90%
}

def check_alerts():
    alerts = []

    summary = dashboard.get_executive_summary()
    ops = dashboard.get_operations_dashboard()

    # Check each threshold
    if ops.metrics.get('error_rate', 0) > ALERT_THRESHOLDS['error_rate']:
        alerts.append(f"🚨 High error rate: {ops.metrics['error_rate']}")

    if float(ops.metrics.get('p99_latency', '0').replace('ms', '')) > ALERT_THRESHOLDS['p99_latency']:
        alerts.append(f"🚨 High latency: {ops.metrics['p99_latency']}")

    return alerts
```

### Incident Severity Levels

| Level | Criteria | Response Time | Examples |
|-------|----------|---------------|----------|
| **P1** | Service down | < 15 min | API unreachable, data loss |
| **P2** | Major degradation | < 1 hour | Error rate > 5%, latency > 2s |
| **P3** | Minor issues | < 4 hours | Error rate > 1%, feature broken |
| **P4** | Low priority | Next business day | UI bug, docs issue |

### Escalation Path

1. **P1/P2**: Page on-call engineer → Slack #incidents → Founder notification
2. **P3**: Slack #engineering → Review in daily standup
3. **P4**: Create GitHub issue → Prioritize in sprint

---

## SQL Queries for Common Questions

### "How many users signed up this week?"

```sql
SELECT
    DATE(created_at) as date,
    COUNT(*) as signups
FROM users
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY DATE(created_at)
ORDER BY date;
```

### "What's our conversion funnel?"

```sql
WITH funnel AS (
    SELECT
        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as signups,
        COUNT(*) FILTER (WHERE activated_at IS NOT NULL AND created_at > NOW() - INTERVAL '30 days') as activated,
        COUNT(*) FILTER (WHERE subscribed_at IS NOT NULL AND created_at > NOW() - INTERVAL '30 days') as converted
    FROM users
)
SELECT
    signups,
    activated,
    converted,
    ROUND(activated::numeric / NULLIF(signups, 0) * 100, 1) as activation_rate,
    ROUND(converted::numeric / NULLIF(activated, 0) * 100, 1) as conversion_rate
FROM funnel;
```

### "Who are our power users?"

```sql
SELECT
    u.email,
    t.name as company,
    COUNT(tj.id) as training_jobs,
    SUM(tj.compute_cost) as total_spend,
    MAX(tj.created_at) as last_activity
FROM users u
JOIN tenants t ON u.tenant_id = t.id
JOIN training_jobs tj ON tj.user_id = u.id
WHERE tj.created_at > NOW() - INTERVAL '30 days'
GROUP BY u.id, u.email, t.name
ORDER BY total_spend DESC
LIMIT 25;
```

### "Which features are most used?"

```sql
SELECT
    feature_name,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(*) as total_uses,
    COUNT(*) / 30.0 as uses_per_day
FROM feature_usage_events
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY feature_name
ORDER BY unique_users DESC;
```

### "What's causing churn?"

```sql
SELECT
    churn_reason,
    COUNT(*) as count,
    ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 1) as percentage
FROM tenants
WHERE status = 'churned'
  AND churned_at > NOW() - INTERVAL '90 days'
GROUP BY churn_reason
ORDER BY count DESC;
```

---

## Tool Integrations

### Grafana Dashboard Setup

```yaml
# grafana-dashboard.yaml
apiVersion: 1
providers:
  - name: TenSafe
    folder: Business Metrics
    type: file
    options:
      path: /var/lib/grafana/dashboards

# Connect to Prometheus metrics
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090

  - name: PostgreSQL
    type: postgres
    url: postgres:5432
    database: tensafe
```

### Metabase Queries

Set up these saved questions in Metabase:

1. **Daily MRR Trend** - Line chart of MRR over time
2. **User Funnel** - Funnel visualization of signup → activation → conversion
3. **Feature Adoption** - Bar chart of privacy feature usage
4. **Churn Analysis** - Pie chart of churn reasons
5. **Revenue by Plan** - Stacked bar chart of revenue per plan

### Slack Integration

```python
import requests
from tensorguard.analytics import FounderDashboard

def post_daily_summary():
    dashboard = FounderDashboard()
    summary = dashboard.get_executive_summary()

    message = f"""
*Daily TenSafe Summary*
• MRR: {summary['summary']['mrr']} ({summary['summary']['mrr_growth']})
• Active Users: {summary['summary']['active_users']:,}
• Churn Rate: {summary['summary']['churn_rate']}
• Uptime: {summary['summary']['uptime']}
• Status: {summary['status'].upper()}
"""

    if summary['alerts']:
        message += "\n*Alerts:*\n"
        for alert in summary['alerts']:
            message += f"• {alert}\n"

    requests.post(
        "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        json={"text": message}
    )
```

### Export to CSV

```python
from tensorguard.analytics import FounderDashboard

dashboard = FounderDashboard()

# Export for spreadsheet analysis
csv_data = dashboard.export_metrics(format="csv")
with open("metrics_export.csv", "w") as f:
    f.write(csv_data)

# Export for data warehouse
json_data = dashboard.export_metrics(format="json")
# Send to BigQuery, Snowflake, etc.
```

---

## Weekly Review Checklist

### Monday Morning (30 min)

- [ ] Review executive summary dashboard
- [ ] Check weekend alerts and incidents
- [ ] Review MRR and growth metrics
- [ ] Check churn from last week
- [ ] Review feature adoption trends

### Friday Afternoon (15 min)

- [ ] Export weekly metrics
- [ ] Note any concerning trends
- [ ] Plan next week's priorities
- [ ] Send weekly summary to team

### Monthly Deep Dive (2 hours)

- [ ] Full cohort analysis
- [ ] Revenue forecast update
- [ ] Capacity planning review
- [ ] Competitor monitoring
- [ ] NPS/CSAT review
- [ ] Strategic metrics review with leadership

---

## Quick Reference

### Dashboard URLs

| Dashboard | URL |
|-----------|-----|
| Executive Summary | `/admin/dashboard` |
| Revenue Metrics | `/admin/revenue` |
| User Analytics | `/admin/users` |
| Operations | `/admin/ops` |
| Grafana | `grafana.internal:3000` |
| Prometheus | `prometheus.internal:9090` |

### Key Contacts

| Role | Responsibility |
|------|----------------|
| On-call Engineer | P1/P2 incidents |
| Engineering Lead | Technical decisions |
| Product Lead | Feature priorities |
| Finance | Revenue questions |

### Useful Commands

```bash
# Get current system status
curl https://api.tensafe.io/status

# Check recent errors
kubectl logs -l app=tensafe-api --tail=100 | grep ERROR

# View real-time metrics
watch -n 5 'curl -s localhost:9090/metrics | grep tensafe_'
```
