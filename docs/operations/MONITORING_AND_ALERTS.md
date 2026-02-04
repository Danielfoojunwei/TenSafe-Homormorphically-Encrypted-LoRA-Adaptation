# TenSafe Monitoring & Alerting Configuration

**Version**: 1.0.0
**Last Updated**: 2026-02-04
**Purpose**: Complete monitoring stack configuration and alert rules for production operations

---

## Table of Contents

1. [Monitoring Stack Overview](#monitoring-stack-overview)
2. [Alert Severity Levels](#alert-severity-levels)
3. [Complete Alert Rules](#complete-alert-rules)
4. [PagerDuty Integration](#pagerduty-integration)
5. [Slack Integration](#slack-integration)
6. [Daily Automated Reports](#daily-automated-reports)
7. [Dashboard URLs](#dashboard-urls)

---

## Monitoring Stack Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MONITORING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   TenSafe    │───▶│    OTEL     │───▶│  Prometheus  │              │
│  │    Pods      │    │  Collector   │    │              │              │
│  └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                             │                    │                       │
│                             │                    │                       │
│                             ▼                    ▼                       │
│                      ┌──────────────┐    ┌──────────────┐              │
│                      │    Loki      │    │ Alertmanager │              │
│                      │   (Logs)     │    │              │              │
│                      └──────┬───────┘    └──────┬───────┘              │
│                             │                    │                       │
│                             │                    │                       │
│                             ▼                    ▼                       │
│                      ┌──────────────────────────────────┐              │
│                      │           GRAFANA                 │              │
│                      │  ┌─────────────────────────────┐ │              │
│                      │  │ Dashboards  │ Alerts │ Logs │ │              │
│                      │  └─────────────────────────────┘ │              │
│                      └──────────────────────────────────┘              │
│                                     │                                    │
│                    ┌────────────────┼────────────────┐                  │
│                    ▼                ▼                ▼                  │
│             ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│             │PagerDuty │    │  Slack   │    │  Email   │               │
│             └──────────┘    └──────────┘    └──────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Endpoints

| Component | Internal URL | External URL | Port |
|-----------|-------------|--------------|------|
| Prometheus | prometheus.monitoring:9090 | prometheus.tensafe.io | 9090 |
| Grafana | grafana.monitoring:3000 | grafana.tensafe.io | 3000 |
| Alertmanager | alertmanager.monitoring:9093 | alertmanager.tensafe.io | 9093 |
| Loki | loki.monitoring:3100 | - | 3100 |
| OTEL Collector | otel-collector.monitoring:4317 | - | 4317 |

---

## Alert Severity Levels

### Severity Definitions

| Level | Name | Response Time | Who Gets Paged | Examples |
|-------|------|---------------|----------------|----------|
| **P1** | Critical | < 15 min | On-call + Backup + Manager | API down, data loss, security breach |
| **P2** | High | < 1 hour | On-call engineer | Error rate > 5%, latency > 2s |
| **P3** | Medium | < 4 hours | Slack notification | Error rate > 1%, warnings |
| **P4** | Low | Next business day | Email digest | Informational, minor issues |

### Escalation Matrix

```
Time Since Alert    Action
─────────────────────────────────────────────────────
0 min               Alert fires → On-call paged
15 min (P1)         No ack → Backup paged
30 min (P1)         No ack → Manager paged
1 hour (P1/P2)      No ack → All engineers + leadership
2 hours (P1)        No resolution → War room convened
```

### On-Call Rotation

```yaml
# oncall-rotation.yaml
schedules:
  - name: primary
    rotation: weekly
    start_day: monday
    start_time: "09:00"
    timezone: "America/Los_Angeles"
    members:
      - engineer_a
      - engineer_b
      - engineer_c
      - engineer_d

  - name: secondary
    rotation: weekly
    offset: 1  # One week behind primary
    members:
      - engineer_b
      - engineer_c
      - engineer_d
      - engineer_a
```

---

## Complete Alert Rules

### Category 1: Service Availability

```yaml
# prometheus-rules/availability.yaml
groups:
  - name: tensafe.availability
    interval: 30s
    rules:
      # API Down - P1
      - alert: TenSafeAPIDown
        expr: up{job="tensafe-api"} == 0
        for: 1m
        labels:
          severity: critical
          priority: P1
          team: platform
        annotations:
          summary: "CRITICAL: TenSafe API is DOWN"
          description: "The TenSafe API has been unreachable for more than 1 minute."
          runbook_url: "https://runbooks.tensafe.io/api-down"
          dashboard_url: "https://grafana.tensafe.io/d/api-health"

      # Partial Outage - P2
      - alert: TenSafePartialOutage
        expr: |
          count(up{job="tensafe-api"} == 1) /
          count(up{job="tensafe-api"}) < 0.5
        for: 2m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "HIGH: More than 50% of API pods are down"
          description: "{{ $value | humanizePercentage }} of pods are healthy"

      # Database Down - P1
      - alert: PostgreSQLDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
          priority: P1
        annotations:
          summary: "CRITICAL: PostgreSQL database is DOWN"
          runbook_url: "https://runbooks.tensafe.io/db-down"

      # Redis Down - P2
      - alert: RedisDown
        expr: redis_up == 0
        for: 2m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "HIGH: Redis cache is DOWN"

      # Health Check Failing - P2
      - alert: HealthCheckFailing
        expr: |
          probe_success{job="blackbox",target=~".*tensafe.*"} == 0
        for: 3m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "Health check failing for {{ $labels.target }}"
```

### Category 2: Performance

```yaml
# prometheus-rules/performance.yaml
groups:
  - name: tensafe.performance
    interval: 30s
    rules:
      # High Latency P95 - P2
      - alert: HighLatencyP95
        expr: |
          histogram_quantile(0.95,
            sum(rate(tensafe_http_request_duration_seconds_bucket{job="tensafe"}[5m])) by (le)
          ) > 0.2
        for: 5m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "P95 latency is {{ $value | humanizeDuration }}"
          description: "API response time exceeds 200ms at P95"

      # Critical Latency - P1
      - alert: CriticalLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(tensafe_http_request_duration_seconds_bucket{job="tensafe"}[5m])) by (le)
          ) > 1.0
        for: 3m
        labels:
          severity: critical
          priority: P1
        annotations:
          summary: "CRITICAL: P95 latency exceeds 1 second"

      # High Error Rate - P2
      - alert: HighErrorRate
        expr: |
          sum(rate(tensafe_http_requests_total{status=~"5.."}[5m])) /
          sum(rate(tensafe_http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "Error rate is {{ $value | humanizePercentage }}"
          description: "More than 1% of requests are failing with 5xx errors"

      # Critical Error Rate - P1
      - alert: CriticalErrorRate
        expr: |
          sum(rate(tensafe_http_requests_total{status=~"5.."}[5m])) /
          sum(rate(tensafe_http_requests_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
          priority: P1
        annotations:
          summary: "CRITICAL: Error rate exceeds 5%"

      # Request Queue Backup - P2
      - alert: RequestQueueBackup
        expr: tensafe_request_queue_depth > 100
        for: 5m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "Request queue depth is {{ $value }}"
          description: "Requests are queueing - consider scaling up"

      # Training Job Stuck - P3
      - alert: TrainingJobStuck
        expr: |
          increase(tensafe_training_steps_total[30m]) == 0
          and tensafe_training_jobs_active > 0
        for: 30m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "Training job appears stuck"
          description: "No training progress in 30 minutes"
```

### Category 3: Resources

```yaml
# prometheus-rules/resources.yaml
groups:
  - name: tensafe.resources
    interval: 30s
    rules:
      # High CPU - P3
      - alert: HighCPUUsage
        expr: |
          avg(rate(container_cpu_usage_seconds_total{namespace="tensafe"}[5m])) /
          avg(kube_pod_container_resource_limits{namespace="tensafe",resource="cpu"}) > 0.8
        for: 10m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "CPU usage is {{ $value | humanizePercentage }}"
          action: "Consider scaling up pods"

      # Critical CPU - P2
      - alert: CriticalCPUUsage
        expr: |
          avg(rate(container_cpu_usage_seconds_total{namespace="tensafe"}[5m])) /
          avg(kube_pod_container_resource_limits{namespace="tensafe",resource="cpu"}) > 0.95
        for: 5m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "CRITICAL: CPU usage exceeds 95%"

      # High Memory - P3
      - alert: HighMemoryUsage
        expr: |
          avg(container_memory_working_set_bytes{namespace="tensafe"}) /
          avg(kube_pod_container_resource_limits{namespace="tensafe",resource="memory"}) > 0.85
        for: 10m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "Memory usage is {{ $value | humanizePercentage }}"

      # High GPU - P3
      - alert: HighGPUUsage
        expr: avg(DCGM_FI_DEV_GPU_UTIL{namespace="tensafe"}) > 90
        for: 15m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "GPU utilization is {{ $value }}%"
          action: "Consider adding GPU nodes"

      # Disk Space - P2
      - alert: DiskSpaceLow
        expr: |
          (node_filesystem_avail_bytes{mountpoint="/"} /
           node_filesystem_size_bytes{mountpoint="/"}) < 0.15
        for: 10m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "Disk space below 15%"
          action: "Expand volume or clean up"

      # Pod Restarts - P3
      - alert: PodCrashLooping
        expr: |
          increase(kube_pod_container_status_restarts_total{namespace="tensafe"}[1h]) > 5
        for: 5m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "Pod {{ $labels.pod }} has restarted {{ $value }} times in 1 hour"

      # Insufficient Pods - P2
      - alert: InsufficientPods
        expr: |
          count(kube_pod_status_ready{namespace="tensafe",condition="true"}) < 3
        for: 2m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "Only {{ $value }} pods running (minimum: 3)"
```

### Category 4: Privacy & Security

```yaml
# prometheus-rules/privacy-security.yaml
groups:
  - name: tensafe.privacy
    interval: 30s
    rules:
      # Privacy Budget Warning - P3
      - alert: PrivacyBudgetWarning
        expr: tensafe_dp_epsilon_spent > 7
        for: 1m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "Privacy budget (epsilon) at {{ $value }}/10"
          description: "Approaching privacy budget limit"

      # Privacy Budget Critical - P2
      - alert: PrivacyBudgetCritical
        expr: tensafe_dp_epsilon_spent > 9
        for: 1m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "CRITICAL: Privacy budget nearly exhausted ({{ $value }}/10)"
          action: "Stop training to preserve privacy guarantees"

      # Unusual HE Operations - P3
      - alert: UnusualHEOperations
        expr: |
          rate(tensafe_he_operations_total[5m]) >
          2 * avg_over_time(rate(tensafe_he_operations_total[5m])[7d:1h])
        for: 15m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "HE operations rate unusually high"

      # Certificate Expiring - P3
      - alert: TLSCertExpiringSoon
        expr: |
          (probe_ssl_earliest_cert_expiry - time()) / 86400 < 30
        for: 1h
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "TLS certificate expires in {{ $value }} days"

      # Authentication Failures - P3
      - alert: HighAuthFailures
        expr: |
          sum(rate(tensafe_auth_failures_total[5m])) > 10
        for: 5m
        labels:
          severity: medium
          priority: P3
          team: security
        annotations:
          summary: "High authentication failure rate: {{ $value }}/s"
          description: "Possible brute force attempt"

      # Suspicious API Activity - P2
      - alert: SuspiciousAPIActivity
        expr: |
          sum(rate(tensafe_http_requests_total[5m])) by (client_ip) > 1000
        for: 5m
        labels:
          severity: high
          priority: P2
          team: security
        annotations:
          summary: "Unusually high request rate from {{ $labels.client_ip }}"
```

### Category 5: Database

```yaml
# prometheus-rules/database.yaml
groups:
  - name: tensafe.database
    interval: 30s
    rules:
      # High Connections - P3
      - alert: DatabaseConnectionsHigh
        expr: |
          pg_stat_activity_count / pg_settings_max_connections > 0.8
        for: 10m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "Database connections at {{ $value | humanizePercentage }}"

      # Critical Connections - P2
      - alert: DatabaseConnectionsCritical
        expr: |
          pg_stat_activity_count / pg_settings_max_connections > 0.95
        for: 5m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "Database connections nearly exhausted"

      # Replication Lag - P2
      - alert: DatabaseReplicationLag
        expr: pg_replication_lag_seconds > 30
        for: 5m
        labels:
          severity: high
          priority: P2
        annotations:
          summary: "Database replication lag: {{ $value }}s"

      # Slow Queries - P3
      - alert: SlowQueries
        expr: |
          pg_stat_statements_mean_time_seconds{} > 1
        for: 10m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "Slow queries detected (avg > 1s)"

      # Database Size Growing Fast - P4
      - alert: DatabaseGrowthHigh
        expr: |
          deriv(pg_database_size_bytes[1h]) > 100000000  # 100MB/hour
        for: 1h
        labels:
          severity: low
          priority: P4
        annotations:
          summary: "Database growing faster than expected"
```

### Category 6: Business Metrics

```yaml
# prometheus-rules/business.yaml
groups:
  - name: tensafe.business
    interval: 5m
    rules:
      # No Signups Today - P4
      - alert: NoSignupsToday
        expr: |
          increase(tensafe_signups_total[24h]) == 0
          and hour() > 12  # Only alert after noon
        labels:
          severity: low
          priority: P4
          team: growth
        annotations:
          summary: "No new signups today"

      # Unusual Traffic Drop - P3
      - alert: TrafficDropped
        expr: |
          sum(rate(tensafe_http_requests_total[1h])) <
          0.5 * avg_over_time(sum(rate(tensafe_http_requests_total[1h]))[7d:1h])
        for: 30m
        labels:
          severity: medium
          priority: P3
        annotations:
          summary: "Traffic dropped more than 50% from normal"

      # Customer Churned - P4
      - alert: CustomerChurned
        expr: increase(tensafe_customer_churned_total[24h]) > 0
        labels:
          severity: low
          priority: P4
          team: customer-success
        annotations:
          summary: "{{ $value }} customer(s) churned in last 24h"

      # Large Customer Activity Drop - P3
      - alert: EnterpriseCustomerInactive
        expr: |
          tensafe_customer_last_activity_hours{tier="enterprise"} > 168  # 7 days
        labels:
          severity: medium
          priority: P3
          team: customer-success
        annotations:
          summary: "Enterprise customer {{ $labels.customer }} inactive for 7+ days"
```

---

## PagerDuty Integration

### Configuration

```yaml
# alertmanager.yaml
global:
  resolve_timeout: 5m
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    # P1 Critical - Immediate page
    - match:
        priority: P1
      receiver: 'pagerduty-critical'
      continue: true

    # P2 High - Page with delay
    - match:
        priority: P2
      receiver: 'pagerduty-high'
      group_wait: 2m
      continue: true

    # P3 Medium - Slack only
    - match:
        priority: P3
      receiver: 'slack-alerts'

    # P4 Low - Email digest
    - match:
        priority: P4
      receiver: 'email-digest'
      group_wait: 1h
      group_interval: 6h

receivers:
  - name: 'default'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_P1_KEY}'
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'
        details:
          firing: '{{ template "pagerduty.default.instances" .Alerts.Firing }}'

  - name: 'pagerduty-high'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_P2_KEY}'
        severity: error
        description: '{{ .CommonAnnotations.summary }}'

  - name: 'slack-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        color: '{{ if eq .Status "firing" }}danger{{ else }}good{{ end }}'

  - name: 'email-digest'
    email_configs:
      - to: 'ops@tensafe.io'
        send_resolved: true
```

---

## Slack Integration

### Slack Channels

| Channel | Purpose | Alerts |
|---------|---------|--------|
| #alerts | All alerts | P1, P2, P3 |
| #alerts-critical | Critical only | P1 |
| #ops | Operations discussion | Manual posts |
| #oncall | On-call handoffs | Shift changes |
| #incidents | Incident tracking | Active incidents |

### Daily Summary Bot

```python
# slack_daily_summary.py
import os
from datetime import datetime
import requests

SLACK_WEBHOOK = os.environ['SLACK_WEBHOOK_URL']

def post_daily_summary(metrics: dict):
    """Post daily summary to Slack."""
    status_emoji = {
        'green': ':large_green_circle:',
        'yellow': ':large_yellow_circle:',
        'red': ':red_circle:'
    }

    message = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"TenSafe Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*System Status:*\n{status_emoji[metrics['status']]} {metrics['status'].upper()}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Uptime (24h):*\n{metrics['uptime']}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Error Rate:*\n{metrics['error_rate']}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*P95 Latency:*\n{metrics['p95_latency']}ms"
                    }
                ]
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*DAU:*\n{metrics['dau']:,}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*New Signups:*\n{metrics['signups']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*MRR:*\n${metrics['mrr']:,.0f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Incidents:*\n{metrics['incidents']}"
                    }
                ]
            }
        ]
    }

    if metrics.get('alerts'):
        message['blocks'].append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Active Alerts:*\n" + "\n".join(f"• {a}" for a in metrics['alerts'])
            }
        })

    requests.post(SLACK_WEBHOOK, json=message)
```

---

## Daily Automated Reports

### Morning Report (08:00 UTC)

```bash
# Cron: 0 8 * * * /opt/tensafe/scripts/morning_report.sh

#!/bin/bash
set -e

# Generate metrics
tensafe-admin metrics daily-summary --format json > /tmp/metrics.json

# Post to Slack
python3 /opt/tensafe/scripts/slack_daily_summary.py /tmp/metrics.json

# Send email digest
tensafe-admin reports email-digest \
  --to ops@tensafe.io,leadership@tensafe.io \
  --metrics /tmp/metrics.json

echo "Morning report sent at $(date)"
```

### Evening Report (18:00 UTC)

```bash
# Cron: 0 18 * * * /opt/tensafe/scripts/evening_report.sh

#!/bin/bash
set -e

# Generate end-of-day summary
tensafe-admin metrics eod-summary --format json > /tmp/eod_metrics.json

# Post on-call handoff
tensafe-admin oncall generate-handoff --output /tmp/handoff.md

# Post to Slack
cat /tmp/handoff.md | curl -X POST \
  -H 'Content-type: application/json' \
  --data "{\"text\": \"$(cat /tmp/handoff.md)\"}" \
  ${SLACK_ONCALL_WEBHOOK}

echo "Evening report sent at $(date)"
```

### Weekly Report (Monday 09:00 UTC)

```bash
# Cron: 0 9 * * 1 /opt/tensafe/scripts/weekly_report.sh

#!/bin/bash
set -e

# Generate weekly summary
tensafe-admin reports weekly \
  --start "7 days ago" \
  --end "today" \
  --format html \
  --output /tmp/weekly_report.html

# Send to leadership
tensafe-admin reports send \
  --to leadership@tensafe.io \
  --subject "TenSafe Weekly Report - Week $(date +%V)" \
  --body /tmp/weekly_report.html

echo "Weekly report sent at $(date)"
```

---

## Dashboard URLs

### Quick Links

| Dashboard | URL | Refresh |
|-----------|-----|---------|
| Executive Summary | https://grafana.tensafe.io/d/exec-summary | 1 min |
| API Performance | https://grafana.tensafe.io/d/api-perf | 30 sec |
| Infrastructure | https://grafana.tensafe.io/d/infra | 30 sec |
| Privacy Metrics | https://grafana.tensafe.io/d/privacy | 1 min |
| Customer Health | https://grafana.tensafe.io/d/customer-health | 5 min |
| Business Metrics | https://grafana.tensafe.io/d/business | 15 min |
| Alerts Overview | https://grafana.tensafe.io/d/alerts | 1 min |
| On-Call Status | https://grafana.tensafe.io/d/oncall | 5 min |

### Mobile App

Install Grafana mobile app and configure:
1. Server URL: https://grafana.tensafe.io
2. API Key: Generate from Settings > API Keys
3. Enable push notifications for P1/P2 alerts

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-04 | Platform Ops | Initial configuration |
