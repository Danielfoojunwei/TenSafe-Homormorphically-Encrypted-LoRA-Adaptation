# TenSafe Capacity Planning Guide

**Version**: 1.0.0
**Last Updated**: 2026-02-04
**Purpose**: Guide for planning, monitoring, and scaling capacity to serve more customers

---

## Table of Contents

1. [Capacity Planning Overview](#capacity-planning-overview)
2. [Resource Requirements per Customer](#resource-requirements-per-customer)
3. [Current Capacity Assessment](#current-capacity-assessment)
4. [Scaling Decision Framework](#scaling-decision-framework)
5. [Growth Projections](#growth-projections)
6. [Cost Modeling](#cost-modeling)
7. [Capacity Planning Checklist](#capacity-planning-checklist)

---

## Capacity Planning Overview

### Key Questions This Guide Answers

1. **Can we serve more customers?** - Current headroom analysis
2. **When do we need to scale?** - Trigger thresholds
3. **How much will it cost?** - Cost projections
4. **What's the lead time?** - Planning horizons

### Planning Horizons

| Horizon | Look-Ahead | Actions | Review Frequency |
|---------|------------|---------|------------------|
| **Tactical** | 1-7 days | Auto-scaling, pod adjustments | Daily |
| **Operational** | 1-4 weeks | Node scaling, resource upgrades | Weekly |
| **Strategic** | 1-6 months | Architecture changes, new regions | Monthly |
| **Long-term** | 6-24 months | Major investments, new services | Quarterly |

---

## Resource Requirements per Customer

### Per-Customer Resource Consumption

Based on observed usage patterns across customer tiers:

| Resource | Free Tier | Pro ($99) | Business ($499) | Enterprise |
|----------|-----------|-----------|-----------------|------------|
| API Calls/day | 100-500 | 1K-10K | 10K-100K | 100K+ |
| Training Jobs/month | 0-5 | 5-50 | 50-500 | Unlimited |
| Storage (MB) | 10-100 | 100-500 | 500-2000 | 2-10 GB |
| GPU Hours/month | 0 | 1-10 | 10-100 | 100+ |

### Infrastructure Cost per Customer

```
┌─────────────────────────────────────────────────────────────────────────┐
│             MONTHLY INFRASTRUCTURE COST PER CUSTOMER                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Free Tier:     ~$0.50/month   (mostly shared infrastructure)          │
│   Pro Tier:      ~$15/month     (moderate compute + some GPU)           │
│   Business:      ~$80/month     (dedicated compute + GPU hours)         │
│   Enterprise:    ~$200+/month   (dedicated resources, SLA overhead)     │
│                                                                          │
│   Target Gross Margin: 70-80%                                           │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ Tier       │ Price │ Cost  │ Margin │ Target                     │  │
│   ├────────────┼───────┼───────┼────────┼────────────────────────────│  │
│   │ Free       │ $0    │ $0.50 │ -∞     │ Conversion to paid         │  │
│   │ Pro        │ $99   │ $15   │ 85%    │ Self-serve revenue         │  │
│   │ Business   │ $499  │ $80   │ 84%    │ Mid-market expansion       │  │
│   │ Enterprise │ $2K+  │ $200+ │ 80%+   │ High-touch accounts        │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Resource Scaling Factors

| Customer Milestone | Compute Pods | GPU Nodes | DB Size | Est. Monthly Cost |
|--------------------|--------------|-----------|---------|-------------------|
| 100 customers | 3 | 1 | 50 GB | $3,000 |
| 500 customers | 5 | 2 | 200 GB | $8,000 |
| 1,000 customers | 8 | 4 | 500 GB | $15,000 |
| 5,000 customers | 15 | 8 | 2 TB | $45,000 |
| 10,000 customers | 25 | 15 | 5 TB | $90,000 |
| 50,000 customers | 60 | 40 | 20 TB | $350,000 |

---

## Current Capacity Assessment

### How to Check Current Capacity

```bash
# Run capacity assessment
tensafe-admin capacity status

# Or manually check:
kubectl top pods -n tensafe
kubectl top nodes
kubectl get hpa -n tensafe
```

### Capacity Dashboard Metrics

| Metric | Current | Capacity | Utilization | Status |
|--------|---------|----------|-------------|--------|
| API Pods | ___ | ___ max | ___% | |
| CPU Cores | ___ used | ___ total | ___% | |
| Memory | ___ GB | ___ GB | ___% | |
| GPU Nodes | ___ | ___ | ___% | |
| DB Connections | ___ | ___ | ___% | |
| Storage | ___ GB | ___ GB | ___% | |

### Utilization Zones

```
┌────────────────────────────────────────────────────────────────┐
│                    UTILIZATION ZONES                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   0%    20%    40%    60%    70%    80%    90%    100%        │
│   ├──────┼──────┼──────┼──────┼──────┼──────┼──────┤          │
│   │      OPTIMAL      │ PLAN │ SCALE│URGENT│ FULL │          │
│   │   (green zone)    │      │  UP  │      │      │          │
│   └────────────────────┴──────┴──────┴──────┴──────┘          │
│                                                                 │
│   < 40%  : Under-utilized (consider scaling down)              │
│   40-60% : Optimal operating range                              │
│   60-70% : Healthy with room for spikes                        │
│   70-80% : Plan scaling within 2 weeks                         │
│   80-90% : Scale up within 1 week                              │
│   > 90%  : Scale immediately                                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Scaling Decision Framework

### When to Scale Up

| Trigger | Condition | Action | Lead Time |
|---------|-----------|--------|-----------|
| **CPU** | > 70% avg for 1 hour | Add 25% more pods | Immediate |
| **Memory** | > 80% avg for 1 hour | Add 25% more pods | Immediate |
| **Latency** | P95 > 100ms for 30 min | Double pods | Immediate |
| **GPU** | > 85% for 2 hours | Add GPU node | 15-30 min |
| **Storage** | > 80% used | Expand volume | 1 hour |
| **DB Connections** | > 70% | Increase pool/replica | 30 min |
| **Queue Depth** | > 50 for 10 min | Add workers | Immediate |

### When to Scale Down

| Trigger | Condition | Action | Wait Period |
|---------|-----------|--------|-------------|
| **CPU** | < 30% avg for 4 hours | Remove 1 pod | 15 min cooldown |
| **Memory** | < 40% avg for 4 hours | Remove 1 pod | 15 min cooldown |
| **GPU** | < 30% for 8 hours | Remove GPU node | 1 hour |
| **Time-based** | Off-peak hours | Scale to minimum | After 22:00 |

### Scaling Decision Tree

```
                    ┌─────────────────┐
                    │ Current Load?   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │  < 40%   │      │  40-70%  │      │  > 70%   │
    │  Low     │      │  Normal  │      │  High    │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                 │
         ▼                 ▼                 ▼
    Consider           Monitor           Scale Up
    Scale Down         Only              Immediately
         │                 │                 │
         ▼                 ▼                 ▼
    Wait 4+ hours      Check every       ┌──────────┐
    before acting      30 minutes        │ > 90%?   │
                                         └────┬─────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                                    ▼                   ▼
                               Double              Add 25%
                               Capacity            Capacity
```

### Auto-Scaling Configuration

```yaml
# Current auto-scaling settings
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: tensafe-hpa
spec:
  minReplicas: 3          # Never go below 3
  maxReplicas: 50         # Maximum scale
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70    # Scale at 70% CPU
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80    # Scale at 80% Memory
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0    # Scale up immediately
      policies:
      - type: Percent
        value: 100                     # Can double
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min to scale down
      policies:
      - type: Percent
        value: 10                      # Max 10% decrease
        periodSeconds: 60
```

---

## Growth Projections

### Projection Calculator

```python
# Calculate capacity requirements for target customer count

def project_capacity(current_customers: int, target_customers: int):
    """Project infrastructure needs for customer growth."""

    growth_factor = target_customers / current_customers

    # Current baseline (adjust to your actual values)
    current = {
        "api_pods": 5,
        "cpu_cores": 20,
        "memory_gb": 80,
        "gpu_nodes": 2,
        "storage_tb": 0.5,
        "db_connections": 100,
    }

    # Not all resources scale linearly
    scaling_factors = {
        "api_pods": 0.8,       # Economies of scale
        "cpu_cores": 0.8,
        "memory_gb": 0.9,
        "gpu_nodes": 0.7,      # More efficient at scale
        "storage_tb": 1.0,     # Linear with customers
        "db_connections": 0.6, # Connection pooling helps
    }

    projected = {}
    for resource, current_value in current.items():
        factor = scaling_factors[resource]
        projected[resource] = current_value * (growth_factor ** factor)

    return projected

# Example: Project from 500 to 5000 customers
projection = project_capacity(500, 5000)
print(projection)
# {
#   'api_pods': 31.5,
#   'cpu_cores': 126.2,
#   'memory_gb': 504.9,
#   'gpu_nodes': 10.0,
#   'storage_tb': 5.0,
#   'db_connections': 251.2
# }
```

### 12-Month Growth Projection Table

Assuming 15% monthly growth:

| Month | Customers | API Pods | GPU Nodes | Storage | Monthly Cost |
|-------|-----------|----------|-----------|---------|--------------|
| Now | 500 | 5 | 2 | 200 GB | $8,000 |
| +3 mo | 760 | 7 | 3 | 300 GB | $12,000 |
| +6 mo | 1,155 | 10 | 4 | 460 GB | $18,000 |
| +9 mo | 1,756 | 14 | 6 | 700 GB | $28,000 |
| +12 mo | 2,670 | 20 | 9 | 1.1 TB | $42,000 |

### Critical Milestones

| Milestone | When (at 15% growth) | Required Changes |
|-----------|---------------------|------------------|
| 1,000 customers | ~5 months | Add read replica DB |
| 2,500 customers | ~11 months | Multi-AZ deployment |
| 5,000 customers | ~17 months | Sharded database |
| 10,000 customers | ~22 months | Multi-region |

---

## Cost Modeling

### Current Cost Breakdown

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MONTHLY COST BREAKDOWN                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Component           │ Current │ % of Total │ Scaling Behavior         │
│   ────────────────────┼─────────┼────────────┼──────────────────────────│
│   Compute (K8s)       │ $3,000  │    25%     │ Scales with traffic      │
│   GPU Instances       │ $5,000  │    42%     │ Scales with training     │
│   Database (RDS)      │ $1,200  │    10%     │ Step function            │
│   Storage (S3/EBS)    │ $500    │     4%     │ Linear with customers    │
│   Network/CDN         │ $600    │     5%     │ Scales with traffic      │
│   Monitoring/Logging  │ $400    │     3%     │ Semi-fixed               │
│   Security/Compliance │ $300    │     3%     │ Fixed overhead           │
│   Other               │ $1,000  │     8%     │ Various                  │
│   ────────────────────┼─────────┼────────────┼──────────────────────────│
│   TOTAL               │ $12,000 │   100%     │                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cost Optimization Opportunities

| Opportunity | Potential Savings | Effort | Priority |
|-------------|-------------------|--------|----------|
| Reserved instances | 30-40% on compute | Low | High |
| Spot instances for training | 50-70% on GPU | Medium | High |
| Right-sizing pods | 10-20% on compute | Low | Medium |
| Storage tiering | 20-30% on storage | Medium | Medium |
| Cache optimization | 10-15% overall | High | Low |

### Unit Economics Target

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      UNIT ECONOMICS TARGETS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Metric              │ Current │ Target  │ Best-in-Class              │
│   ────────────────────┼─────────┼─────────┼────────────────────────────│
│   Gross Margin        │   75%   │   80%   │     85%                    │
│   Infrastructure/Rev  │   20%   │   15%   │     10%                    │
│   Cost per Customer   │  $24/mo │  $18/mo │    $12/mo                  │
│   LTV:CAC             │   4:1   │   5:1   │     7:1                    │
│                                                                          │
│   Key Levers:                                                           │
│   • Increase ARPU through upsells                                       │
│   • Optimize GPU utilization (batch jobs, spot instances)               │
│   • Improve cache hit rates to reduce compute                           │
│   • Right-size database as traffic patterns become predictable          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Capacity Planning Checklist

### Weekly Capacity Review

```markdown
## Weekly Capacity Review - Week of [DATE]

### Current State
- [ ] Total customers: ___
- [ ] Active users (MAU): ___
- [ ] API calls (weekly): ___
- [ ] Training jobs (weekly): ___

### Resource Utilization
- [ ] CPU avg: ___% (target: 40-60%)
- [ ] Memory avg: ___% (target: 50-70%)
- [ ] GPU avg: ___% (target: 50-80%)
- [ ] Storage: ___% (target: < 70%)
- [ ] DB connections: ___% (target: < 60%)

### Capacity Headroom
- [ ] Days until CPU limit (at current growth): ___
- [ ] Days until memory limit: ___
- [ ] Days until GPU limit: ___
- [ ] Days until storage limit: ___

### Actions Needed
- [ ] Scale up required? Yes / No
- [ ] Scale down possible? Yes / No
- [ ] Cost optimization opportunities: ___

### Upcoming Events
- [ ] Expected traffic spikes: ___
- [ ] Planned maintenance: ___
- [ ] New customer onboarding: ___

### Sign-off
Reviewed by: _______________
Date: _______________
```

### Monthly Capacity Planning

```markdown
## Monthly Capacity Planning - [MONTH YEAR]

### Growth Analysis
- [ ] Customer growth rate: ___%
- [ ] Traffic growth rate: ___%
- [ ] Revenue growth rate: ___%

### Capacity Forecast (Next 3 Months)
| Resource | Current | Month 1 | Month 2 | Month 3 |
|----------|---------|---------|---------|---------|
| Customers | ___ | ___ | ___ | ___ |
| API Pods | ___ | ___ | ___ | ___ |
| GPU Nodes | ___ | ___ | ___ | ___ |
| Storage | ___ | ___ | ___ | ___ |
| Monthly Cost | $___ | $___ | $___ | $___ |

### Infrastructure Changes Needed
- [ ] ________________________________
- [ ] ________________________________
- [ ] ________________________________

### Budget Request
- [ ] Additional monthly spend: $___
- [ ] One-time setup costs: $___
- [ ] Approved: Yes / No / Pending

### Risk Assessment
- [ ] Capacity risks: ___
- [ ] Cost risks: ___
- [ ] Mitigation plans: ___
```

### Quarterly Strategic Review

```markdown
## Quarterly Capacity Strategy - Q_ [YEAR]

### Executive Summary
- Current capacity utilization: ___%
- Projected end-of-quarter: ___%
- Budget status: On track / Over / Under

### Key Decisions Needed
1. ________________________________
2. ________________________________
3. ________________________________

### Architecture Considerations
- [ ] Need multi-region? ___
- [ ] Need database sharding? ___
- [ ] Need new service? ___

### Investment Recommendations
| Investment | Cost | Benefit | Priority | Timeline |
|------------|------|---------|----------|----------|
| ___ | $___ | ___ | ___ | ___ |
| ___ | $___ | ___ | ___ | ___ |
| ___ | $___ | ___ | ___ | ___ |

### Approval
CTO: _______________ Date: ___
CEO: _______________ Date: ___
```

---

## Quick Reference Commands

```bash
# Check current capacity
tensafe-admin capacity status

# Project capacity needs
tensafe-admin capacity project --growth-rate 15 --months 6

# Estimate costs
tensafe-admin capacity cost --customers 5000

# Get scaling recommendations
tensafe-admin capacity recommend

# Export capacity report
tensafe-admin capacity report --format pdf --output capacity-report.pdf
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-04 | Platform Ops | Initial guide |
