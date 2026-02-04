# TenSafe Incident Response Runbook

**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Document Owner**: Platform Operations Team

---

## Table of Contents

1. [Overview](#overview)
2. [Incident Declaration](#incident-declaration)
3. [Severity Classification](#severity-classification)
4. [On-Call Procedures](#on-call-procedures)
5. [Incident Response Process](#incident-response-process)
6. [Communication Templates](#communication-templates)
7. [Escalation Matrix](#escalation-matrix)
8. [Post-Mortem Template](#post-mortem-template)
9. [Common Incident Playbooks](#common-incident-playbooks)

---

## Overview

This runbook defines the incident response procedures for TenSafe. An incident is any unplanned interruption or reduction in service quality that impacts customers or business operations.

### Key Contacts

| Role | Primary | Backup | Contact |
|------|---------|--------|---------|
| Incident Commander | On-call Engineer | Engineering Manager | PagerDuty |
| Communications Lead | On-call Engineer | Product Manager | Slack #incident-comms |
| Technical Lead | On-call Engineer | Senior Engineer | PagerDuty |
| Executive Sponsor | VP Engineering | CTO | Emergency hotline |

### Communication Channels

| Channel | Purpose |
|---------|---------|
| PagerDuty | Alert routing and escalation |
| #incident-active (Slack) | Active incident coordination |
| #incident-comms (Slack) | Stakeholder updates |
| Status Page | Customer-facing updates |
| War Room (Zoom) | Voice coordination for P1/P2 |

---

## Incident Declaration

### When to Declare an Incident

Declare an incident when:

- [ ] Service is completely unavailable
- [ ] Error rate exceeds 1% for 5+ minutes
- [ ] P95 latency exceeds 500ms for 5+ minutes
- [ ] Data integrity issue suspected
- [ ] Security breach suspected
- [ ] Customer-reported widespread outage
- [ ] Privacy/compliance violation detected

### How to Declare an Incident

#### Option 1: PagerDuty

```
Page on-call: Use PagerDuty mobile app or web interface
Service: TenSafe Production
```

#### Option 2: Slack

```
/incident create
Title: [Brief description]
Severity: P1/P2/P3/P4
```

#### Option 3: Manual Declaration

```bash
# In #incident-active Slack channel
@oncall Declaring incident: [Brief description]
Severity: P[1-4]
Initial symptoms: [What you observed]
```

---

## Severity Classification

### Severity Definitions

| Severity | Name | Impact | Response Time | Example |
|----------|------|--------|---------------|---------|
| **P1** | Critical | Complete service outage, data loss, security breach | 15 minutes | API completely down, data breach |
| **P2** | High | Major feature unavailable, significant degradation | 30 minutes | Training API down, 50%+ error rate |
| **P3** | Medium | Minor feature unavailable, performance degradation | 4 hours | Metrics not collecting, slow responses |
| **P4** | Low | Cosmetic issues, minor bugs | 24 hours | Dashboard display issue |

### Severity Assessment Criteria

#### P1 - Critical

- [ ] Complete service unavailability
- [ ] Data loss or corruption confirmed
- [ ] Security breach confirmed
- [ ] Privacy violation (PII exposure)
- [ ] Revenue-impacting outage
- [ ] All customers affected

#### P2 - High

- [ ] Major feature completely unavailable
- [ ] Error rate > 10%
- [ ] P95 latency > 5x normal
- [ ] Significant customer impact (> 25% of users)
- [ ] DP budget tracking unavailable
- [ ] HE-LoRA operations failing

#### P3 - Medium

- [ ] Minor feature unavailable
- [ ] Error rate 1-10%
- [ ] P95 latency 2-5x normal
- [ ] Limited customer impact (< 25% of users)
- [ ] Observability degraded
- [ ] Non-critical automation failing

#### P4 - Low

- [ ] Cosmetic issues
- [ ] Internal tooling issues
- [ ] Documentation errors
- [ ] Minor performance degradation
- [ ] Single customer non-critical issue

---

## On-Call Procedures

### On-Call Responsibilities

1. **Acknowledge alerts** within 5 minutes
2. **Assess severity** and escalate if needed
3. **Start incident process** for P1/P2
4. **Document actions** taken
5. **Hand off** to next on-call with context

### On-Call Rotation

| Week | Primary | Secondary |
|------|---------|-----------|
| Odd weeks | Engineer A | Engineer B |
| Even weeks | Engineer B | Engineer C |
| Holidays | Volunteer + compensation | Manager backup |

### Initial Response Checklist

When alerted:

- [ ] Acknowledge the alert in PagerDuty
- [ ] Check Grafana dashboards for impact
- [ ] Review recent changes (deployments, config changes)
- [ ] Assess severity level
- [ ] If P1/P2: Declare incident, join war room
- [ ] If P3/P4: Begin troubleshooting, document in ticket

### Quick Assessment Commands

```bash
# Check service status
kubectl get pods -n tensafe
kubectl get deployments -n tensafe

# Check recent events
kubectl get events -n tensafe --sort-by='.lastTimestamp' | tail -20

# Check error logs
kubectl logs -l app.kubernetes.io/name=tensafe -n tensafe --tail=100 | grep -i error

# Check health endpoints
kubectl exec -it deployment/tensafe-server -n tensafe -- curl -s http://localhost:8000/health

# Check metrics
kubectl exec -it deployment/tensafe-server -n tensafe -- curl -s http://localhost:9090/metrics | grep tensafe_
```

---

## Incident Response Process

### Phase 1: Detection & Triage (0-15 minutes)

1. **Alert received** - Acknowledge within 5 minutes
2. **Initial assessment**
   - What is broken?
   - Who is affected?
   - What is the blast radius?
3. **Severity assignment**
4. **Incident declaration** (if P1/P2)
5. **Assemble response team**

### Phase 2: Investigation (15-60 minutes)

1. **Establish communication**
   - Open war room (P1/P2)
   - Create incident channel
   - Notify stakeholders
2. **Gather data**
   - Review dashboards
   - Check logs
   - Review recent changes
3. **Form hypothesis**
4. **Test hypothesis**
5. **Identify root cause or mitigations**

### Phase 3: Mitigation (Ongoing)

1. **Implement mitigation**
   - Rollback if deployment-related
   - Scale if capacity-related
   - Failover if infrastructure-related
2. **Validate mitigation**
3. **Communicate progress**
4. **Monitor for recurrence**

### Phase 4: Resolution

1. **Confirm service restored**
2. **Validate with monitoring**
3. **Update status page**
4. **Notify stakeholders**
5. **Schedule post-mortem**

### Phase 5: Post-Incident

1. **Conduct post-mortem** (within 48 hours for P1/P2)
2. **Document findings**
3. **Create action items**
4. **Share learnings**
5. **Track remediation**

---

## Communication Templates

### Internal Notification (Slack)

#### Incident Declared
```
:rotating_light: INCIDENT DECLARED - P[1/2/3]

**Summary**: [Brief description of the issue]
**Impact**: [Who/what is affected]
**Status**: Investigating

**Incident Commander**: @[name]
**War Room**: [link if P1/P2]

Updates will be posted every [15/30] minutes.
```

#### Incident Update
```
:warning: INCIDENT UPDATE - P[X] - [HH:MM elapsed]

**Current Status**: [Investigating/Mitigating/Monitoring]
**What we know**: [Latest findings]
**What we're doing**: [Current actions]
**Next update**: [time]
```

#### Incident Resolved
```
:white_check_mark: INCIDENT RESOLVED - P[X]

**Summary**: [What happened]
**Duration**: [Start time] - [End time] ([duration])
**Impact**: [Summary of customer impact]
**Resolution**: [How it was fixed]

Post-mortem scheduled for [date/time].
```

### External Communication (Status Page)

#### Investigating
```
Title: Investigating elevated error rates

We are currently investigating reports of elevated error rates affecting
the TenSafe API. Some users may experience timeouts or failed requests.

Our engineering team is actively investigating the issue. We will provide
updates as more information becomes available.

Posted: [timestamp]
```

#### Identified
```
Title: Issue identified - Implementing fix

We have identified the cause of the elevated error rates and are
implementing a fix. The issue is related to [high-level explanation].

We expect the fix to be deployed within [timeframe].

Posted: [timestamp]
```

#### Resolved
```
Title: Resolved - Service restored

The issue causing elevated error rates has been resolved. All systems
are now operating normally.

Duration: [X hours Y minutes]
Impact: [Brief impact summary]

We apologize for any inconvenience this may have caused. A detailed
post-incident report will be published within 48 hours.

Posted: [timestamp]
```

### Customer Email Template (P1 Only)

```
Subject: TenSafe Service Incident - [Date]

Dear [Customer Name],

We are writing to inform you of a service incident that affected
TenSafe on [date] from [start time] to [end time] UTC.

INCIDENT SUMMARY
[Brief description of what happened]

IMPACT
[Description of how customers were affected]

ROOT CAUSE
[High-level explanation of what caused the issue]

REMEDIATION
[What we've done to fix the issue]

PREVENTION
[What we're doing to prevent recurrence]

We sincerely apologize for any disruption this may have caused to
your operations. If you have any questions or concerns, please
contact your account representative or support@tensafe.io.

Sincerely,
TenSafe Operations Team
```

---

## Escalation Matrix

### Escalation Triggers

| Condition | Action |
|-----------|--------|
| No response to P1 alert in 15 min | Auto-escalate to secondary |
| No progress on P1 in 30 min | Escalate to Engineering Manager |
| P1 duration > 1 hour | Escalate to VP Engineering |
| P1 duration > 2 hours | Escalate to CTO |
| Data breach confirmed | Immediately escalate to Security + Legal |
| Regulatory impact | Escalate to Compliance + Legal |

### Escalation Contacts

| Level | Role | Response Time | Contact Method |
|-------|------|---------------|----------------|
| L1 | On-Call Engineer | 5 minutes | PagerDuty |
| L2 | Secondary On-Call | 10 minutes | PagerDuty |
| L3 | Engineering Manager | 15 minutes | PagerDuty + Phone |
| L4 | VP Engineering | 30 minutes | Phone |
| L5 | CTO | 30 minutes | Phone |
| Security | Security Team Lead | 15 minutes | PagerDuty |
| Legal | General Counsel | 1 hour | Phone |

### How to Escalate

```bash
# PagerDuty CLI
pd incident:escalate --incident-id <id> --level L3

# Or use PagerDuty UI:
# 1. Open incident
# 2. Click "Escalate"
# 3. Select escalation policy level
```

### Escalation Communication

When escalating, include:
- Current severity
- Duration so far
- Impact summary
- Actions taken
- Specific help needed

---

## Post-Mortem Template

### Post-Mortem Document

```markdown
# Post-Mortem: [Incident Title]

**Date**: [Date of incident]
**Duration**: [Start time] - [End time] ([Total duration])
**Severity**: P[1/2/3/4]
**Authors**: [Names]
**Status**: Draft / Final

---

## Executive Summary

[2-3 sentence summary of what happened, impact, and resolution]

---

## Timeline (All times in UTC)

| Time | Event |
|------|-------|
| HH:MM | [What happened] |
| HH:MM | [Alert fired] |
| HH:MM | [On-call acknowledged] |
| HH:MM | [Incident declared] |
| HH:MM | [Root cause identified] |
| HH:MM | [Mitigation applied] |
| HH:MM | [Service restored] |
| HH:MM | [Incident closed] |

---

## Impact

### Customer Impact
- [X] customers affected
- [Y] requests failed
- [Z] minutes of downtime
- [Revenue impact if applicable]

### Internal Impact
- [Team impact]
- [Process impact]

### Privacy/Compliance Impact
- [Any privacy implications]
- [Any compliance implications]

---

## Root Cause Analysis

### What happened
[Detailed technical explanation]

### Why it happened
[5 Whys or similar analysis]

1. Why? [First level]
2. Why? [Second level]
3. Why? [Third level]
4. Why? [Fourth level]
5. Why? [Root cause]

### Contributing factors
- [Factor 1]
- [Factor 2]
- [Factor 3]

---

## What Went Well

- [Thing 1]
- [Thing 2]
- [Thing 3]

---

## What Could Be Improved

- [Improvement 1]
- [Improvement 2]
- [Improvement 3]

---

## Action Items

| Priority | Action | Owner | Due Date | Status |
|----------|--------|-------|----------|--------|
| P1 | [Immediate fix] | @name | [date] | Done |
| P2 | [Short-term improvement] | @name | [date] | In Progress |
| P3 | [Long-term prevention] | @name | [date] | Planned |

---

## Lessons Learned

[Key takeaways for the broader team]

---

## Appendix

### Relevant Links
- [Incident Slack channel]
- [Grafana dashboard during incident]
- [Relevant logs]
- [Related tickets]

### Supporting Data
[Charts, graphs, or additional data]
```

### Post-Mortem Meeting Agenda

1. **Review timeline** (10 min)
2. **Impact assessment** (5 min)
3. **Root cause discussion** (15 min)
4. **What went well** (5 min)
5. **What could be improved** (10 min)
6. **Action item assignment** (10 min)
7. **Questions and wrap-up** (5 min)

---

## Common Incident Playbooks

### Playbook: API Completely Down

1. **Verify outage**
   ```bash
   curl -v https://api.tensafe.dev/health
   kubectl get pods -n tensafe
   ```

2. **Check for recent deployments**
   ```bash
   kubectl rollout history deployment/tensafe-server -n tensafe
   ```

3. **If recent deployment - rollback**
   ```bash
   kubectl rollout undo deployment/tensafe-server -n tensafe
   ```

4. **If not deployment - check infrastructure**
   ```bash
   kubectl get nodes
   kubectl describe pods -n tensafe
   ```

5. **Check database connectivity**
   ```bash
   kubectl exec -it deployment/tensafe-server -n tensafe -- \
     curl -s http://localhost:8000/health/db
   ```

### Playbook: High Error Rate

1. **Identify error patterns**
   ```bash
   kubectl logs -l app.kubernetes.io/name=tensafe -n tensafe --tail=500 | \
     grep -i "error\|exception" | sort | uniq -c | sort -rn | head -20
   ```

2. **Check if specific endpoint**
   - Review Grafana dashboard by endpoint

3. **Check if specific pod**
   ```bash
   kubectl top pods -n tensafe
   ```

4. **Restart problematic pods** (if identified)
   ```bash
   kubectl delete pod <pod-name> -n tensafe
   ```

### Playbook: Database Connection Issues

1. **Check database status**
   ```bash
   kubectl get pods -n tensafe -l app=postgresql
   ```

2. **Check connection count**
   ```bash
   kubectl exec -it tensafe-postgresql-0 -n tensafe -- \
     psql -U tensafe -c "SELECT count(*) FROM pg_stat_activity;"
   ```

3. **If connection exhaustion - scale API pods down temporarily**
   ```bash
   kubectl scale deployment tensafe-server -n tensafe --replicas=2
   ```

4. **Check for long-running queries**
   ```bash
   kubectl exec -it tensafe-postgresql-0 -n tensafe -- \
     psql -U tensafe -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE state != 'idle' ORDER BY duration DESC LIMIT 10;"
   ```

---

## Related Documentation

- [SECURITY_INCIDENTS.md](SECURITY_INCIDENTS.md) - Security-specific incidents
- [MONITORING_ALERTS.md](MONITORING_ALERTS.md) - Alert definitions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [DISASTER_RECOVERY.md](DISASTER_RECOVERY.md) - DR procedures

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-02-03 | Platform Ops | Initial runbook |
