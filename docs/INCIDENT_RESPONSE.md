# TenSafe Incident Response Procedures

**Document Classification:** Internal Operations
**Version:** 1.0
**Last Updated:** February 3, 2026
**Owner:** Site Reliability Engineering (SRE)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Severity Levels](#2-severity-levels)
3. [Incident Roles](#3-incident-roles)
4. [Detection and Triage](#4-detection-and-triage)
5. [Escalation Procedures](#5-escalation-procedures)
6. [Communication Protocols](#6-communication-protocols)
7. [Resolution Process](#7-resolution-process)
8. [Post-Incident Activities](#8-post-incident-activities)
9. [Tools and Resources](#9-tools-and-resources)

---

## 1. Overview

### 1.1 Purpose

This document establishes TenSafe's incident response procedures to ensure:
- Rapid detection and response to service disruptions
- Clear communication with stakeholders
- Minimal customer impact
- Continuous improvement through post-incident learning

### 1.2 Scope

These procedures apply to all incidents affecting:
- TenSafe production services
- Customer-facing APIs and applications
- Infrastructure and platform components
- Security events
- Data integrity issues

### 1.3 Definitions

| Term | Definition |
|------|------------|
| **Incident** | Unplanned service degradation or outage |
| **Incident Commander (IC)** | Person coordinating incident response |
| **Time to Detection (TTD)** | Time from incident start to detection |
| **Time to Mitigation (TTM)** | Time from detection to mitigation |
| **Time to Resolution (TTR)** | Time from detection to full resolution |
| **MTTD/MTTM/MTTR** | Mean (average) of above metrics |

---

## 2. Severity Levels

### 2.1 Severity Definitions

#### P1 - Critical
**Definition:** Complete service outage or critical security breach

**Criteria (any of):**
- All users unable to access core services
- Complete API unavailability
- Active security breach with data exposure risk
- Data loss or corruption affecting multiple customers
- Payment processing failure

**Response Requirements:**
- All-hands response
- Executive notification
- Status page update within 10 minutes
- Customer communication within 15 minutes
- Continuous updates every 30 minutes

**Target Resolution:** 2 hours

---

#### P2 - Major
**Definition:** Major functionality impaired affecting many users

**Criteria (any of):**
- Core feature unavailable (training, inference, encryption)
- Performance degradation >50% of baseline
- >25% of requests failing
- Single region outage
- Authentication service degraded

**Response Requirements:**
- On-call engineer + backup
- Manager notification
- Status page update within 15 minutes
- Customer communication within 30 minutes
- Updates every hour

**Target Resolution:** 4 hours

---

#### P3 - Minor
**Definition:** Minor functionality impaired with workaround available

**Criteria (any of):**
- Non-critical feature unavailable
- Performance degradation 10-50%
- Intermittent errors (<10% of requests)
- Single customer impact
- Monitoring/alerting issues

**Response Requirements:**
- On-call engineer
- Status page update if customer-visible
- Updates every 4 hours

**Target Resolution:** 24 hours

---

#### P4 - Low
**Definition:** Minor issue with minimal impact

**Criteria (any of):**
- Cosmetic issues
- Documentation errors
- Non-urgent maintenance
- Feature requests from incidents

**Response Requirements:**
- Assigned to engineering queue
- No immediate action required

**Target Resolution:** 30 days

---

### 2.2 Severity Selection Flowchart

```
Start
  |
  v
Is service completely unavailable for all users?
  |
  +--> Yes --> P1 CRITICAL
  |
  v
Is a core feature (training/inference/encryption) unavailable?
  |
  +--> Yes --> P2 MAJOR
  |
  v
Is there a security incident with potential data exposure?
  |
  +--> Yes --> P1 CRITICAL
  |
  v
Are >25% of requests failing or latency >2x baseline?
  |
  +--> Yes --> P2 MAJOR
  |
  v
Is there a workaround available?
  |
  +--> No --> P2 MAJOR
  |
  +--> Yes, affecting multiple customers --> P3 MINOR
  |
  +--> Yes, single customer or internal --> P4 LOW
  |
  v
End
```

---

## 3. Incident Roles

### 3.1 Incident Commander (IC)

**Responsibilities:**
- Overall coordination of incident response
- Decision-making authority during incident
- Resource allocation
- Communication coordination
- Escalation decisions
- Declaring incident resolved

**Selection:**
- P1: Engineering Manager or above
- P2: Senior Engineer or above
- P3/P4: On-call engineer

### 3.2 Technical Lead

**Responsibilities:**
- Technical investigation and diagnosis
- Coordinating engineering response
- Implementing fixes
- Advising IC on technical decisions

**Selection:**
- Subject matter expert for affected system
- Senior engineer from owning team

### 3.3 Communications Lead

**Responsibilities:**
- Customer communications
- Status page updates
- Internal stakeholder updates
- Executive briefings

**Selection:**
- P1: Customer Success Manager or Support Lead
- P2/P3: On-call engineer or support

### 3.4 Scribe

**Responsibilities:**
- Documenting timeline
- Recording decisions
- Capturing actions taken
- Creating post-incident notes

**Selection:**
- Any available team member
- Not the IC or Technical Lead

### 3.5 Subject Matter Experts (SMEs)

**Responsibilities:**
- Providing domain expertise
- Assisting with investigation
- Implementing fixes in their area

**Selection:**
- As needed based on affected systems

---

## 4. Detection and Triage

### 4.1 Detection Sources

| Source | Description | Alert Destination |
|--------|-------------|-------------------|
| Automated Monitoring | Prometheus/Grafana alerts | PagerDuty |
| Health Checks | Status service checks | PagerDuty |
| Customer Reports | Support tickets, emails | Support queue |
| Internal Reports | Slack #incidents channel | On-call |
| Security Tools | SIEM, WAF alerts | Security team |

### 4.2 Initial Triage (First 5 Minutes)

1. **Acknowledge alert** in PagerDuty
2. **Verify the issue** - Is this a real incident?
3. **Assess severity** using criteria above
4. **Declare incident** if P1/P2:
   ```
   /incident declare "Brief description" severity=P1
   ```
5. **Begin investigation** while assembling team

### 4.3 Triage Checklist

```markdown
[ ] Alert acknowledged
[ ] Issue verified (not false positive)
[ ] Severity assessed
[ ] Incident declared (if P1/P2)
[ ] IC identified
[ ] Communication channel established
[ ] Initial status page update (if needed)
```

---

## 5. Escalation Procedures

### 5.1 Automatic Escalations

| Condition | Escalation |
|-----------|------------|
| P1 not acknowledged in 5 min | Backup on-call |
| P1 not acknowledged in 10 min | Engineering Manager |
| P2 not acknowledged in 15 min | Backup on-call |
| P1 open > 30 min | VP Engineering |
| P1 open > 1 hour | CTO |
| Any security incident | Security Lead immediately |

### 5.2 Manual Escalation Criteria

Escalate to management when:
- Additional resources needed
- Decision requires authority beyond IC
- Customer communication needed beyond standard
- Incident scope expanding
- Resolution taking longer than target

### 5.3 Escalation Contacts

| Role | Contact Method | Response Time |
|------|---------------|---------------|
| Backup On-call | PagerDuty | 5 minutes |
| Engineering Manager | Slack + Phone | 10 minutes |
| VP Engineering | Phone | 15 minutes |
| CTO | Phone | 15 minutes |
| Security Lead | PagerDuty | 5 minutes |
| CEO (P1 only) | Phone | 30 minutes |

### 5.4 Escalation Template

```
ESCALATION REQUEST

Incident: [Incident ID]
Severity: [P1/P2]
Duration: [Time since detection]
Current Status: [Brief status]

Reason for Escalation:
[Why escalation is needed]

Requested Action:
[What you need from escalation target]

Current Team:
- IC: [Name]
- Tech Lead: [Name]
- Others: [Names]

Bridge: [Link to call/channel]
```

---

## 6. Communication Protocols

### 6.1 Internal Communication

#### Incident Channel
- Create dedicated Slack channel: `#inc-YYYYMMDD-brief-desc`
- Pin: incident document, status page link, bridge link
- Use for all incident-related discussion

#### Bridge Call
- P1: Always establish video bridge
- P2: Voice bridge recommended
- Link: https://meet.tensafe.io/incident

#### Status Updates (Internal)
```
[TIME] STATUS UPDATE

Current State: [Investigating/Identified/Monitoring/Resolved]

Summary:
[2-3 sentence summary]

Actions Taken:
- [Action 1]
- [Action 2]

Next Steps:
- [Next action] (Owner: [Name], ETA: [Time])

Customer Impact:
[Description of customer impact]

Next Update: [Time]
```

### 6.2 External Communication

#### Status Page Updates

**Template - Investigating:**
```
[Service Name] - Investigating Issues

We are currently investigating issues with [service].
Users may experience [symptoms].

We will provide an update within [time].

Posted: [timestamp]
```

**Template - Identified:**
```
[Service Name] - Issue Identified

We have identified the cause of [issue].
Our team is implementing a fix.

Impact: [description]
Workaround: [if available]

Next update in [time].

Posted: [timestamp]
```

**Template - Monitoring:**
```
[Service Name] - Fix Implemented

A fix has been implemented and we are monitoring the results.
[Service] should be returning to normal operation.

Users who continue to experience issues should [action].

Posted: [timestamp]
```

**Template - Resolved:**
```
[Service Name] - Resolved

The issue affecting [service] has been resolved.

Duration: [start time] to [end time] ([total duration])
Impact: [summary of impact]

A post-incident review will be conducted.

Posted: [timestamp]
```

#### Customer Email (P1 Only)

```
Subject: TenSafe Service Incident - [Brief Description]

Dear [Customer],

We are writing to inform you of a service incident affecting TenSafe.

INCIDENT SUMMARY
- Start Time: [UTC timestamp]
- Services Affected: [list]
- Current Status: [status]

IMPACT
[Description of customer impact]

WHAT WE'RE DOING
[Actions being taken]

NEXT STEPS
- [If action required from customer]
- We will provide updates at [frequency]

STATUS PAGE
https://status.tensafe.io

We apologize for any inconvenience. Our team is working to resolve
this as quickly as possible.

TenSafe Operations Team
```

### 6.3 Communication Frequency

| Severity | Status Page | Internal Updates | Customer Email |
|----------|-------------|------------------|----------------|
| P1 | Every 30 min | Every 15 min | Hourly |
| P2 | Every hour | Every 30 min | End of incident |
| P3 | Initial + resolved | Every 4 hours | N/A |
| P4 | N/A | As needed | N/A |

---

## 7. Resolution Process

### 7.1 Investigation Approach

1. **Gather data:**
   - Metrics dashboards
   - Logs (structured query)
   - Recent deployments
   - Recent changes

2. **Form hypothesis:**
   - What changed?
   - What's the correlation?
   - What's the causation?

3. **Test hypothesis:**
   - Safe, reversible tests
   - A/B comparison
   - Canary rollback

4. **Implement fix:**
   - Mitigation first (stop bleeding)
   - Root cause fix later

### 7.2 Common Mitigation Actions

| Symptom | Quick Mitigations |
|---------|------------------|
| High error rate | Rollback, traffic shift, circuit breaker |
| High latency | Scale up, traffic shed, cache |
| Resource exhaustion | Restart, scale, rate limit |
| Dependency failure | Failover, circuit breaker, cache |
| Security incident | Isolate, revoke, block |

### 7.3 Rollback Procedure

1. Identify deployment to rollback
2. Notify IC of rollback plan
3. Execute rollback:
   ```bash
   kubectl rollout undo deployment/[name] -n [namespace]
   ```
4. Monitor for improvement
5. Declare mitigation if successful

### 7.4 Resolution Criteria

Incident can be resolved when:
- [ ] Root cause addressed (or mitigated with tracking ticket)
- [ ] Error rates returned to baseline
- [ ] Latency returned to baseline
- [ ] No ongoing customer impact
- [ ] Monitoring confirms stability (15+ minutes)
- [ ] IC declares resolved

---

## 8. Post-Incident Activities

### 8.1 Post-Incident Review (PIR) Requirements

| Severity | PIR Required | Timeline | Attendees |
|----------|--------------|----------|-----------|
| P1 | Yes | 72 hours | All involved + management |
| P2 | Yes | 1 week | All involved |
| P3 | Optional | 2 weeks | Owning team |
| P4 | No | N/A | N/A |

### 8.2 PIR Document Template

```markdown
# Post-Incident Review: [Incident Title]

**Incident ID:** [ID]
**Date:** [Date]
**Duration:** [Start] to [End] ([Total])
**Severity:** [P1/P2/P3]
**Author:** [Name]
**Review Date:** [Date]

## Summary
[2-3 paragraph executive summary]

## Impact
- **Customers Affected:** [Number/percentage]
- **Requests Failed:** [Number]
- **Revenue Impact:** [If applicable]
- **SLA Impact:** [Minutes of downtime]

## Timeline
| Time (UTC) | Event |
|------------|-------|
| [Time] | [Event description] |
| [Time] | [Event description] |

## Root Cause Analysis

### What Happened
[Detailed technical explanation]

### Why It Happened
[5 Whys or other analysis]

### Contributing Factors
- [Factor 1]
- [Factor 2]

## What Went Well
- [Positive item]
- [Positive item]

## What Could Be Improved
- [Improvement area]
- [Improvement area]

## Action Items
| Action | Owner | Priority | Due Date | Status |
|--------|-------|----------|----------|--------|
| [Action] | [Name] | P1/P2/P3 | [Date] | Open |

## Lessons Learned
[Key takeaways for the organization]

## Appendix
- [Link to logs]
- [Link to metrics]
- [Link to related documents]
```

### 8.3 PIR Meeting Agenda

1. **Introduction** (5 min)
   - Set blameless tone
   - Review meeting goals

2. **Timeline Review** (15 min)
   - Walk through events
   - Fill in gaps

3. **Root Cause Discussion** (20 min)
   - Technical deep dive
   - Contributing factors

4. **Response Evaluation** (10 min)
   - What went well
   - What could improve

5. **Action Item Creation** (15 min)
   - Identify improvements
   - Assign owners

6. **Wrap-up** (5 min)
   - Summarize actions
   - Set follow-up

### 8.4 Action Item Tracking

- All action items tracked in Jira with `incident-action` label
- P1 actions reviewed weekly by Engineering Manager
- P2 actions reviewed bi-weekly
- Metrics tracked: action completion rate, time to complete

---

## 9. Tools and Resources

### 9.1 Incident Management Tools

| Tool | Purpose | URL/Access |
|------|---------|------------|
| PagerDuty | Alerting, on-call | pagerduty.com/tensafe |
| Slack | Communication | #incidents, #inc-* channels |
| status.tensafe.io | Status page | status.tensafe.io/admin |
| Grafana | Metrics | grafana.internal.tensafe.io |
| Kibana | Logs | kibana.internal.tensafe.io |
| Jira | Action tracking | jira.tensafe.io |

### 9.2 Runbooks

| Service | Runbook Location |
|---------|-----------------|
| API Gateway | docs/runbooks/api-gateway.md |
| Training Service | docs/runbooks/training.md |
| Inference Service | docs/runbooks/inference.md |
| Database | docs/runbooks/database.md |
| Authentication | docs/runbooks/auth.md |
| Encryption Service | docs/runbooks/encryption.md |

### 9.3 Quick Reference

#### Declare Incident
```
/incident declare "description" severity=P1
```

#### Update Status Page
```
/status update "message" state=investigating
```

#### Page Additional Responder
```
/pd page @username "reason"
```

#### Rollback Deployment
```
kubectl rollout undo deployment/[name] -n production
```

### 9.4 On-Call Information

| Team | Schedule | Escalation |
|------|----------|------------|
| Platform | Weekly rotation | Engineering Manager |
| Security | Weekly rotation | Security Lead |
| Support | Follow-the-sun | Support Manager |

### 9.5 Emergency Contacts

| Role | Name | Phone |
|------|------|-------|
| VP Engineering | [Name] | [Number] |
| CTO | [Name] | [Number] |
| Security Lead | [Name] | [Number] |
| AWS TAM | [Name] | [Number] |
| GCP TAM | [Name] | [Number] |

---

## Appendix A: Incident Command Cheat Sheet

```
INCIDENT COMMANDER CHECKLIST

FIRST 5 MINUTES
[ ] Acknowledge alert
[ ] Verify incident
[ ] Assess severity
[ ] Declare incident
[ ] Establish communication channel

FIRST 15 MINUTES
[ ] Assign roles (Tech Lead, Comms, Scribe)
[ ] Update status page
[ ] Begin investigation
[ ] Escalate if needed

ONGOING
[ ] Maintain communication cadence
[ ] Track timeline
[ ] Make go/no-go decisions
[ ] Manage resources

RESOLUTION
[ ] Verify stability (15+ minutes)
[ ] Update status page - resolved
[ ] Send internal summary
[ ] Schedule PIR (if needed)
[ ] Close incident channel after 24h
```

---

## Appendix B: Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Feb 2026 | SRE Team | Initial release |

---

*This document is reviewed and updated quarterly by the SRE team.*
