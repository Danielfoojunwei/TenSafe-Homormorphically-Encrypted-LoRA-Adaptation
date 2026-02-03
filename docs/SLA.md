# TenSafe Service Level Agreement (SLA)

**Effective Date:** February 2026
**Version:** 1.0
**Last Updated:** February 3, 2026

This Service Level Agreement ("SLA") describes the service level commitments for TenSafe's privacy-preserving ML platform services.

---

## 1. Service Availability Commitment

### 1.1 Uptime Guarantees

TenSafe commits to the following monthly uptime percentages based on your subscription tier:

| Tier | Monthly Uptime | Maximum Downtime/Month | Annual Downtime |
|------|----------------|------------------------|-----------------|
| **Community (Free)** | No SLA | N/A | N/A |
| **Pro** | 99.9% | 43.8 minutes | 8.76 hours |
| **Business** | 99.95% | 21.9 minutes | 4.38 hours |
| **Enterprise** | 99.99% | 4.38 minutes | 52.6 minutes |

### 1.2 Uptime Calculation

**Monthly Uptime Percentage** is calculated as:

```
Uptime % = ((Total Minutes - Downtime Minutes) / Total Minutes) × 100
```

**Downtime** is defined as a period where:
- The TenSafe API returns HTTP 5xx errors for more than 50% of requests over 5 minutes
- Core services (training, inference, encryption) are unavailable
- Users cannot authenticate or access the platform

**Uptime** is measured at our external monitoring points and excludes Scheduled Maintenance periods.

---

## 2. Service Level Objectives (SLOs)

### 2.1 API Response Time

| Endpoint Category | P50 Latency | P95 Latency | P99 Latency |
|-------------------|-------------|-------------|-------------|
| Health Checks | < 50ms | < 100ms | < 200ms |
| Authentication | < 100ms | < 250ms | < 500ms |
| Training Job Submit | < 200ms | < 500ms | < 1s |
| Inference (HE-LoRA) | < 500ms | < 1s | < 2s |
| Model Downloads | N/A (throughput based) | N/A | N/A |

### 2.2 Throughput Objectives

| Service | Minimum Throughput |
|---------|-------------------|
| API Gateway | 10,000 requests/second |
| Training Jobs | 1,000 concurrent jobs |
| Inference | 5,000 requests/second |
| Batch Processing | 100 TB/day |

### 2.3 Data Durability

| Data Type | Durability |
|-----------|------------|
| Model Artifacts | 99.999999999% (11 nines) |
| Training Data | 99.999999999% (11 nines) |
| Audit Logs | 99.999999999% (11 nines) |
| Encryption Keys | 99.999999999% (11 nines) |

---

## 3. Scheduled Maintenance

### 3.1 Maintenance Windows

TenSafe performs scheduled maintenance during the following windows:

| Region | Primary Window (UTC) | Backup Window (UTC) |
|--------|---------------------|---------------------|
| US East | Tuesday 06:00-10:00 | Thursday 06:00-10:00 |
| US West | Tuesday 10:00-14:00 | Thursday 10:00-14:00 |
| EU | Tuesday 02:00-06:00 | Thursday 02:00-06:00 |
| Asia Pacific | Monday 18:00-22:00 | Wednesday 18:00-22:00 |

### 3.2 Maintenance Notification

| Maintenance Type | Minimum Notice |
|-----------------|----------------|
| Standard Maintenance | 7 days |
| Security Patches (Non-Critical) | 72 hours |
| Security Patches (Critical) | 24 hours |
| Emergency Maintenance | 1 hour (when possible) |

### 3.3 Maintenance Exclusion

Scheduled maintenance periods are **excluded** from downtime calculations for SLA purposes when:
- Advance notice was provided per the schedule above
- Maintenance occurred within the published maintenance window
- Total monthly maintenance does not exceed 4 hours

---

## 4. Support Response Times

### 4.1 Response Time by Severity and Tier

| Severity | Community | Pro | Business | Enterprise |
|----------|-----------|-----|----------|------------|
| P1 - Critical | Best Effort | 4 hours | 1 hour | 15 minutes |
| P2 - Major | Best Effort | 8 hours | 4 hours | 1 hour |
| P3 - Minor | Best Effort | 24 hours | 8 hours | 4 hours |
| P4 - Low | Best Effort | 72 hours | 24 hours | 8 hours |

### 4.2 Severity Definitions

| Severity | Definition | Examples |
|----------|------------|----------|
| **P1 - Critical** | Complete service outage or data loss risk | All training jobs failing, API completely unavailable, security breach |
| **P2 - Major** | Major functionality impaired | Inference latency >10x normal, 50%+ job failures, authentication issues |
| **P3 - Minor** | Minor functionality impaired | Single component degraded, non-critical feature unavailable |
| **P4 - Low** | Questions or cosmetic issues | Documentation questions, UI improvements, feature requests |

### 4.3 Resolution Time Targets

| Severity | Target Resolution Time |
|----------|----------------------|
| P1 - Critical | 4 hours |
| P2 - Major | 8 hours |
| P3 - Minor | 72 hours |
| P4 - Low | 30 days |

---

## 5. Service Credits

### 5.1 Credit Eligibility

If TenSafe fails to meet the Monthly Uptime Percentage commitment, eligible customers may request service credits.

### 5.2 Credit Amounts

| Monthly Uptime | Pro Credit | Business Credit | Enterprise Credit |
|----------------|------------|-----------------|-------------------|
| 99.0% - 99.9% | 10% | 10% | 25% |
| 95.0% - 99.0% | 25% | 25% | 50% |
| 90.0% - 95.0% | 50% | 50% | 75% |
| < 90.0% | 100% | 100% | 100% |

Credits are calculated as a percentage of the monthly subscription fee for affected services.

### 5.3 Credit Request Process

1. Submit a credit request via support within 30 days of the incident
2. Include:
   - Account identifier
   - Affected service(s)
   - Date and time of the incident
   - Description of impact
3. TenSafe will validate the claim and apply credits within 60 days

### 5.4 Credit Limitations

- Maximum credit per month: 100% of monthly fee
- Credits are not transferable
- Credits expire after 12 months
- Credits cannot be exchanged for cash

---

## 6. SLA Exclusions

The SLA does not apply to:

### 6.1 Customer-Caused Issues

- Misuse or misconfiguration of TenSafe services
- Actions that violate the Terms of Service
- Custom code or integrations not provided by TenSafe

### 6.2 External Factors

- Force majeure events (natural disasters, war, terrorism)
- Internet connectivity issues outside TenSafe's network
- DNS issues outside TenSafe's control
- Third-party service failures (cloud providers, CDNs)

### 6.3 Specific Scenarios

- Preview, beta, or experimental features
- Free tier services
- Scheduled maintenance (as defined in Section 3)
- Customer-initiated shutdowns or scaling events
- Rate limiting due to abuse prevention
- Regional outages in non-contracted regions

### 6.4 Data-Related

- Data corruption caused by customer-provided data
- Loss of data due to customer deletion
- Encryption key loss due to customer actions

---

## 7. Monitoring and Reporting

### 7.1 Status Page

Real-time service status is available at: **https://status.tensafe.io**

The status page provides:
- Current system status
- Component-level status
- Active incidents and updates
- Scheduled maintenance calendar
- Historical uptime data (90 days)

### 7.2 Uptime Reports

| Tier | Report Frequency | Report Contents |
|------|------------------|-----------------|
| Pro | Monthly | Uptime %, incidents, maintenance |
| Business | Weekly | Uptime %, incidents, maintenance, latency metrics |
| Enterprise | Daily | Full metrics, custom dashboards, SLA tracking |

### 7.3 Notification Channels

Subscribe to status updates via:
- Email notifications
- RSS feed
- Slack integration (Business+)
- PagerDuty integration (Enterprise)
- Custom webhooks (Enterprise)

---

## 8. Data Protection Commitments

### 8.1 Encryption Standards

| Data State | Encryption | Standard |
|------------|------------|----------|
| At Rest | AES-256-GCM | FIPS 140-2 |
| In Transit | TLS 1.3 | NIST SP 800-52 |
| HE-LoRA Operations | CKKS/BFV | 128-bit security |
| Key Management | HSM-backed | FIPS 140-2 Level 3 |

### 8.2 Compliance Certifications

TenSafe maintains the following certifications:
- SOC 2 Type II
- ISO 27001
- ISO 27701
- HIPAA (with BAA)
- GDPR compliant

### 8.3 Data Residency

Enterprise customers may specify data residency requirements:
- US (us-east-1, us-west-2)
- EU (eu-west-1, eu-central-1)
- Asia Pacific (ap-northeast-1, ap-southeast-1)

---

## 9. Disaster Recovery

### 9.1 Recovery Objectives

| Metric | Standard | Enterprise |
|--------|----------|------------|
| Recovery Time Objective (RTO) | 4 hours | 1 hour |
| Recovery Point Objective (RPO) | 1 hour | 15 minutes |

### 9.2 Backup Schedule

| Data Type | Backup Frequency | Retention |
|-----------|-----------------|-----------|
| Databases | Continuous + Daily | 30 days |
| Model Artifacts | Real-time replication | 90 days |
| Audit Logs | Continuous | 7 years |
| Configuration | Real-time | 90 days |

### 9.3 Geographic Redundancy

- Primary: Multi-AZ deployment
- Secondary: Cross-region replication (Enterprise)
- Tertiary: Cold storage archive

---

## 10. Changes to This SLA

TenSafe reserves the right to modify this SLA. Changes will be:
- Posted to the TenSafe website
- Communicated via email to affected customers
- Effective 30 days after posting (or longer for material changes)

Changes that materially reduce service commitments will not apply retroactively.

---

## 11. Contact Information

**Support Portal:** https://support.tensafe.io
**Status Page:** https://status.tensafe.io
**Security Issues:** security@tensafe.io
**SLA Claims:** sla-claims@tensafe.io

---

## Appendix A: Definitions

| Term | Definition |
|------|------------|
| **Downtime** | Period when service is unavailable as defined in Section 1.2 |
| **Monthly Uptime Percentage** | Calculated as defined in Section 1.2 |
| **Scheduled Maintenance** | Planned maintenance with advance notice per Section 3 |
| **Service Credit** | Credit applied to account as compensation for SLA breach |
| **P50/P95/P99** | 50th, 95th, and 99th percentile latency measurements |
| **RTO** | Recovery Time Objective - maximum acceptable downtime after disaster |
| **RPO** | Recovery Point Objective - maximum acceptable data loss window |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | February 2026 | Initial SLA release |

---

*This SLA is part of the TenSafe Terms of Service. By using TenSafe services, you agree to be bound by this SLA.*
