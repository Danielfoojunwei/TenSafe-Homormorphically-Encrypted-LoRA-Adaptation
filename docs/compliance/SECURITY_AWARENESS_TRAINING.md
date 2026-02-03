# Security Awareness Training Program

**Version**: 1.0
**Last Updated**: 2026-02-03
**Document Owner**: Security Team
**Review Frequency**: Annual

---

## Overview

This document outlines the security awareness training program for TenSafe/TensorGuard platform personnel. The program ensures all team members understand their security responsibilities and can identify and respond to security threats.

### Compliance Requirements

| Framework | Requirement | Training Topic |
|-----------|-------------|----------------|
| SOC 2 CC1.4 | Security awareness | All modules |
| ISO 27001 A.6.3 | Information security awareness | All modules |
| ISO 27701 6.4.2.2 | Privacy awareness | Module 5 |
| HIPAA §164.308(a)(5) | Security awareness training | Modules 1-4, 7 |

---

## Training Modules

### Module 1: Security Fundamentals

**Duration**: 45 minutes
**Audience**: All personnel
**Frequency**: Upon hire, annual refresher

#### Learning Objectives

After completing this module, participants will be able to:

1. Explain the importance of information security
2. Identify common security threats
3. Describe their role in maintaining security
4. Report security incidents appropriately

#### Topics Covered

**1.1 Introduction to Information Security**
- CIA Triad: Confidentiality, Integrity, Availability
- Defense in depth principle
- Shared responsibility model

**1.2 Threat Landscape**
- Common attack vectors
- Targeted vs. opportunistic attacks
- Insider threats

**1.3 Security Policies**
- Acceptable use policy
- Data classification
- Incident reporting procedures

**1.4 Your Role in Security**
- Personal responsibility
- Security culture
- Reporting mechanisms

#### Assessment

- 10-question multiple choice quiz
- Passing score: 80%
- Unlimited attempts with 24-hour cooldown

---

### Module 2: Password and Authentication Security

**Duration**: 30 minutes
**Audience**: All personnel
**Frequency**: Upon hire, annual refresher

#### Learning Objectives

1. Create strong, unique passwords
2. Use multi-factor authentication effectively
3. Recognize and avoid credential theft attempts
4. Manage credentials securely

#### Topics Covered

**2.1 Password Best Practices**
```
Strong Password Requirements:
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, special characters
- No personal information
- Unique for each account
- Not based on dictionary words
```

**2.2 Multi-Factor Authentication (MFA)**
- What MFA is and why it matters
- Types of MFA (TOTP, hardware keys, push notifications)
- Setting up and using MFA
- Protecting backup codes

**2.3 Credential Management**
- Password managers
- Single Sign-On (SSO)
- Never share credentials
- Secure credential storage

**2.4 Recognizing Credential Attacks**
- Phishing attempts
- Social engineering
- Credential stuffing
- Brute force attacks

#### Hands-On Exercise

- Set up TOTP authenticator app
- Generate and securely store backup codes
- Identify phishing emails (5 sample scenarios)

---

### Module 3: Phishing and Social Engineering

**Duration**: 45 minutes
**Audience**: All personnel
**Frequency**: Upon hire, quarterly simulations

#### Learning Objectives

1. Identify phishing emails and messages
2. Recognize social engineering tactics
3. Respond appropriately to suspected attacks
4. Report phishing attempts

#### Topics Covered

**3.1 Types of Phishing**

| Type | Description | Example |
|------|-------------|---------|
| Email Phishing | Mass emails with malicious links | "Your account suspended" |
| Spear Phishing | Targeted attacks on individuals | CEO impersonation |
| Smishing | SMS-based phishing | Fake delivery notifications |
| Vishing | Voice call phishing | IT support impersonation |

**3.2 Red Flags to Watch For**

```
Email Red Flags:
[ ] Urgent or threatening language
[ ] Unexpected attachments
[ ] Mismatched sender address
[ ] Generic greetings
[ ] Grammar/spelling errors
[ ] Suspicious links (hover to check)
[ ] Requests for credentials or money
[ ] Unexpected requests from executives
```

**3.3 Social Engineering Tactics**
- Pretexting (fake scenarios)
- Baiting (USB drops, fake downloads)
- Quid pro quo (fake IT support)
- Tailgating (physical access)

**3.4 Response Procedures**

```
If you suspect phishing:
1. DO NOT click links or open attachments
2. DO NOT reply to the message
3. Report to security@company.com
4. Forward suspicious email as attachment
5. Delete the message after reporting
```

#### Phishing Simulation Program

- Quarterly simulated phishing campaigns
- Immediate training for those who click
- Aggregate reporting (no individual shaming)
- Progressive difficulty

---

### Module 4: Data Handling and Classification

**Duration**: 30 minutes
**Audience**: All personnel
**Frequency**: Upon hire, annual refresher

#### Learning Objectives

1. Classify data according to policy
2. Handle each classification level appropriately
3. Protect sensitive data in transit and at rest
4. Dispose of data securely

#### Data Classification Levels

| Level | Description | Examples | Handling |
|-------|-------------|----------|----------|
| **Public** | No restriction | Marketing materials, blog posts | No special handling |
| **Internal** | Business use only | Org charts, procedures | Don't share externally |
| **Confidential** | Restricted access | Financial data, contracts | Encrypt, need-to-know |
| **Restricted** | Highly sensitive | PII, PHI, secrets | Encryption mandatory, audit trail |

#### Handling Guidelines

**4.1 Data in Transit**
```
Requirements by Classification:
- Public: HTTP acceptable
- Internal: HTTPS required
- Confidential: HTTPS + authenticated
- Restricted: HTTPS + E2E encryption + logging
```

**4.2 Data at Rest**
```
Storage Requirements:
- Public: Standard storage
- Internal: Access-controlled storage
- Confidential: Encrypted storage
- Restricted: Encrypted + HSM-protected keys
```

**4.3 Data Disposal**
```
Disposal Methods:
- Digital: Secure deletion (multiple overwrites)
- Physical: Cross-cut shredding
- Media: Degaussing or physical destruction
- Cloud: Verify deletion, key destruction
```

#### Practical Exercises

- Classify 10 sample documents
- Identify improper data handling scenarios
- Complete secure file transfer exercise

---

### Module 5: Privacy and Data Protection

**Duration**: 45 minutes
**Audience**: All personnel (extended for data handlers)
**Frequency**: Upon hire, annual refresher

#### Learning Objectives

1. Understand key privacy principles
2. Recognize personal data and special categories
3. Handle data subject requests
4. Comply with privacy regulations

#### Topics Covered

**5.1 Privacy Principles**

| Principle | Description |
|-----------|-------------|
| Lawfulness | Process data only with valid legal basis |
| Purpose Limitation | Use data only for specified purposes |
| Data Minimization | Collect only what's necessary |
| Accuracy | Keep data accurate and up-to-date |
| Storage Limitation | Don't keep data longer than needed |
| Security | Protect data with appropriate measures |
| Accountability | Demonstrate compliance |

**5.2 Personal Data Categories**

```
Standard Personal Data:
- Name, address, email, phone
- IP address, device identifiers
- Employment information
- Financial information

Special Category Data (extra protection):
- Health information
- Biometric data
- Racial/ethnic origin
- Political opinions
- Religious beliefs
- Sexual orientation
- Trade union membership
```

**5.3 Data Subject Rights**

| Right | Description | Response Time |
|-------|-------------|---------------|
| Access | Get copy of their data | 30 days |
| Rectification | Correct inaccurate data | 30 days |
| Erasure | Delete their data | 30 days |
| Restriction | Limit processing | 72 hours |
| Portability | Get data in portable format | 30 days |
| Objection | Object to processing | 30 days |

**5.4 Handling Privacy Requests**

```
When receiving a data subject request:
1. Verify the requester's identity
2. Log the request immediately
3. Route to privacy@company.com
4. Don't process without verification
5. Meet response deadlines
```

---

### Module 6: Secure Development Practices

**Duration**: 60 minutes
**Audience**: Developers, DevOps, QA
**Frequency**: Upon hire, annual refresher

#### Learning Objectives

1. Apply secure coding principles
2. Identify common vulnerabilities
3. Use security tools effectively
4. Follow secure development lifecycle

#### OWASP Top 10 (2021)

| Rank | Vulnerability | Prevention |
|------|--------------|------------|
| A01 | Broken Access Control | RBAC, least privilege, deny by default |
| A02 | Cryptographic Failures | Use strong algorithms, proper key management |
| A03 | Injection | Parameterized queries, input validation |
| A04 | Insecure Design | Threat modeling, secure design patterns |
| A05 | Security Misconfiguration | Hardening, remove defaults |
| A06 | Vulnerable Components | Dependency scanning, updates |
| A07 | Auth Failures | MFA, strong passwords, rate limiting |
| A08 | Software/Data Integrity | Code signing, integrity checks |
| A09 | Logging Failures | Comprehensive logging, monitoring |
| A10 | SSRF | Input validation, allowlists |

#### Secure Coding Guidelines

**6.1 Input Validation**
```python
# BAD - SQL Injection vulnerable
query = f"SELECT * FROM users WHERE id = {user_input}"

# GOOD - Parameterized query
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_input,))
```

**6.2 Authentication**
```python
# Use strong password hashing
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["argon2"])
hashed = pwd_context.hash(password)
```

**6.3 Secrets Management**
```python
# BAD - Hardcoded secrets
API_KEY = "sk-1234567890abcdef"

# GOOD - Environment variables or secrets manager
API_KEY = os.environ.get("API_KEY")
```

**6.4 Error Handling**
```python
# BAD - Exposes internal details
except Exception as e:
    return {"error": str(e)}

# GOOD - Generic error message
except Exception as e:
    logger.error(f"Internal error: {e}")
    return {"error": "An internal error occurred"}
```

#### Security Tools

| Tool | Purpose | When to Use |
|------|---------|-------------|
| SAST | Static code analysis | Every commit (CI) |
| DAST | Dynamic testing | Before deployment |
| Dependency Scanner | Vulnerable libraries | Every build |
| Secret Scanner | Exposed credentials | Pre-commit hook |
| Container Scanner | Image vulnerabilities | Before deployment |

---

### Module 7: Incident Response

**Duration**: 30 minutes
**Audience**: All personnel (extended for responders)
**Frequency**: Upon hire, annual refresher, post-incident

#### Learning Objectives

1. Recognize security incidents
2. Report incidents promptly
3. Preserve evidence
4. Follow response procedures

#### What Constitutes an Incident?

```
Report immediately if you observe:
- Unauthorized access attempts
- Malware or ransomware
- Data breach or exposure
- Lost/stolen devices
- Suspicious emails (phishing)
- Social engineering attempts
- System compromises
- Policy violations
```

#### Incident Response Steps

```
1. IDENTIFY
   - Recognize the incident
   - Assess initial severity

2. CONTAIN
   - Isolate affected systems
   - Preserve evidence
   - Don't destroy logs

3. REPORT
   - Contact: security@company.com
   - Call: [Security Hotline]
   - Slack: #security-incidents

4. COOPERATE
   - Provide information to responders
   - Follow instructions
   - Document your observations
```

#### Incident Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| P1 Critical | Active breach, data exfil | Immediate | Ransomware attack |
| P2 High | Significant threat | 1 hour | Compromised account |
| P3 Medium | Potential threat | 4 hours | Phishing campaign |
| P4 Low | Minor issue | 24 hours | Policy violation |

#### Evidence Preservation

```
DO:
- Screenshot error messages
- Note timestamps
- Preserve logs
- Document actions taken
- Save suspicious emails

DON'T:
- Delete files
- Reboot systems (unless instructed)
- Attempt to "fix" the issue
- Share details publicly
- Contact attacker
```

---

### Module 8: Physical Security

**Duration**: 20 minutes
**Audience**: All personnel with physical access
**Frequency**: Upon hire, annual refresher

#### Topics Covered

**8.1 Access Control**
- Badge usage and protection
- Visitor procedures
- Tailgating prevention
- Secure areas

**8.2 Clean Desk Policy**
```
Before leaving your workspace:
[ ] Lock computer (Win+L or Cmd+Ctrl+Q)
[ ] Secure sensitive documents
[ ] Clear whiteboards
[ ] Lock filing cabinets
[ ] Remove portable media
```

**8.3 Device Security**
```
Mobile devices and laptops:
- Never leave unattended
- Use cable locks when appropriate
- Enable device encryption
- Enable remote wipe capability
- Report lost/stolen immediately
```

**8.4 Visitor Management**
- All visitors must sign in
- Visitors must be escorted
- Visitor badges clearly visible
- Report unescorted visitors

---

## Training Schedule

### Required Training by Role

| Role | Required Modules | Deadline |
|------|------------------|----------|
| All Personnel | 1, 2, 3, 4, 5, 7, 8 | 30 days from hire |
| Developers | + Module 6 | 30 days from hire |
| Managers | + Extended Module 7 | 30 days from hire |
| Security Team | All modules + advanced | 14 days from hire |

### Annual Refresher Schedule

| Quarter | Training Focus |
|---------|----------------|
| Q1 | Modules 1-2 + new threats |
| Q2 | Modules 3-4 + phishing simulation |
| Q3 | Module 5 + privacy updates |
| Q4 | Modules 7-8 + incident drill |

---

## Compliance Tracking

### Training Records

All training completion is recorded with:
- Employee ID
- Module completed
- Completion date
- Assessment score
- Certificate issued

### Non-Compliance Escalation

| Days Overdue | Action |
|--------------|--------|
| 7 days | Email reminder |
| 14 days | Manager notification |
| 21 days | HR notification |
| 30 days | Access restriction |

### Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Completion rate | >95% | [TRACK] |
| Average score | >85% | [TRACK] |
| Phishing click rate | <5% | [TRACK] |
| Time to complete | <30 days | [TRACK] |

---

## Resources

### Contact Information

- Security Team: security@tensafe.io
- Privacy Team: privacy@tensafe.io
- Security Incidents: [SECURITY_HOTLINE]
- Training Questions: training@tensafe.io

### Reference Materials

- [Information Security Policy](../policies/INFORMATION_SECURITY_POLICY.md)
- [Acceptable Use Policy](../policies/ACCEPTABLE_USE_POLICY.md)
- [Data Classification Guide](../policies/DATA_CLASSIFICATION.md)
- [Incident Response Plan](../policies/INCIDENT_RESPONSE_PLAN.md)

### External Resources

- OWASP: https://owasp.org
- SANS Security Awareness: https://www.sans.org/security-awareness-training
- NIST Cybersecurity Framework: https://www.nist.gov/cyberframework
- HHS HIPAA Training: https://www.hhs.gov/hipaa/for-professionals/training

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | Security Team | Initial release |

---

*This training program is part of TenSafe's commitment to maintaining the highest security standards. All personnel are required to complete assigned training modules and maintain awareness of security best practices.*
