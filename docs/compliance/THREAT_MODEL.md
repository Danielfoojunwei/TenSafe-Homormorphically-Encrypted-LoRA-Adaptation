# Threat Model

> **Purpose**: Document security threats, attack vectors, and mitigations for the
> TensorGuard/TensorTrace platform to support compliance evidence collection.

---

## Table of Contents

1. [Overview](#overview)
2. [System Boundaries](#system-boundaries)
3. [Threat Actors](#threat-actors)
4. [STRIDE Analysis](#stride-analysis)
5. [Attack Vectors by Component](#attack-vectors-by-component)
6. [ML-Specific Threats](#ml-specific-threats)
7. [Mitigations Matrix](#mitigations-matrix)
8. [Residual Risks](#residual-risks)

---

## Overview

This threat model analyzes security risks for the TensorGuard/TensorTrace MLOps platform,
focusing on privacy-preserving federated learning for robotic fleets.

### Scope

- Training pipeline (data ingestion, preprocessing, LoRA fine-tuning)
- Inference serving (model serving, response generation)
- Artifact storage (adapters, checkpoints, configurations)
- Logging and telemetry (audit logs, metrics)
- Control plane (orchestration, management APIs)

### Out of Scope

- Physical security of deployment infrastructure
- End-user device security
- Third-party cloud provider security (assumed trusted)

---

## System Boundaries

```
+------------------------------------------------------------------+
|                        TRUST BOUNDARY                              |
|  +-----------------------------------------------------------+   |
|  |                    TensorGuard Platform                    |   |
|  |  +----------+  +----------+  +----------+  +----------+   |   |
|  |  | Training |  |Inference |  | Artifact |  | Control  |   |   |
|  |  | Pipeline |  | Service  |  |  Store   |  |  Plane   |   |   |
|  |  +----------+  +----------+  +----------+  +----------+   |   |
|  |       |              |             |             |         |   |
|  |  +---------------------------------------------------+    |   |
|  |  |              Internal Network (mTLS)               |    |   |
|  |  +---------------------------------------------------+    |   |
|  +-----------------------------------------------------------+   |
|                              |                                    |
+------------------------------+------------------------------------+
                               |
              +----------------+----------------+
              |                                 |
      +-------+-------+                 +-------+-------+
      | External APIs |                 |  Edge Agents  |
      | (Public)      |                 |  (Fleet)      |
      +---------------+                 +---------------+
```

### Trust Zones

| Zone | Trust Level | Description |
|------|-------------|-------------|
| Internal Platform | High | Core platform services |
| Edge Agents | Medium | Robotic fleet devices |
| External APIs | Low | Public-facing endpoints |
| External Data | Untrusted | Training data sources |

---

## Threat Actors

### TA-1: External Attacker

| Attribute | Value |
|-----------|-------|
| **Motivation** | Data theft, service disruption, competitive advantage |
| **Capability** | Network attacks, API exploitation, social engineering |
| **Access** | External network, public APIs |
| **Resources** | Moderate (tools, time) |

### TA-2: Malicious Insider

| Attribute | Value |
|-----------|-------|
| **Motivation** | Data exfiltration, sabotage, financial gain |
| **Capability** | Direct system access, credential abuse |
| **Access** | Internal systems, privileged accounts |
| **Resources** | High (legitimate access) |

### TA-3: Compromised Edge Device

| Attribute | Value |
|-----------|-------|
| **Motivation** | Lateral movement, data collection |
| **Capability** | Device-level access, local network access |
| **Access** | Edge network, device credentials |
| **Resources** | Limited (compromised device only) |

### TA-4: Supply Chain Attacker

| Attribute | Value |
|-----------|-------|
| **Motivation** | Widespread compromise, persistent access |
| **Capability** | Dependency poisoning, build system compromise |
| **Access** | Package repositories, CI/CD systems |
| **Resources** | High (sophisticated tooling) |

### TA-5: Nation-State Actor

| Attribute | Value |
|-----------|-------|
| **Motivation** | Espionage, IP theft, critical infrastructure disruption |
| **Capability** | Advanced persistent threats, zero-days, quantum computing |
| **Access** | Multiple vectors, long-term presence |
| **Resources** | Very high (nation-state funding) |

---

## STRIDE Analysis

### Spoofing (S)

| Threat ID | Description | Component | Impact |
|-----------|-------------|-----------|--------|
| S-1 | Attacker impersonates legitimate service | API Gateway | High |
| S-2 | Forged device identity for edge agent | Edge Agent | High |
| S-3 | Replay of authentication tokens | All | Medium |
| S-4 | Certificate impersonation | mTLS | High |

**Mitigations**:
- mTLS for all service-to-service communication
- JWT with short expiration and refresh rotation
- Device attestation using TPM
- Certificate pinning and rotation

### Tampering (T)

| Threat ID | Description | Component | Impact |
|-----------|-------------|-----------|--------|
| T-1 | Modification of training data | Data Pipeline | High |
| T-2 | Tampering with model weights | Artifact Store | Critical |
| T-3 | Log manipulation to hide attacks | Audit System | High |
| T-4 | Configuration modification | Control Plane | High |

**Mitigations**:
- Cryptographic signing of all artifacts (TGSP)
- Immutable audit log with hash chain
- Version control for configurations
- Integrity checks on data ingestion

### Repudiation (R)

| Threat ID | Description | Component | Impact |
|-----------|-------------|-----------|--------|
| R-1 | Denial of data access | All | Medium |
| R-2 | Denial of model deployment | Control Plane | Medium |
| R-3 | Denial of inference requests | Inference | Low |

**Mitigations**:
- Comprehensive audit logging
- Non-repudiation through digital signatures
- Immutable log storage
- Correlation IDs across systems

### Information Disclosure (I)

| Threat ID | Description | Component | Impact |
|-----------|-------------|-----------|--------|
| I-1 | Training data exposure | Data Pipeline | Critical |
| I-2 | Model weight theft | Artifact Store | High |
| I-3 | Inference prompt/response leakage | Inference | High |
| I-4 | Log exposure with sensitive data | Audit System | Medium |
| I-5 | PII in model outputs | Inference | High |

**Mitigations**:
- Encryption at rest (AES-256-GCM)
- Encryption in transit (TLS 1.3)
- PII scanning and redaction
- Differential privacy in training
- Access control with least privilege

### Denial of Service (D)

| Threat ID | Description | Component | Impact |
|-----------|-------------|-----------|--------|
| D-1 | API flooding | API Gateway | High |
| D-2 | Resource exhaustion in training | Training | Medium |
| D-3 | Storage exhaustion | Artifact Store | Medium |
| D-4 | Inference service overload | Inference | High |

**Mitigations**:
- Rate limiting at API gateway
- Resource quotas and limits
- Autoscaling with caps
- Circuit breakers for graceful degradation
- Queue-based request handling

### Elevation of Privilege (E)

| Threat ID | Description | Component | Impact |
|-----------|-------------|-----------|--------|
| E-1 | RBAC bypass | Control Plane | Critical |
| E-2 | Container escape | All | Critical |
| E-3 | SQL/command injection | APIs | High |
| E-4 | Privilege escalation via API | Control Plane | High |

**Mitigations**:
- RBAC with default-deny
- Container security hardening
- Input validation and parameterized queries
- Least privilege service accounts
- Regular security audits

---

## Attack Vectors by Component

### Training Pipeline

| Vector | Threat | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Data Poisoning | Inject malicious samples | Medium | High | Input validation, anomaly detection |
| Gradient Inversion | Recover training data | Low | Critical | Differential privacy, secure aggregation |
| Backdoor Injection | Insert trojan behavior | Low | Critical | Model integrity checks, behavioral testing |

### Inference Service

| Vector | Threat | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Prompt Injection | Execute unintended actions | High | Medium | Input sanitization, output filtering |
| Model Extraction | Steal model weights | Medium | High | Rate limiting, watermarking |
| Membership Inference | Detect training data | Medium | Medium | Output perturbation, DP |

### Artifact Store

| Vector | Threat | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| Unauthorized Access | Steal adapters | Medium | High | RBAC, encryption |
| Integrity Compromise | Modify weights | Low | Critical | Signing, hash verification |
| Ransomware | Encrypt/destroy artifacts | Low | Critical | Backups, immutable storage |

### Control Plane

| Vector | Threat | Likelihood | Impact | Mitigation |
|--------|--------|------------|--------|------------|
| API Exploitation | Gain control | Medium | Critical | Input validation, auth |
| Credential Theft | Impersonate admin | Medium | Critical | MFA, short-lived tokens |
| SSRF | Access internal services | Medium | High | Network segmentation |

---

## ML-Specific Threats

### Training Phase Threats

#### T-ML-1: Data Poisoning

**Description**: Attacker injects malicious samples into training data to influence model behavior.

**Attack Scenario**:
1. Attacker gains access to data pipeline
2. Injects adversarial examples
3. Model learns incorrect patterns
4. Deployed model exhibits malicious behavior

**Mitigations**:
- Data provenance tracking
- Anomaly detection on ingested data
- Multi-source data validation
- Human review sampling

#### T-ML-2: Gradient Inversion Attack

**Description**: Attacker reconstructs training data from model gradients in federated learning.

**Attack Scenario**:
1. Attacker intercepts gradient updates
2. Uses optimization to recover training samples
3. Extracts private training data

**Mitigations**:
- Differential privacy (epsilon tracking)
- Secure aggregation (N2HE)
- Gradient compression/sparsification
- Minimum batch sizes

### Inference Phase Threats

#### T-ML-3: Prompt Injection

**Description**: Attacker crafts prompts to bypass safety measures or extract sensitive information.

**Attack Scenario**:
1. Attacker sends malicious prompt
2. Prompt overrides system instructions
3. Model reveals sensitive data or performs harmful action

**Mitigations**:
- Input sanitization
- System prompt isolation
- Output filtering
- Behavioral monitoring

#### T-ML-4: Model Extraction

**Description**: Attacker queries model to reconstruct a functionally equivalent copy.

**Attack Scenario**:
1. Attacker sends many queries
2. Collects input-output pairs
3. Trains surrogate model

**Mitigations**:
- Query rate limiting
- API watermarking
- Response perturbation
- Access logging and anomaly detection

#### T-ML-5: Membership Inference

**Description**: Attacker determines if specific data was used in training.

**Attack Scenario**:
1. Attacker queries model with suspected training samples
2. Analyzes confidence scores
3. Infers training data membership

**Mitigations**:
- Differential privacy
- Output calibration
- Confidence score rounding
- Regularization during training

---

## Mitigations Matrix

### Control Mapping to Mitigations

| Control | ISO 27001 | SOC 2 | Mitigations |
|---------|-----------|-------|-------------|
| Authentication | A.9.4 | CC6.1 | mTLS, JWT, MFA |
| Authorization | A.9.2 | CC6.3 | RBAC, least privilege |
| Encryption (Rest) | A.10.1 | C1.1 | AES-256-GCM, KEK/DEK |
| Encryption (Transit) | A.10.1 | C1.1 | TLS 1.3, mTLS |
| Logging | A.12.4 | CC7.2 | Immutable logs, hash chain |
| Integrity | A.12.2 | PI1.4 | TGSP signing, hash verification |
| Input Validation | A.14.2 | CC7.1 | Schema validation, sanitization |
| Rate Limiting | A.12.1 | A1.1 | Per-client limits, circuit breakers |
| Privacy | - | P1-P8 | DP, PII scanning, redaction |

### Evidence Artifacts by Threat

| Threat Category | Evidence Artifact | Location |
|-----------------|-------------------|----------|
| Authentication | `auth_config.json` | `reports/compliance/<sha>/` |
| Encryption | `encryption_config.json` | `reports/compliance/<sha>/` |
| Integrity | `hash_manifest.json` | `reports/compliance/<sha>/` |
| PII Protection | `pii_scan.json` | `reports/compliance/<sha>/` |
| Secrets Hygiene | `secrets_scan.json` | `reports/compliance/<sha>/` |
| Audit Trail | `audit_integrity.json` | `reports/compliance/<sha>/` |

---

## Residual Risks

### Accepted Risks

| Risk ID | Description | Justification | Compensating Control |
|---------|-------------|---------------|---------------------|
| RR-1 | Zero-day vulnerabilities | Cannot fully prevent | Defense in depth, rapid patching |
| RR-2 | Insider threat with admin access | Operational necessity | Audit logging, separation of duties |
| RR-3 | Advanced persistent threats | Resource constraints | Monitoring, incident response |
| RR-4 | Quantum computing threats | Timeline uncertain | PQC migration in progress |

### Risk Acceptance Criteria

- Critical risks: Must be mitigated
- High risks: Mitigation required within 30 days
- Medium risks: Mitigation required within 90 days
- Low risks: Accept or mitigate opportunistically

### Monitoring and Detection

| Risk Area | Monitoring Method | Alert Threshold |
|-----------|-------------------|-----------------|
| Authentication failures | Log analysis | 5 failures/minute |
| Data exfiltration | Traffic analysis | Unusual volumes |
| Model extraction | Query pattern analysis | High query rates |
| Integrity violations | Hash verification | Any mismatch |
| PII exposure | Automated scanning | Any detection |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-29 | TensorGuard Team | Initial threat model |
