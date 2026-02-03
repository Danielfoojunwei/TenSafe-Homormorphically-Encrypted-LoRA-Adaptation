# Data Processing Agreement (DPA) Template

**Version**: 1.0
**Effective Date**: [DATE]
**Document Classification**: CONFIDENTIAL

---

## PARTIES

This Data Processing Agreement ("Agreement" or "DPA") is entered into between:

**Data Controller** ("Controller"):
- Organization Name: [CONTROLLER_NAME]
- Address: [CONTROLLER_ADDRESS]
- Contact: [CONTROLLER_CONTACT]
- Data Protection Officer: [DPO_NAME] ([DPO_EMAIL])

**Data Processor** ("Processor"):
- Organization Name: TenSafe / TensorGuard Platform
- Address: [PROCESSOR_ADDRESS]
- Contact: privacy@tensafe.io
- Data Protection Officer: [TENSAFE_DPO]

Collectively referred to as the "Parties."

---

## 1. DEFINITIONS

**1.1** "Personal Data" means any information relating to an identified or identifiable natural person as defined in Article 4(1) GDPR.

**1.2** "Processing" means any operation performed on Personal Data, including collection, recording, organization, structuring, storage, adaptation, retrieval, consultation, use, disclosure, alignment, restriction, erasure, or destruction.

**1.3** "Data Subject" means the identified or identifiable natural person to whom the Personal Data relates.

**1.4** "Sub-processor" means any third party engaged by the Processor to process Personal Data on behalf of the Controller.

**1.5** "Supervisory Authority" means the independent public authority responsible for monitoring GDPR application.

**1.6** "Technical and Organizational Measures" means security measures implemented to ensure appropriate protection of Personal Data.

---

## 2. SCOPE AND PURPOSE

### 2.1 Subject Matter

This DPA governs the processing of Personal Data by the Processor on behalf of the Controller in connection with the TenSafe/TensorGuard federated learning platform services.

### 2.2 Nature and Purpose of Processing

| Processing Activity | Purpose | Legal Basis |
|---------------------|---------|-------------|
| Model Training | Federated learning on encrypted data | Legitimate Interest / Contract |
| Access Control | User authentication and authorization | Contract Performance |
| Audit Logging | Security and compliance monitoring | Legal Obligation |
| Analytics | Service improvement (aggregated only) | Legitimate Interest |

### 2.3 Types of Personal Data

The following categories of Personal Data may be processed:

- [ ] User account information (name, email, organization)
- [ ] Authentication credentials (hashed passwords, MFA tokens)
- [ ] Access logs and audit trails
- [ ] Model training metadata
- [ ] Encrypted training data (homomorphically encrypted)
- [ ] Other: [SPECIFY]

### 2.4 Categories of Data Subjects

- [ ] Employees of Controller
- [ ] Contractors of Controller
- [ ] End users of Controller's services
- [ ] Other: [SPECIFY]

### 2.5 Duration of Processing

Processing shall continue for the duration of the service agreement, unless terminated earlier in accordance with Section 12.

---

## 3. CONTROLLER OBLIGATIONS

### 3.1 Lawful Processing

The Controller warrants that:

a) It has a lawful basis for processing Personal Data under GDPR Article 6 (and Article 9 for special categories)

b) It has provided appropriate privacy notices to Data Subjects

c) It has obtained necessary consents where required

d) The processing instructions given to the Processor comply with applicable law

### 3.2 Data Accuracy

The Controller is responsible for ensuring the accuracy and quality of Personal Data provided to the Processor.

### 3.3 Risk Assessment

The Controller shall conduct appropriate Data Protection Impact Assessments (DPIAs) where required under GDPR Article 35.

---

## 4. PROCESSOR OBLIGATIONS

### 4.1 Processing Limitations (GDPR Article 28(3)(a))

The Processor shall:

a) Process Personal Data only on documented instructions from the Controller

b) Not process Personal Data for any purpose other than as specified in this DPA

c) Inform the Controller if, in the Processor's opinion, an instruction infringes GDPR

### 4.2 Confidentiality (GDPR Article 28(3)(b))

The Processor shall ensure that persons authorized to process Personal Data:

a) Have committed to confidentiality or are under statutory confidentiality obligations

b) Process Personal Data only as instructed by the Controller

c) Receive appropriate training on data protection requirements

### 4.3 Security Measures (GDPR Article 28(3)(c))

The Processor implements the following Technical and Organizational Measures:

#### 4.3.1 Encryption

| Data State | Algorithm | Key Size | Standard |
|------------|-----------|----------|----------|
| At Rest | AES-256-GCM | 256-bit | NIST SP 800-38D |
| In Transit | TLS 1.3 | 256-bit | RFC 8446 |
| During Processing | CKKS (Homomorphic) | 128-bit security | CKKS Scheme |

#### 4.3.2 Access Control

- Role-Based Access Control (RBAC) with least privilege
- Multi-Factor Authentication (MFA) for administrative access
- Unique user identification and session management
- Automatic session timeout (30 minutes)

#### 4.3.3 Audit Logging

- Comprehensive audit trail with 60+ event types
- Tamper-evident logging with SHA-256 hash chain
- 365-day log retention
- Automated anomaly detection

#### 4.3.4 Data Integrity

- SHA-256 hash verification for all artifacts
- Digital signatures (Ed25519 + Dilithium3 hybrid)
- Input validation and sanitization
- Immutable audit records

### 4.4 Sub-processors (GDPR Article 28(3)(d))

#### 4.4.1 Authorization

- [ ] Controller provides general authorization for Sub-processors
- [ ] Controller requires specific authorization for each Sub-processor

#### 4.4.2 Current Sub-processors

| Sub-processor | Purpose | Location | DPA Status |
|---------------|---------|----------|------------|
| [Cloud Provider] | Infrastructure hosting | [Region] | Signed |
| [EXAMPLE] | [PURPOSE] | [LOCATION] | [STATUS] |

#### 4.4.3 Changes to Sub-processors

The Processor shall:

a) Provide [30] days' notice before engaging new Sub-processors

b) Allow the Controller to object to new Sub-processors

c) Ensure Sub-processors are bound by equivalent data protection obligations

### 4.5 Data Subject Rights (GDPR Article 28(3)(e))

The Processor shall assist the Controller in responding to Data Subject requests for:

| Right | GDPR Article | Response Time | Process |
|-------|--------------|---------------|---------|
| Access | Article 15 | 30 days | DSAR Portal |
| Rectification | Article 16 | 30 days | Support Ticket |
| Erasure | Article 17 | 30 days | Automated + Manual |
| Restriction | Article 18 | 72 hours | Flag in System |
| Portability | Article 20 | 30 days | JSON/CSV Export |
| Objection | Article 21 | 30 days | Case Review |

### 4.6 Security Incident Notification (GDPR Article 28(3)(f))

#### 4.6.1 Notification Timeline

The Processor shall notify the Controller of any Personal Data Breach:

- **Initial Notice**: Within 24 hours of becoming aware
- **Detailed Report**: Within 48 hours with full incident details
- **Root Cause Analysis**: Within 14 days

#### 4.6.2 Notification Content

Breach notifications shall include:

a) Nature of the breach including categories and approximate number of Data Subjects

b) Name and contact details of the DPO or other contact

c) Likely consequences of the breach

d) Measures taken or proposed to address the breach

### 4.7 DPIA Assistance (GDPR Article 28(3)(f))

The Processor shall assist the Controller with:

a) Data Protection Impact Assessments

b) Prior consultations with Supervisory Authorities

c) Providing necessary information about processing operations

### 4.8 Audit Rights (GDPR Article 28(3)(h))

The Processor shall:

a) Make available all information necessary to demonstrate compliance

b) Allow for and contribute to audits and inspections

c) Provide Controller with audit reports upon request

**Audit Frequency**: [Annual / Upon reasonable request with 30 days' notice]

### 4.9 Deletion and Return (GDPR Article 28(3)(g))

Upon termination of services, the Processor shall:

- [ ] Delete all Personal Data within [30] days
- [ ] Return all Personal Data to Controller
- [ ] Certify deletion in writing

---

## 5. INTERNATIONAL TRANSFERS

### 5.1 Transfer Restrictions

Personal Data shall not be transferred outside the European Economic Area (EEA) unless:

a) The destination country has an adequacy decision (GDPR Article 45)

b) Appropriate safeguards are in place (GDPR Article 46)

c) A derogation applies (GDPR Article 49)

### 5.2 Transfer Mechanisms

| Destination | Mechanism | Documentation |
|-------------|-----------|---------------|
| [USA] | EU-US Data Privacy Framework | [Certificate #] |
| [UK] | UK Adequacy Decision | N/A |
| [Other] | Standard Contractual Clauses | Attached as Annex |

### 5.3 Supplementary Measures

Where SCCs are used, the following supplementary measures apply:

a) End-to-end encryption with Controller-held keys

b) Homomorphic encryption for data processing

c) Access controls preventing foreign government access

d) Transparency reporting for government requests

---

## 6. SPECIAL CATEGORIES OF DATA

### 6.1 Processing Restrictions

If special categories of data (GDPR Article 9) are processed:

- [ ] Explicit consent obtained from Data Subjects
- [ ] Processing necessary for employment law obligations
- [ ] Processing necessary for vital interests
- [ ] Other legal basis: [SPECIFY]

### 6.2 Additional Safeguards

For special category data, the Processor shall implement:

a) Enhanced access controls (additional approval workflow)

b) Encryption at all times (no plaintext processing)

c) Audit log review within 24 hours of any access

d) Annual security assessment specific to sensitive data

---

## 7. COMPLIANCE CERTIFICATIONS

### 7.1 Current Certifications

The Processor maintains the following compliance certifications:

| Certification | Scope | Valid Until | Auditor |
|---------------|-------|-------------|---------|
| SOC 2 Type II | Security, Availability | [DATE] | [AUDITOR] |
| ISO 27001:2022 | ISMS | [DATE] | [AUDITOR] |
| ISO 27701:2019 | PIMS | [DATE] | [AUDITOR] |

### 7.2 HIPAA (if applicable)

- [ ] This DPA incorporates HIPAA Business Associate Agreement provisions
- [ ] Processor agrees to comply with HIPAA Security Rule requirements

---

## 8. LIABILITY AND INDEMNIFICATION

### 8.1 Processor Liability

The Processor shall be liable for damages caused by processing that:

a) Does not comply with GDPR obligations specific to processors

b) Acts outside or contrary to lawful Controller instructions

### 8.2 Limitation of Liability

Total liability under this DPA shall not exceed: [AMOUNT or refer to main agreement]

### 8.3 Indemnification

Each party shall indemnify the other against claims arising from its breach of this DPA.

---

## 9. TERM AND TERMINATION

### 9.1 Term

This DPA shall remain in effect for the duration of the underlying service agreement.

### 9.2 Termination for Breach

Either party may terminate this DPA if the other party:

a) Materially breaches this DPA and fails to cure within 30 days of notice

b) Repeatedly breaches data protection obligations

### 9.3 Effect of Termination

Upon termination:

a) Processor shall cease all processing of Personal Data

b) Processor shall delete or return Personal Data per Section 4.9

c) Provisions that should survive termination shall remain in effect

---

## 10. GOVERNING LAW AND JURISDICTION

### 10.1 Governing Law

This DPA shall be governed by the laws of [JURISDICTION].

### 10.2 Jurisdiction

Disputes shall be resolved in the courts of [JURISDICTION].

### 10.3 Supervisory Authority

The lead Supervisory Authority for this processing is: [AUTHORITY NAME]

---

## 11. AMENDMENTS

### 11.1 Modification

This DPA may only be amended in writing signed by both Parties.

### 11.2 Regulatory Changes

If changes in data protection law require amendments to this DPA, the Parties shall negotiate in good faith to update the agreement.

---

## 12. ANNEXES

The following annexes form part of this DPA:

- **Annex A**: Technical and Organizational Measures (detailed)
- **Annex B**: List of Sub-processors
- **Annex C**: Standard Contractual Clauses (if applicable)
- **Annex D**: Controller's Processing Instructions

---

## SIGNATURES

**Data Controller**:

Signature: _________________________

Name: [NAME]

Title: [TITLE]

Date: _________________________


**Data Processor (TenSafe)**:

Signature: _________________________

Name: [NAME]

Title: [TITLE]

Date: _________________________

---

## ANNEX A: TECHNICAL AND ORGANIZATIONAL MEASURES

### A.1 Physical Security

| Measure | Implementation |
|---------|----------------|
| Data Center Security | SOC 2 Type II certified facilities |
| Access Control | Biometric + badge access |
| Environmental Controls | Fire suppression, climate control |
| Redundancy | Multi-region deployment |

### A.2 Network Security

| Measure | Implementation |
|---------|----------------|
| Firewall | Web Application Firewall (WAF) |
| Intrusion Detection | IDS/IPS monitoring |
| DDoS Protection | Cloud-native DDoS mitigation |
| Network Segmentation | VPC isolation per tenant |

### A.3 Application Security

| Measure | Implementation |
|---------|----------------|
| Authentication | JWT + Argon2id password hashing |
| Authorization | RBAC with least privilege |
| Input Validation | SQL, XSS, Command injection prevention |
| Secure Development | OWASP guidelines, code review |

### A.4 Data Security

| Measure | Implementation |
|---------|----------------|
| Encryption at Rest | AES-256-GCM |
| Encryption in Transit | TLS 1.3 |
| Key Management | KEK/DEK hierarchy, 90-day rotation |
| Data Masking | PII redaction in logs |

### A.5 Operational Security

| Measure | Implementation |
|---------|----------------|
| Monitoring | 24/7 security monitoring |
| Incident Response | Documented IR procedures |
| Backup | Daily backups, 30-day retention |
| Disaster Recovery | RTO: 4 hours, RPO: 1 hour |

---

## ANNEX B: SUB-PROCESSOR LIST

| Sub-processor | Service | Location | Data Processed |
|---------------|---------|----------|----------------|
| [Provider 1] | Cloud Hosting | [Region] | All categories |
| [Provider 2] | CDN | [Region] | Static assets only |
| [Provider 3] | Email | [Region] | Contact info |

Last Updated: [DATE]

---

## ANNEX C: STANDARD CONTRACTUAL CLAUSES

[If applicable, attach EU Commission Standard Contractual Clauses for international transfers]

---

## ANNEX D: PROCESSING INSTRUCTIONS

The Controller instructs the Processor to process Personal Data for the following purposes:

1. **User Authentication**: Process account credentials for access control
2. **Model Training**: Process encrypted training data using homomorphic encryption
3. **Audit Logging**: Record access and processing activities
4. **Service Delivery**: Provide federated learning platform services
5. **Support**: Respond to user inquiries and technical issues

Additional Instructions:
[CONTROLLER TO SPECIFY]

---

*This template is provided for informational purposes. Organizations should consult with legal counsel to ensure compliance with applicable data protection laws.*
