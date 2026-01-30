# Data Flow Documentation

> **Purpose**: Document data flows for training, inference, artifacts, and logs to support
> compliance evidence collection for ISO/IEC 27701, ISO/IEC 27001, and SOC 2.

---

## Table of Contents

1. [Overview](#overview)
2. [Training Pipeline Data Flow](#training-pipeline-data-flow)
3. [Inference Pipeline Data Flow](#inference-pipeline-data-flow)
4. [Artifact Storage Data Flow](#artifact-storage-data-flow)
5. [Logging and Telemetry Data Flow](#logging-and-telemetry-data-flow)
6. [Data Classification](#data-classification)
7. [Data Retention](#data-retention)
8. [Security Controls by Flow](#security-controls-by-flow)

---

## Overview

TensorGuard/TensorTrace is a post-quantum secure MLOps platform for privacy-preserving
federated learning. This document describes the data flows across all system components.

### System Components

```
+------------------+     +------------------+     +------------------+
|   Data Sources   |---->|  Training Pipe   |---->|  Artifact Store  |
|  - Datasets      |     |  - Preprocessing |     |  - Adapters      |
|  - Prompts       |     |  - LoRA Training |     |  - Checkpoints   |
+------------------+     |  - Evaluation    |     |  - Configs       |
                         +------------------+     +------------------+
                                  |                       |
                                  v                       v
                         +------------------+     +------------------+
                         |  Telemetry Sys   |     |  Inference Svc   |
                         |  - Audit Logs    |     |  - Model Serving |
                         |  - Metrics       |     |  - Response Gen  |
                         +------------------+     +------------------+
```

---

## Training Pipeline Data Flow

### 1. Data Ingestion

```
[External Data Sources]
         |
         | (1) Download/Import
         v
+------------------+
|   Raw Dataset    |
|   Storage        |
|   /data/raw/     |
+------------------+
         |
         | (2) Validation
         |     - Schema check
         |     - PII scan
         v
+------------------+
|   Staging Area   |
|   /data/stage/   |
+------------------+
```

**Data Types Handled**:
- Training datasets (prompts, completions)
- Evaluation datasets
- Benchmark datasets

**Security Controls**:
- Input validation on all ingested data
- PII scanning before staging
- Access logging for data downloads
- Encryption at rest for staged data

### 2. Preprocessing

```
[Staging Area]
         |
         | (3) Preprocessing
         |     - Tokenization
         |     - Length filtering
         |     - Quality filtering
         |     - PII redaction
         v
+------------------+
|   Processed      |
|   Dataset        |
|   /data/proc/    |
+------------------+
         |
         | (4) Dataset hash
         |     recorded
         v
[Training Pipeline]
```

**Data Transformations**:
- Tokenization with configurable tokenizer
- Maximum length enforcement (default: 2048 tokens)
- Quality filtering (language detection, formatting)
- Optional PII redaction

**Metrics Emitted**:
- `columns_dropped_pct`: Percentage of columns removed
- `examples_filtered_pct`: Percentage of examples filtered
- `dataset_hash`: SHA-256 of processed dataset

### 3. Model Training

```
[Processed Dataset]
         |
         | (5) Training Loop
         |     - Forward pass
         |     - Loss computation
         |     - Backward pass
         |     - Optimizer step
         v
+------------------+
|   Adapter        |
|   Checkpoints    |
|   /artifacts/    |
|   adapters/      |
+------------------+
         |
         | (6) Adapter hash
         |     recorded
         v
[Artifact Store]
```

**Privacy Controls**:
- Differential privacy (optional): epsilon budget tracked
- Gradient clipping to prevent memorization
- No raw training data in checkpoints

**Artifacts Produced**:
- LoRA adapter weights (`.safetensors`)
- Training configuration (`.json`)
- Training metrics log

### 4. Evaluation

```
[Trained Adapter]
         |
         | (7) Evaluation
         |     - Benchmark suite
         |     - Metric computation
         v
+------------------+
|   Evaluation     |
|   Results        |
|   /reports/      |
|   bench/         |
+------------------+
```

**Data Handled**:
- Evaluation prompts (may contain sensitive examples)
- Model responses (checked for PII leakage)
- Benchmark metrics

---

## Inference Pipeline Data Flow

### 1. Request Ingestion

```
[Client Request]
         |
         | (1) API Gateway
         |     - Authentication
         |     - Rate limiting
         |     - Input validation
         v
+------------------+
|   Request        |
|   Queue          |
+------------------+
         |
         | (2) Request logging
         |     (redacted)
         v
[Inference Service]
```

**Security Controls**:
- mTLS/JWT authentication required
- Request rate limiting
- Input sanitization
- Request logging with PII redaction

### 2. Model Inference

```
[Request Queue]
         |
         | (3) Model Selection
         |     - Adapter routing
         |     - Safety checks
         v
+------------------+
|   Inference      |
|   Engine         |
|   - Base model   |
|   - Adapter      |
+------------------+
         |
         | (4) Generation
         |     - Token sampling
         |     - Safety filtering
         v
+------------------+
|   Response       |
|   Buffer         |
+------------------+
```

**Privacy Controls**:
- Prompt isolation between tenants
- No prompt caching by default
- Response safety filtering

### 3. Response Delivery

```
[Response Buffer]
         |
         | (5) Response validation
         |     - PII check
         |     - Safety filter
         v
+------------------+
|   Response       |
|   Formatting     |
+------------------+
         |
         | (6) Response logging
         |     (metadata only)
         v
[Client]
```

**Data Logged**:
- Request ID
- Latency metrics
- Token counts
- Safety filter triggers
- **NOT logged**: Full prompts/responses (by default)

---

## Artifact Storage Data Flow

### 1. Artifact Creation

```
[Training/Inference Output]
         |
         | (1) Artifact packaging
         |     - TGSP format
         |     - Encryption
         |     - Signing
         v
+------------------+
|   TGSP Package   |
|   - Payload      |
|   - Manifest     |
|   - Signatures   |
+------------------+
```

**TGSP (TensorGuard Secure Package)**:
- Deterministic packaging format
- HPKE encryption (hybrid post-quantum)
- Dilithium signatures (PQC)

### 2. Artifact Storage

```
[TGSP Package]
         |
         | (2) Storage
         |     - Encrypted at rest
         |     - Access controlled
         v
+------------------+
|   Artifact       |
|   Store          |
|   /artifacts/    |
|   - adapters/    |
|   - models/      |
|   - configs/     |
+------------------+
         |
         | (3) Audit log entry
         v
[Audit Log]
```

**Storage Controls**:
- Encryption at rest (AES-256-GCM or CKKS for HE)
- Key management with KEK/DEK hierarchy
- Access control per artifact
- Immutable audit trail

### 3. Artifact Retrieval

```
[Retrieval Request]
         |
         | (4) Authorization check
         v
+------------------+
|   Access         |
|   Control        |
|   Layer          |
+------------------+
         |
         | (5) Decryption
         |     Signature verify
         v
[Requesting Service]
```

---

## Logging and Telemetry Data Flow

### 1. Event Generation

```
[System Components]
    |
    +-- Training events
    +-- Inference events
    +-- Security events
    +-- Access events
    |
    v
+------------------+
|   Event          |
|   Emitter        |
|   - Formatting   |
|   - Redaction    |
+------------------+
```

**Event Types**:
- `STAGE`: Pipeline stage completions
- `SYSTEM`: Resource metrics
- `SECURITY`: Auth/access events
- `FORENSICS`: Audit-critical events

### 2. Log Processing

```
[Event Emitter]
         |
         | (1) Batching
         v
+------------------+
|   Event          |
|   Queue          |
+------------------+
         |
         | (2) Hash chain
         |     computation
         v
+------------------+
|   Audit Log      |
|   Store          |
|   - Immutable    |
|   - Signed       |
+------------------+
```

**Integrity Controls**:
- Hash chain linking events
- Periodic signature of log segments
- Tamper detection

### 3. Telemetry Export

```
[Audit Log Store]
         |
         | (3) Export to
         |     external systems
         v
+------------------+
|   SIEM/          |
|   Observability  |
|   - OpenTelemetry|
|   - Prometheus   |
+------------------+
```

---

## Data Classification

| Data Category | Classification | Retention | Encryption | Access Control |
|---------------|----------------|-----------|------------|----------------|
| Training Datasets | Internal/Confidential | Per policy | At rest | RBAC |
| Trained Adapters | Internal | Per policy | At rest + transit | RBAC |
| Inference Prompts | Confidential | Session only | Transit | Per-request |
| Inference Responses | Confidential | Session only | Transit | Per-request |
| Audit Logs | Internal | 365 days | At rest | Admin only |
| System Metrics | Public | 90 days | Optional | Read-only |
| Configuration | Internal | Indefinite | At rest | Admin only |

---

## Data Retention

### Default Retention Policies

| Data Type | Retention Period | Disposal Method |
|-----------|------------------|-----------------|
| Training data (raw) | 30 days | Secure delete |
| Training data (processed) | 90 days | Secure delete |
| Adapters/Checkpoints | 1 year | Archive then delete |
| Inference logs | 7 days | Rotate and delete |
| Audit logs | 365 days | Archive then delete |
| System metrics | 90 days | Rotate and delete |

### Retention Enforcement

```
[Scheduled Job: Daily]
         |
         | (1) Scan for expired data
         v
+------------------+
|   Retention      |
|   Scanner        |
+------------------+
         |
         | (2) Secure deletion
         |     - Overwrite
         |     - Key destruction
         v
+------------------+
|   Deletion       |
|   Log Entry      |
+------------------+
```

---

## Security Controls by Flow

### Training Pipeline

| Control | Implementation | Evidence Artifact |
|---------|----------------|-------------------|
| Input Validation | Schema validation + PII scan | `preprocessing_summary.json` |
| Data Encryption | AES-256-GCM at rest | `encryption_config.json` |
| Access Logging | All data access logged | `audit.log` |
| Integrity | Dataset hash recorded | `hash_manifest.json` |

### Inference Pipeline

| Control | Implementation | Evidence Artifact |
|---------|----------------|-------------------|
| Authentication | mTLS/JWT required | `auth_config.json` |
| Rate Limiting | Per-client limits | `rate_limit_config.json` |
| PII Protection | Response filtering | `pii_scan.json` |
| Audit Trail | Request metadata logged | `audit.log` |

### Artifact Storage

| Control | Implementation | Evidence Artifact |
|---------|----------------|-------------------|
| Encryption at Rest | KEK/DEK hierarchy | `key_inventory.json` |
| Integrity | TGSP signatures | `*.tgsp.sig` |
| Access Control | RBAC policies | `rbac_policy.json` |
| Key Rotation | Automated rotation | `key_rotation_audit.log` |

### Logging

| Control | Implementation | Evidence Artifact |
|---------|----------------|-------------------|
| Immutability | Hash chain | `audit_integrity.json` |
| Completeness | Event coverage tracking | `log_coverage.json` |
| Retention | Automated enforcement | `retention_report.json` |
| Redaction | PII removed from logs | `redaction_config.json` |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-29 | TensorGuard Team | Initial data flow documentation |
