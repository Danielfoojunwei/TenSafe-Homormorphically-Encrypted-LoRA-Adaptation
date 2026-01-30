# Compliance Control Matrix

> **Purpose**: This document maps privacy and security controls from ISO/IEC 27701, ISO/IEC 27001,
> and SOC 2 Trust Services Criteria to measurable system telemetry and evidence artifacts.
>
> **Disclaimer**: This is an evidence collection framework, not a certification claim.
> All metrics are objective, machine-readable, and reproducible.

---

## Table of Contents

1. [ISO/IEC 27701 - Privacy Information Management](#isoiec-27701---privacy-information-management)
2. [ISO/IEC 27001 - Information Security Management System (ISMS)](#isoiec-27001---information-security-management-system-isms)
3. [SOC 2 - Trust Services Criteria](#soc-2---trust-services-criteria)
4. [Metrics Summary](#metrics-summary)
5. [Evidence Artifacts Index](#evidence-artifacts-index)

---

## ISO/IEC 27701 - Privacy Information Management

### PIM-1: Consent and Purpose Limitation

| Attribute | Value |
|-----------|-------|
| **Control Family** | Consent & Purpose Limitation |
| **Objective** | Ensure personal data is processed only for specified, legitimate purposes |
| **Measurable Telemetry** | |
| - `purpose_tags` | Array of purpose labels attached to each dataset (training/eval/bench) |
| - `consent_metadata_present` | Boolean: dataset includes consent/license metadata |
| - `purpose_drift_detected` | Boolean: data used for undeclared purpose |
| **Evidence Artifacts** | |
| - Dataset config files | `configs/datasets/*.yaml` with `purpose:` field |
| - Data lineage logs | `reports/compliance/<sha>/data_lineage.json` |
| **Gaps** | Real-time consent tracking not implemented; relies on static dataset metadata |

### PIM-2: Data Minimization

| Attribute | Value |
|-----------|-------|
| **Control Family** | Data Minimization |
| **Objective** | Collect and process only data necessary for the stated purpose |
| **Measurable Telemetry** | |
| - `columns_dropped_pct` | Percentage of dataset columns dropped during preprocessing |
| - `examples_filtered_pct` | Percentage of examples filtered (language/quality/PII) |
| - `max_prompt_length` | Maximum token length enforced |
| - `data_sampling_ratio` | Ratio of data actually used vs. available |
| **Evidence Artifacts** | |
| - Preprocessing summary | `reports/compliance/<sha>/preprocessing_summary.json` |
| - Filter logs | Audit log entries with `event_type=DATA_FILTER` |
| **Gaps** | Column-level tracking requires explicit schema definitions |

### PIM-3: Retention and Disposal

| Attribute | Value |
|-----------|-------|
| **Control Family** | Data Retention |
| **Objective** | Retain data only as long as necessary; ensure secure disposal |
| **Measurable Telemetry** | |
| - `retention_policy_days` | Configured retention period |
| - `retention_enforced` | Boolean: cleanup job executed successfully |
| - `temp_files_deleted` | Count of temporary files cleaned up |
| - `artifact_age_max_days` | Maximum age of retained artifacts |
| **Evidence Artifacts** | |
| - Retention policy config | `configs/retention_policy.yaml` |
| - Cleanup job logs | Audit log entries with `event_type=RETENTION` |
| - Tempdir scan results | `reports/compliance/<sha>/tempdir_scan.json` |
| **Gaps** | Cross-system retention (external storage) requires manual verification |

### PIM-4: Privacy by Design

| Attribute | Value |
|-----------|-------|
| **Control Family** | Privacy by Design |
| **Objective** | Embed privacy into system design and default settings |
| **Measurable Telemetry** | |
| - `dp_enabled` | Boolean: differential privacy enabled |
| - `dp_epsilon` | Privacy budget consumed |
| - `encryption_default` | Boolean: encryption enabled by default |
| - `anonymization_applied` | Boolean: PII redaction in logs enabled |
| **Evidence Artifacts** | |
| - Privacy config | `configs/privacy_config.yaml` |
| - DP accountant state | `reports/compliance/<sha>/dp_state.json` |
| - Log redaction test results | Test output showing redaction effectiveness |
| **Gaps** | DPIA documentation is manual |

### PIM-5: Data Subject Rights

| Attribute | Value |
|-----------|-------|
| **Control Family** | Data Subject Rights |
| **Objective** | Enable data access, correction, deletion, and portability |
| **Measurable Telemetry** | |
| - `data_export_capability` | Boolean: API/tooling for data export exists |
| - `data_deletion_capability` | Boolean: API/tooling for deletion exists |
| - `audit_trail_queryable` | Boolean: audit logs support per-subject queries |
| **Evidence Artifacts** | |
| - API documentation | `docs/api/data_subject_rights.md` |
| - Integration tests | `tests/integration/test_data_subject_*.py` |
| **Gaps** | Automated DSAR workflow not implemented |

---

## ISO/IEC 27001 - Information Security Management System (ISMS)

### ISMS-AC: Access Control

| Attribute | Value |
|-----------|-------|
| **Control Family** | Access Control (A.9) |
| **Objective** | Ensure authorized access and prevent unauthorized access |
| **Measurable Telemetry** | |
| - `authn_method` | Authentication method (API_KEY/JWT/OIDC/mTLS) |
| - `authn_enabled` | Boolean: authentication enforced |
| - `authz_model` | Authorization model (RBAC/ABAC) |
| - `default_deny` | Boolean: default-deny policy active |
| - `admin_role_count` | Number of accounts with admin privileges |
| - `least_privilege_score` | Score (0-100) based on permission analysis |
| - `mfa_enabled` | Boolean: MFA configured |
| **Evidence Artifacts** | |
| - Auth config snapshot | `reports/compliance/<sha>/auth_config.json` |
| - RBAC policy file | `configs/rbac_policy.json` |
| - Policy test results | `reports/compliance/<sha>/policy_tests.json` |
| **Gaps** | Continuous access review automation not implemented |

### ISMS-CRYPTO: Cryptography

| Attribute | Value |
|-----------|-------|
| **Control Family** | Cryptography (A.10) |
| **Objective** | Protect data confidentiality and integrity through encryption |
| **Measurable Telemetry** | |
| - `at_rest_encryption_enabled` | Boolean: artifacts encrypted at rest |
| - `in_transit_encryption_enabled` | Boolean: TLS/mTLS enforced |
| - `tls_min_version` | Minimum TLS version (1.2/1.3) |
| - `kek_present` | Boolean: Key Encryption Key configured |
| - `dek_per_tenant` | Boolean: tenant-specific DEKs |
| - `key_rotation_policy` | Boolean: rotation policy configured |
| - `key_rotation_last_days` | Days since last key rotation |
| - `pqc_enabled` | Boolean: post-quantum crypto enabled |
| **Evidence Artifacts** | |
| - Encryption config | `configs/encryption_config.yaml` |
| - Key inventory | `reports/compliance/<sha>/key_inventory.json` |
| - TLS scan results | `reports/compliance/<sha>/tls_scan.json` |
| **Gaps** | HSM attestation requires external verification |

### ISMS-LOG: Logging and Monitoring

| Attribute | Value |
|-----------|-------|
| **Control Family** | Logging & Monitoring (A.12.4) |
| **Objective** | Record and monitor security events |
| **Measurable Telemetry** | |
| - `audit_log_enabled` | Boolean: audit logging active |
| - `log_integrity_verified` | Boolean: hash chain verification passed |
| - `event_coverage_pct` | Percentage of critical operations logged |
| - `log_retention_days` | Configured log retention period |
| - `log_tamper_detection` | Boolean: immutability/signing enabled |
| - `siem_integration` | Boolean: SIEM forwarding configured |
| **Evidence Artifacts** | |
| - Audit log sample | `reports/compliance/<sha>/audit_log_sample.json` |
| - Integrity report | `reports/compliance/<sha>/audit_integrity.json` |
| - Coverage report | `reports/compliance/<sha>/log_coverage.json` |
| **Gaps** | Real-time alerting requires external SIEM |

### ISMS-CHANGE: Change Management

| Attribute | Value |
|-----------|-------|
| **Control Family** | Change Management (A.12.1, A.14.2) |
| **Objective** | Control changes to systems and ensure traceability |
| **Measurable Telemetry** | |
| - `git_sha` | Current deployment commit hash |
| - `dirty_tree` | Boolean: uncommitted changes present |
| - `ci_run_id` | CI pipeline run identifier |
| - `dependency_lockfile_present` | Boolean: requirements locked |
| - `reproducible_command` | String: command to reproduce build |
| - `code_review_required` | Boolean: PR review enforcement |
| **Evidence Artifacts** | |
| - Run config | `reports/compliance/<sha>/run_config.json` |
| - CI metadata | `reports/compliance/<sha>/ci_metadata.json` |
| - Lockfile hash | Hash of requirements.lock |
| **Gaps** | Manual approval tracking not automated |

### ISMS-BACKUP: Backup and Recovery

| Attribute | Value |
|-----------|-------|
| **Control Family** | Backup & Disaster Recovery (A.12.3, A.17) |
| **Objective** | Ensure data availability and recovery capability |
| **Measurable Telemetry** | |
| - `backup_enabled` | Boolean: backup jobs configured |
| - `backup_last_success_hours` | Hours since last successful backup |
| - `recovery_tested` | Boolean: recovery procedure tested |
| - `rto_configured` | Boolean: RTO defined |
| - `rpo_configured` | Boolean: RPO defined |
| **Evidence Artifacts** | |
| - Backup config | `configs/backup_config.yaml` |
| - Backup job logs | Audit entries with `event_type=BACKUP` |
| - Recovery test results | `reports/compliance/<sha>/recovery_test.json` |
| **Gaps** | Automated DR testing not implemented |

### ISMS-SUPPLIER: Supplier Management

| Attribute | Value |
|-----------|-------|
| **Control Family** | Supplier Relationships (A.15) |
| **Objective** | Manage security in supplier relationships |
| **Measurable Telemetry** | |
| - `dependency_count` | Number of third-party dependencies |
| - `vulnerability_scan_passed` | Boolean: no critical CVEs |
| - `supply_chain_signed` | Boolean: package signatures verified |
| - `sbom_generated` | Boolean: SBOM available |
| **Evidence Artifacts** | |
| - SBOM | `reports/compliance/<sha>/sbom.json` |
| - Vulnerability report | `reports/compliance/<sha>/vuln_scan.json` |
| - Dependency list | `requirements.lock` |
| **Gaps** | Vendor security questionnaires are manual |

---

## SOC 2 - Trust Services Criteria

### TSC-SEC: Security

| Attribute | Value |
|-----------|-------|
| **Control Family** | Security (CC1-CC9) |
| **Objective** | Protect against unauthorized access |
| **Measurable Telemetry** | |
| - `access_control_posture` | Composite score from ISMS-AC metrics |
| - `encryption_posture` | Composite score from ISMS-CRYPTO metrics |
| - `vulnerability_count` | Number of open vulnerabilities |
| - `secrets_exposed` | Count of secrets found in codebase (target: 0) |
| - `penetration_test_passed` | Boolean: last pentest passed |
| **Evidence Artifacts** | |
| - Secrets scan | `reports/compliance/<sha>/secrets_scan.json` |
| - Security test results | `reports/compliance/<sha>/security_tests.json` |
| - Config hardening report | `reports/compliance/<sha>/hardening.json` |
| **Gaps** | External penetration testing is manual |

### TSC-AVAIL: Availability

| Attribute | Value |
|-----------|-------|
| **Control Family** | Availability (A1) |
| **Objective** | Ensure system availability per SLAs |
| **Measurable Telemetry** | |
| - `uptime_pct` | Measured uptime percentage |
| - `benchmark_success_rate` | Success rate of benchmark runs |
| - `timeout_retry_configured` | Boolean: retry policies present |
| - `graceful_degradation_tested` | Boolean: degradation paths tested |
| - `health_check_enabled` | Boolean: health endpoints active |
| **Evidence Artifacts** | |
| - Availability report | `reports/compliance/<sha>/availability.json` |
| - Integration test results | Test logs showing graceful degradation |
| - Config snapshot | Timeout/retry configuration |
| **Gaps** | SLA monitoring requires external tooling |

### TSC-CONF: Confidentiality

| Attribute | Value |
|-----------|-------|
| **Control Family** | Confidentiality (C1) |
| **Objective** | Protect confidential information |
| **Measurable Telemetry** | |
| - `data_classification_applied` | Boolean: all datasets classified |
| - `encryption_at_rest` | Boolean: confidential data encrypted |
| - `encryption_in_transit` | Boolean: TLS enforced |
| - `access_logging_enabled` | Boolean: access to confidential data logged |
| **Evidence Artifacts** | |
| - Classification report | `reports/compliance/<sha>/classification.json` |
| - Encryption posture | From ISMS-CRYPTO evidence |
| - Access logs | Audit log filtered by confidential access |
| **Gaps** | DLP scanning not implemented |

### TSC-PI: Processing Integrity

| Attribute | Value |
|-----------|-------|
| **Control Family** | Processing Integrity (PI1) |
| **Objective** | Ensure complete, valid, accurate, and timely processing |
| **Measurable Telemetry** | |
| - `determinism_score` | Similarity of repeated runs on canonical inputs |
| - `dataset_hash` | Hash of training dataset |
| - `adapter_hash` | Hash of trained adapter |
| - `validation_passed` | Boolean: output validation checks passed |
| - `regression_test_passed` | Boolean: regression suite passed |
| **Evidence Artifacts** | |
| - Hash manifest | `reports/compliance/<sha>/hash_manifest.json` |
| - Regression results | `reports/compliance/<sha>/regression.json` |
| - Validation logs | Audit entries with `event_type=VALIDATION` |
| **Gaps** | Full input/output auditing not implemented |

### TSC-PRIV: Privacy

| Attribute | Value |
|-----------|-------|
| **Control Family** | Privacy (P1-P8) |
| **Objective** | Collect, use, retain, disclose personal information per policy |
| **Measurable Telemetry** | |
| - `pii_scan_dataset_count` | PII matches found in dataset sample |
| - `pii_scan_logs_count` | PII matches found in logs |
| - `pii_scan_artifacts_count` | PII matches found in artifact metadata |
| - `pii_redaction_enabled` | Boolean: log redaction active |
| - `privacy_policy_present` | Boolean: privacy policy documented |
| **Evidence Artifacts** | |
| - PII scan report | `reports/compliance/<sha>/pii_scan.json` |
| - Redaction test results | Test showing redaction effectiveness |
| - Privacy policy | `docs/PRIVACY_POLICY.md` |
| **Gaps** | Automated PII classification is heuristic-based |

---

## Metrics Summary

| Metric ID | Metric Name | Type | Source |
|-----------|-------------|------|--------|
| M-001 | `purpose_tags` | Array[String] | Dataset config |
| M-002 | `columns_dropped_pct` | Float | Preprocessing |
| M-003 | `examples_filtered_pct` | Float | Preprocessing |
| M-004 | `retention_policy_days` | Integer | Config |
| M-005 | `retention_enforced` | Boolean | Cleanup job |
| M-006 | `dp_epsilon` | Float | DP accountant |
| M-007 | `authn_method` | String | Auth config |
| M-008 | `authz_model` | String | Auth config |
| M-009 | `default_deny` | Boolean | Policy config |
| M-010 | `at_rest_encryption_enabled` | Boolean | Encryption config |
| M-011 | `in_transit_encryption_enabled` | Boolean | TLS config |
| M-012 | `kek_present` | Boolean | Key config |
| M-013 | `key_rotation_policy` | Boolean | Key config |
| M-014 | `audit_log_enabled` | Boolean | Log config |
| M-015 | `log_integrity_verified` | Boolean | Integrity check |
| M-016 | `event_coverage_pct` | Float | Coverage analysis |
| M-017 | `git_sha` | String | Git |
| M-018 | `dirty_tree` | Boolean | Git |
| M-019 | `dependency_lockfile_present` | Boolean | Filesystem |
| M-020 | `benchmark_success_rate` | Float | Benchmark runs |
| M-021 | `pii_scan_dataset_count` | Integer | PII scanner |
| M-022 | `pii_scan_logs_count` | Integer | PII scanner |
| M-023 | `secrets_exposed` | Integer | Secrets scanner |
| M-024 | `determinism_score` | Float | Regression test |
| M-025 | `dataset_hash` | String | Hash computation |
| M-026 | `adapter_hash` | String | Hash computation |

---

## Evidence Artifacts Index

| Artifact | Path | Format | Description |
|----------|------|--------|-------------|
| Metrics Bundle | `reports/compliance/<sha>/metrics.json` | JSON | All collected metrics |
| Evidence Report | `reports/compliance/<sha>/evidence.md` | Markdown | Human-readable report |
| Evidence Data | `reports/compliance/<sha>/evidence.json` | JSON | Structured evidence |
| PII Scan Report | `reports/compliance/<sha>/pii_scan.json` | JSON | PII detection counts |
| Secrets Scan | `reports/compliance/<sha>/secrets_scan.json` | JSON | Secrets detection |
| Auth Config | `reports/compliance/<sha>/auth_config.json` | JSON | Auth posture snapshot |
| Encryption Config | `reports/compliance/<sha>/encryption_config.json` | JSON | Encryption settings |
| Audit Integrity | `reports/compliance/<sha>/audit_integrity.json` | JSON | Log hash chain verification |
| Run Config | `reports/compliance/<sha>/run_config.json` | JSON | Reproducibility metadata |
| Hash Manifest | `reports/compliance/<sha>/hash_manifest.json` | JSON | Artifact hashes |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-29 | TensorGuard Team | Initial control matrix |
