# Production Audit & Implementation Plan

## Objective
Deliver a production-ready TensorGuardFlow by eliminating all mock/simulated behaviors in production code paths and aligning the UI with the real backend capabilities and data shapes.

## Audit Summary (Production Impact)
The following components currently contain mock, simulated, placeholder, or stub behaviors that must be replaced or fully gated in production:

### Backend: Runtime/Platform
- **RMF integration uses mock ciphertext and mock keys** in `RmfAdapter.process_task_request`, which currently builds a fake encrypted payload instead of encrypting real robot telemetry. This prevents end-to-end MOAI inference from being real and verifiable. 【F:src/tensorguard/integrations/rmf/adapter.py†L18-L69】
- **PEFT DemoTrainer produces placeholder artifacts and mock metrics**, even though the PEFT workflow is expected to be a real training pipeline. The demo path is explicitly marked as non-production, but needs a production replacement and clear UI handling. 【F:src/tensorguard/integrations/peft_hub/connectors/training_hf.py†L268-L377】
- **TPM simulator is used for identity attestation** with a production gate override; the simulator explicitly marks attestations as untrusted. Production requires hardware-backed attestation or an external attestation service. 【F:src/tensorguard/agent/identity/tpm_simulator.py†L1-L135】
- **Identity agent work polling simulates ACME challenges and deployment** (HTTP-01 is a no-op, deployment is only acknowledged). This blocks real certificate issuance and lifecycle management. 【F:src/tensorguard/agent/identity/work_poller.py†L63-L145】
- **Private CA SPIRE issuance is a stub** and scheduler blocks private CA issuance in production when public trust is disabled. This leaves a production gap for non-public CA workflows. 【F:src/tensorguard/identity/ca/private_ca.py†L170-L216】【F:src/tensorguard/identity/scheduler.py†L382-L440】
- **ML Manager adapter loading and hot-swap/rollback are stubs**; the system does not actually load adapters or validate compatibility, which makes deployments non-functional. 【F:src/tensorguard/agent/ml/manager.py†L42-L229】
- **TrainingWorker uses simplified DP accounting and simulated pruning behavior**, logging that pruning is skipped on mock models and explicitly stating the DP accounting is a placeholder. This is a production safety and compliance gap. 【F:src/tensorguard/agent/ml/worker.py†L115-L238】
- **Observability uses a no-op tracer stub when OTEL is unavailable**, which means traces are silently dropped without a production-grade export path. 【F:src/tensorguard/observability/otel.py†L180-L249】
- **MOAI key loading returns placeholder evaluation keys** if `.eval` artifacts are missing. This can silently bypass required key material. 【F:src/tensorguard/moai/keys.py†L85-L118】

### Frontend: UI Alignment Gaps
- **TGSP Marketplace shows mock package data on backend failure**, which hides real ingestion/verification states and breaks production UX. UI should reflect real registry data and failures. 【F:frontend/src/components/TGSPMarketplace.vue†L1-L196】
- **Identity Manager falls back to mock inventory, policies, and renewals**, masking real inventory response shapes from `/api/v1/identity/*`. 【F:frontend/src/components/IdentityManager.vue†L1-L128】
- **NodePalette defines a static node catalog**, which may not reflect the backend-supported pipeline/actions if the system exposes dynamic capabilities. 【F:frontend/src/components/flow/NodePalette.vue†L1-L124】
- **Identity API inventory endpoint provides the canonical response shape** (endpoints, certificates, expiry summary) that the UI should follow without mock fallback. 【F:src/tensorguard/platform/api/identity_endpoints.py†L126-L206】
- **TGSP community endpoints provide production package data and workflow** that should drive the TGSP Marketplace UI without demo data. 【F:src/tensorguard/platform/api/community_tgsp.py†L1-L116】

## Implementation Plan (Production-Ready)

### Phase 1: Replace Mock/Simulated Backend Behaviors
1. **RMF → MOAI real payload path**
   - Build a real RMF payload adapter that ingests robot telemetry or task state, uses actual MOAI encryption (N2HE/CKKS), and signs/envelopes payloads before calling `/v1/infer`.
   - Wire key management to the MOAI key vault and require real eval keys (no placeholder fallback).【F:src/tensorguard/integrations/rmf/adapter.py†L18-L69】【F:src/tensorguard/moai/keys.py†L85-L118】
   - Add explicit validation for payload schema and metadata to prevent malformed tasks reaching MOAI.

2. **PEFT training pipeline hardening**
   - Ensure the PEFT workflow only starts when production dependencies are installed; remove any UI path that relies on DemoTrainer.
   - Extend the Hugging Face connector to validate datasets, model availability, and output artifacts in a persistent store (e.g., object storage).【F:src/tensorguard/integrations/peft_hub/connectors/training_hf.py†L268-L377】

3. **Identity attestation and CA issuance**
   - Replace TPM simulator usage with hardware-backed attestation (tpm2-tss/TPM2 tools) or integrate cloud attestation (e.g., Azure/AMD SEV/SNP), and ensure the agent refuses simulator paths in production without explicit research mode. 【F:src/tensorguard/agent/identity/tpm_simulator.py†L1-L135】
   - Implement real ACME HTTP-01 challenge handling and deployment steps in the agent (write challenge tokens to webroot/ingress, deploy certificates to target endpoints).【F:src/tensorguard/agent/identity/work_poller.py†L63-L145】
   - Finish Private CA integrations (Vault, step-ca, SPIRE) so the scheduler can issue private certificates in production rather than failing. 【F:src/tensorguard/identity/ca/private_ca.py†L170-L216】【F:src/tensorguard/identity/scheduler.py†L382-L440】

4. **ML adapter lifecycle and DP accounting**
   - Replace MLManager adapter stubs with real adapter registry + loader (download, validation, atomic swap). Provide compatibility checks and rollback safety. 【F:src/tensorguard/agent/ml/manager.py†L42-L229】
   - Implement production DP accounting (RDP accountant or Opacus-based per-sample clipping) and enforce privacy budgets; remove placeholder epsilon estimation. 【F:src/tensorguard/agent/ml/worker.py†L115-L185】
   - Replace simulated pruning with real model-layer pruning hooks and guard by model availability. 【F:src/tensorguard/agent/ml/worker.py†L205-L238】

5. **Observability**
   - Require OTLP exporter configuration in production; fail closed or emit explicit warnings when observability is disabled to avoid silent drops. 【F:src/tensorguard/observability/otel.py†L180-L249】

### Phase 2: UI Alignment With Real Backend Capabilities
1. **TGSP Marketplace UI**
   - Remove mock package fallback and show explicit error states if `/api/v1/tgsp/packages` is unavailable.
   - Surface real package status transitions (uploaded → verified/rejected) and allow retry via backend events. 【F:frontend/src/components/TGSPMarketplace.vue†L1-L196】【F:src/tensorguard/platform/api/community_tgsp.py†L1-L116】

2. **Identity Manager UI**
   - Remove mock inventory/policy/renewal data; bind strictly to `/api/v1/identity/inventory`, `/policies`, `/renewals` response shapes.
   - Map certificate fields to the real API contract (`subject`, `issuer`, `not_after`, `days_to_expiry`). 【F:frontend/src/components/IdentityManager.vue†L1-L128】【F:src/tensorguard/platform/api/identity_endpoints.py†L126-L206】

3. **Flow Node Palette**
   - Replace static node catalog with a backend-driven list of supported triggers/actions (e.g., `/api/v1/flow/nodes`), to avoid diverging UX from actual pipeline capabilities. 【F:frontend/src/components/flow/NodePalette.vue†L1-L124】

### Phase 3: Production Safety Gates & Validation
- Enforce fail-closed behavior for missing secrets (JWT, PQC keys, MOAI eval keys) and reject demo/simulator code paths in production environments. 【F:src/tensorguard/platform/auth.py†L30-L83】【F:src/tensorguard/moai/keys.py†L85-L118】
- Introduce a “production readiness” check at startup that validates dependencies and removes demo access from the UI (e.g., hide demo-only navigation or features unless `TG_ENVIRONMENT=development`).

## Deliverables
- **Backend**: Real RMF payload processing, production PEFT training, fully functional identity issuance, ML adapter management with DP compliance, and enforced observability.
- **Frontend**: UI reflects real data from the TGSP registry and identity APIs; flow nodes are sourced from backend-supported capabilities.
- **Documentation**: Updated production guide listing required env vars, dependencies, and supported integration matrices.

## Rollout Order (Recommended)
1. Identity issuance + attestation
2. ML adapter lifecycle + DP accounting
3. RMF/MOAI integration
4. TGSP Marketplace alignment
5. Observability enforcement
6. Flow node capability catalog

## Success Criteria
- No mock/simulated data shown in production UI.
- Production endpoints return real, persisted records only.
- Identity issuance and attestation validated with real infrastructure.
- PEFT training runs end-to-end with stored artifacts and verifiable metrics.
- Observability exports traces to OTLP without silent no-op fallbacks.
