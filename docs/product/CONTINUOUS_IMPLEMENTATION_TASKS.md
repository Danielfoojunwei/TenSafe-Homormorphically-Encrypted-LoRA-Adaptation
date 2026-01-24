# Continuous Learning Implementation Task Map

## Phase 1: Backend Data Model
- [ ] **Modify `src/tensorguard/platform/models/continuous_models.py`**
  - Add `CandidateEvent(SQLModel)` table for persistent timeline tracking.
  - Ensure `Route`, `Feed`, `Policy` match strict spec.
  - Link `AdapterLifecycleState` to `AdapterArtifact` (foreign key).
- [ ] **Create `src/tensorguard/platform/services/continuous_registry.py`**
  - Implement `ContinuousRegistryService` class.
  - Methods: `create_route`, `connect_feed`, `set_policy`, `record_event`, `register_candidate`, `promote`, `rollback`, `list_timeline`.

## Phase 2: Continuous Orchestrator
- [ ] **Modify `src/tensorguard/tgflow/continuous/orchestrator.py`**
  - Inject `ContinuousRegistryService`.
  - Refactor `run_once` to strictly follow 9-step flow (Ingest -> Novelty -> Train -> Eval -> Package -> Register -> Promote -> Caps).
  - Ensure `TRAIN` step calls real PEFT workflow (or strict simulation flag).
  - Use `CandidateEvent` for all step logging.

## Phase 3: N2HE Privacy Mode
- [ ] **Create `src/tensorguard/privacy/providers/n2he_provider.py`**
  - Implement `N2HEProvider` interface (encrypt_vector, decrypt_decision).
- [ ] **Create `src/tensorguard/privacy/safe_logger.py`**
  - Context manager `safe_log_context(privacy_mode)`.
- [ ] **Modify `src/tensorguard/tgflow/continuous/orchestrator.py`**
  - Add N2HE support in `resolve_route` and `package` steps.

## Phase 4: API Endpoints
- [ ] **Modify `src/tensorguard/platform/api/continuous_endpoints.py`**
  - Update to use `ContinuousRegistryService` and `ContinuousOrchestrator`.
  - Add `POST /tgflow/resolve` endpoint.
  - Add `GET /tgflow/routes/{key}/diff` endpoint.

## Phase 5: Frontend UX
- [ ] **Create/Update `frontend/src/components/core/ContinuousDashboard.vue`**
  - List active routes, risks, quick actions.
- [ ] **Create/Update `frontend/src/components/core/RouteTimeline.vue`**
  - Fetch from `CandidateEvent` table.
- [ ] **Create `frontend/src/components/core/ReleasesRollback.vue`**
  - Rollback UI, stable/canary visualization.
- [ ] **Create `frontend/src/components/core/EvidenceDiff.vue`**
  - Human-readable diff of two adapters.
- [ ] **Modify `frontend/src/router/index.js`**
  - Set Dashboard as default landing.

## Phase 6: Export & Portability
- [ ] **Modify `src/tensorguard/platform/api/continuous_endpoints.py`**
  - Implement `export` endpoint returning JSON/YAML templates.
- [ ] **Update Frontend**
  - Add "Export Spec" button to Timeline.

## Phase 7: Verification
- [ ] **Run Integration Test**
  - `tests/integration/test_route_run_once_end_to_end.py`
