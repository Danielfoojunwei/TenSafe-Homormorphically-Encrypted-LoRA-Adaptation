# TensorGuardFlow Cleanup & Hardening Report

**Date:** 2026-01-18
**Status:** Completed
**Scope:** Core Hardening (Continuous Control Plane)

## 1. Scope Reduction & Organization

We have successfully reorganized the codebase to strictly separate **Shipping Core** from **R&D Labs**.

### Core vs. Labs Separation
| Core (Shipping) | Labs (Non-Shipping) |
|-----------------|---------------------|
| `src/tensorguard/platform` | `src/tensorguard/labs/demo_*` |
| `src/tensorguard/tgflow` | `src/tensorguard/labs/moai` (MOAI Removed from core) |
| `src/tensorguard/tgsp` | `src/tensorguard/labs/edge_agent` |
| `src/tensorguard/privacy` | `src/tensorguard/labs/enablement` |
| `src/tensorguard/evidence` | `src/tensorguard/labs/bench` |

### API Hardening
- **Deactivated Routers**: The following experimental endpoints have been unmounted from `platform/main.py`:
  - `enablement` (Trust Console)
  - `fedmoe` (Expert Routing)
  - `vla` (Robotics)
  - `community_tgsp`
- **Active Routers**:
  - `continuous-learning` (Core Loop)
  - `runs` (Evidence)
  - `settings`
  - `identity` (Auth)

## 2. Artifact Cleanup

Moved ~400MB of binary large objects out of the repository.
- **Deleted**: `TensorGuardFlow_Production_Ready.zip` (124MB), `TensorGuardFlow_Hardened.zip` (124MB).
- **Deleted**: `data/.../video.mp4` x2 (120MB).
- **Updates**: `.gitignore` updated to block `*.zip`, `*.mp4`, `runs/`, `outputs/`.

## 3. Frontend Sanitation

The Frontend has been simplified to focus *only* on the Continuous Learning Control Plane.
- **Rewritten**: `App.vue` now uses `router-view` with a minimal Sidebar.
- **Core Components**: `ContinuousDashboard`, `RouteDetails`, `RouteTimeline`, `ReleasesRollback`, `EvidenceDiff`.
- **Legacy Components**: Moved to `frontend/src/components/labs/`.
- **Navigation**: Sidebar only links to Dashboard.

## 4. Simulation & Determinism

- **Explicit Simulation**: Enforced `TG_SIMULATION=true` check in `orchestrator.py` for mock ingestion.
- **Deterministic Loop**: Created `scripts/smoke_test_core.py` which verifies the complete 9-step loop (Ingest->Novelty->Train->Eval->TGSP->Evidence->Register) in strict determinism mode without external dependencies.
- **Bug Fixes**: Fixed async generator syntax errors in `PeftWorkflow` to ensure robust error handling.

## 5. Documentation

- **Consolidated**: `docs/product/GET_STARTED_CONTINUOUS.md` is the single source of truth for onboarding.
- **Archived**: Moved `USER_GUIDE.md`, `DEPLOYMENT.md` (check this), etc. to `docs/archive/`.

## Verification
- **Smoke Test**: Passed. The core loop functions end-to-end in simulation mode.
- **Build Cleanliness**: No circular dependencies found between Core and Labs.
