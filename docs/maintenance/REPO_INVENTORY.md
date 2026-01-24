# Repository Inventory Report

**Date:** 2026-01-18
**Status:** In Progress (Phase 0)

## Top-Level Structure
- Total Size: ~500 MB (excluding cached environments)
- Key Components: `src/tensorguard`, `frontend`, `docs`
- Bloat Candidates: `data/`, `demo_*`, top-level `.zip` backups

## Large Files (>10MB)
| File | Size | Recommendation |
|------|------|----------------|
| `TensorGuardFlow_Production_Ready.zip` | 124 MB | **DELETE** (Redundant) |
| `TensorGuardFlow_Hardened.zip` | 124 MB | **DELETE** (Redundant) |
| `data/fastumi_pro/.../video.mp4` | 76 MB | **DELETE** (Host externally) |
| `data/synthetic/video.mp4` | 44 MB | **DELETE** (Host externally) |

## Unwanted Directories (Non-Core / Generated)
| Directory | Content | Action |
|-----------|---------|--------|
| `runs/` | Training logs (presumed) | Add to .gitignore |
| `artifacts/` | Benchmark outputs | Add to .gitignore |
| `demo_federation/` | Old R&D demos | Move to `src/tensorguard/labs` or delete |
| `demo_fl_trust/` | Old R&D demos | Move to `src/tensorguard/labs` or delete |
| `demo_integration/` | Old R&D demos | Move to `src/tensorguard/labs` or delete |
| `data/` | Heavy binary datasets | Replace with script/docs |
| `tmp_keys/` | Temporary artifacts | Add to .gitignore |

## Dead Code Candidates
### Python Modules
- `src/tensorguard/moai`: ** STRICTLY UNWANTED IN CORE**.
- `src/tensorguard/bench`: Likely research benchmarks, not production core.
- `src/tensorguard/edge_agent`: Robotics specific?
- `src/tensorguard/enablement`: Platform service, check if used in Core.

### Frontend Components
- `frontend/src/components` contains 26 files.
- `Router` only likely uses `ContinuousDashboard`, `RouteDetails`, `RouteTimeline`, `ReleasesRollback`, `EvidenceDiff`.
- `EvalArena`, `FleetsDevices`, `ForensicsPanel`, `IntegrationsHub` etc. seem legacy/unused in Continuous flow.

## Next Steps
1. Delete Zip files & Videos.
2. Migration non-core modules to `labs`.
3. Purge MOAI from shipping paths.
