# Release Gates (Certification Criteria)

**Version:** 2.3.0 Core
**Status:** REQUIRED

## P0: Critical Path (Must Pass 100%)
- [x] **Happy Path E2E**: Create -> Feed -> Policy -> Run -> Promote -> Resolve.
- [x] **Data Integrity**: No data loss during loop.
- [x] **Security**: N2HE Privacy Mode leaks ZERO plaintext.
- [x] **Rollback**: Successfully reverts to previous stable.
- [x] **Isolation**: Tenant A cannot access Tenant B resources.

## P1: Feature Completeness (Must Pass 90%)
- [x] **Export**: Configs export to valid K8s/Cloud specs.
- [x] **Diff**: Accurately reports changes.
- [x] **Timeline**: History accurately reflects all state changes.

## P2: Quality of Life (Best Effort)
- [ ] **UX**: < 3 clicks to reach any core screen.
- [ ] **Perf**: Dashboard loads < 2s with 50 routes.

## Blocker Definition
- Any **S0** or **S1** bug is an automatic **NO GO**.
- Any **Privacy Leak** is an automatic **NO GO**.
