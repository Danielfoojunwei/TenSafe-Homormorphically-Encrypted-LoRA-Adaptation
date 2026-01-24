# UX Verification Checklist

**Goal:** Ensure the core "Happy Path" is intuitive and completes in < 30 minutes.

## general
- [ ] No "orphan" screens without back buttons.
- [ ] Loading states shown for async operations (>1s).
- [ ] Error messages are actionable (not "Error 500").

## Dashboard
- [ ] "Create Route" is the primary call-to-action.
- [ ] Active routes show status (Healthy/At Risk) clearly.
- [ ] Risk indicators (red/yellow) are explained via tooltips.

## Route Wizard
- [ ] Steps are clearly numbered.
- [ ] Validation prevents "Next" if required fields missing.
- [ ] Guided Mode is default.
- [ ] Privacy Mode toggle explains implication ("Encrypted Routing").

## Timeline
- [ ] Events ordered newest-first.
- [ ] Verdicts (PASS/FAIL) have distinct icons/colors.
- [ ] Clicking an event shows details (Metrics/Logs).

## Releases
- [ ] Active Stable vs Canary clearly distinguished.
- [ ] Rollback button is red/danger-styled.
- [ ] Promotion confirmation dialog appears.

## Evidence
- [ ] Diff view highlights "What changed" (not just raw JSON).
- [ ] Signature status (Verified) is visible.
