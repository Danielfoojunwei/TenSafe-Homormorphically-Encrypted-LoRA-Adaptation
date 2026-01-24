# Bug Report Template

**Title:** [Area] Short description of the issue

**Environment:**
- **Build/Commit:** [e.g. `abc1234`]
- **Mode:** [Local/CI/Staging]
- **Privacy Mode:** [On/Off]

**Severity:** [S0-S4] (See rubric below)

**Steps to Reproduce:**
1. Go to ...
2. Click on ...
3. Run command ...

**Expected Result:**
[What should happen]

**Actual Result:**
[What actually happened]

**Evidence:**
- [Logs]
- [Screenshots]
- [Response JSON]

---

## Severity Rubric
- **S0 (Critical)**: Data loss, security breach, signature bypass, plaintext leak. **BLOCKS RELEASE**.
- **S1 (Major)**: Core journey broken (Create/Promote/Resolve fails). **BLOCKS RELEASE**.
- **S2 (Medium)**: Incorrect gates, wrong logic, broken secondary flows (Diff/Export).
- **S3 (Minor)**: UX friction, confusing copy, non-blocking errors.
- **S4 (Cosmetic)**: typo, spacing, color.
