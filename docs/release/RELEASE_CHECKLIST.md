# Release Checklist

## Version: X.Y.Z

**Release Manager**: ___________
**Release Date**: ___________

## Pre-Release Checks

### 1. Code Quality (All Must Pass)
- [ ] `make lint` - No ruff errors
- [ ] `make typecheck` - No pyright errors
- [ ] `make test` - All tests pass
- [ ] `make test-security` - Security tests pass
- [ ] Coverage ≥ 80% on changed files

### 2. Security Audit
- [ ] No new dependencies with known vulnerabilities (`pip-audit`)
- [ ] Secrets scan clean (`make compliance-smoke`)
- [ ] Crypto hygiene tests pass (`pytest tests/security/test_crypto_tamper.py`)
- [ ] Error handling tests pass (`pytest tests/security/test_error_handling.py`)

### 3. Documentation
- [ ] README.md up to date
- [ ] CHANGELOG.md updated with release notes
- [ ] docs/MATURITY.md reflects current status
- [ ] API_SPEC.md matches implementation

### 4. Version Bumps
- [ ] `pyproject.toml` version updated
- [ ] `src/tensorguard/platform/main.py` version updated
- [ ] Git tag created: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`

### 5. Final Verification
- [ ] Clean install works: `pip install -e .`
- [ ] Server boots: `python -m tensorguard.platform.main`
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] SDK imports work: `python -c "from tg_tinker import ServiceClient"`

## Release Process

### 1. Prepare Release Branch
```bash
git checkout main
git pull origin main
git checkout -b release/vX.Y.Z
```

### 2. Run Pre-Release Verification
```bash
./scripts/release/pre_release_check.py --version X.Y.Z
```

### 3. Update Changelog
Add entry to CHANGELOG.md:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Security
- ...
```

### 4. Create Release
```bash
# Commit version changes
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release vX.Y.Z"

# Merge to main
git checkout main
git merge release/vX.Y.Z

# Tag release
git tag -a vX.Y.Z -m "Release vX.Y.Z"

# Push
git push origin main --tags
```

### 5. Post-Release
- [ ] GitHub release created with release notes
- [ ] PyPI package published (if applicable)
- [ ] Announcement sent (if applicable)

## Rollback Procedure

If issues are found after release:

1. **Revert the release commit**:
   ```bash
   git revert HEAD
   git push origin main
   ```

2. **Delete the tag** (if not yet pushed to PyPI):
   ```bash
   git tag -d vX.Y.Z
   git push origin :refs/tags/vX.Y.Z
   ```

3. **Create hotfix branch**:
   ```bash
   git checkout -b hotfix/vX.Y.Z+1
   ```

## Emergency Security Release

For critical security issues:

1. Create hotfix branch from the affected release tag
2. Apply minimal fix only
3. Run security tests
4. Release with patch version bump (X.Y.Z+1)
5. Backport to affected supported versions
