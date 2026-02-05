#!/bin/bash
# CI Smoke Test Script
# Verifies that the package can be installed and imported correctly
# Run this before committing changes to packaging configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "TenSafe CI Smoke Test"
echo "========================================"
echo ""

# Create temporary directory for venv
VENV_DIR=$(mktemp -d)
trap "rm -rf $VENV_DIR" EXIT

echo "Step 1: Creating clean virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Step 2: Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q

echo "Step 3: Installing package from source..."
cd "$REPO_ROOT"
pip install . -q

echo "Step 4: Verifying imports (from /tmp to avoid CWD in path)..."
cd /tmp

# Test core package imports
python -c "
import sys
print(f'Python: {sys.version}')

# Test tensafe
import tensafe
print(f'tensafe: OK ({tensafe.__file__})')

# Test tensorguard
import tensorguard
print(f'tensorguard: OK ({tensorguard.__file__})')

# Test tg_tinker
import tg_tinker
print(f'tg_tinker: OK ({tg_tinker.__file__})')

# Test main entry point
from tensorguard.platform.main import app
print('tensorguard.platform.main.app: OK')

# Test TG-Tinker routes
from tensorguard.platform.tg_tinker_api.routes import router
print('tensorguard.platform.tg_tinker_api.routes: OK')

# Test TG-Tinker client
from tg_tinker.client import TinkerClient
print('tg_tinker.client.TinkerClient: OK')

# Test tensafe core
from tensafe.core.orchestrator import TenSafeOrchestrator
print('tensafe.core.orchestrator: OK')

print('')
print('All imports successful!')
"

echo ""
echo "Step 5: Running packaging smoke test..."
cd "$REPO_ROOT"
pip install pytest -q
python -m pytest tests/test_packaging_imports.py -v --tb=short

echo ""
echo "========================================"
echo "CI Smoke Test: PASSED"
echo "========================================"
