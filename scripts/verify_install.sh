#!/usr/bin/env bash
#
# verify_install.sh - Verify TenSafe installation in a clean environment
#
# Usage: ./scripts/verify_install.sh [--with-server] [--with-pqc] [--with-dev]
#
# This script creates a fresh virtual environment and verifies that:
# 1. requirements.txt installs successfully
# 2. pip install -e . succeeds
# 3. pytest collection completes without import errors
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="/tmp/tensafe_verify_$$"

# Parse arguments
EXTRAS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --with-server)
            EXTRAS="${EXTRAS},server"
            shift
            ;;
        --with-pqc)
            EXTRAS="${EXTRAS},pqc"
            shift
            ;;
        --with-dev)
            EXTRAS="${EXTRAS},dev"
            shift
            ;;
        --with-all)
            EXTRAS=",all"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Remove leading comma from EXTRAS
EXTRAS="${EXTRAS#,}"

echo "=================================================="
echo "TenSafe Installation Verification"
echo "=================================================="
echo ""
echo "Repository: $REPO_ROOT"
echo "Venv: $VENV_DIR"
echo "Extras: ${EXTRAS:-none}"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    rm -rf "$VENV_DIR"
}
trap cleanup EXIT

# Step 1: Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Python version: $(python --version)"

# Step 2: Upgrade pip
echo ""
echo "[2/5] Upgrading pip, setuptools, wheel..."
pip install -U pip setuptools wheel -q

# Step 3: Install requirements.txt
echo ""
echo "[3/5] Installing requirements.txt..."
if pip install -r "$REPO_ROOT/requirements.txt" -q; then
    echo "  ✓ requirements.txt installed successfully"
else
    echo "  ✗ requirements.txt installation FAILED"
    exit 1
fi

# Step 4: Install package in editable mode
echo ""
echo "[4/5] Installing package (pip install -e .)..."
if [ -n "$EXTRAS" ]; then
    INSTALL_CMD="pip install -e \"$REPO_ROOT[$EXTRAS]\" -q"
else
    INSTALL_CMD="pip install -e \"$REPO_ROOT\" -q"
fi

if eval "$INSTALL_CMD"; then
    echo "  ✓ Package installed successfully"
else
    echo "  ✗ Package installation FAILED"
    exit 1
fi

# Step 5: Run pytest collection
echo ""
echo "[5/5] Running pytest collection..."
cd "$REPO_ROOT"

# Install pytest if not already installed
pip install pytest pytest-asyncio -q

# Run pytest in collection-only mode
if pytest --collect-only -q 2>&1 | tee /tmp/pytest_collect_$$.log | tail -20; then
    ERRORS=$(grep -c "ERROR" /tmp/pytest_collect_$$.log || true)
    if [ "$ERRORS" -gt 0 ]; then
        echo ""
        echo "  ⚠ pytest collection completed with $ERRORS errors"
        echo "  See full output above for details"
        exit 1
    else
        echo ""
        echo "  ✓ pytest collection completed successfully"
    fi
else
    echo ""
    echo "  ✗ pytest collection FAILED"
    exit 1
fi

echo ""
echo "=================================================="
echo "All verification steps passed!"
echo "=================================================="
