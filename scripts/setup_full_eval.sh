#!/bin/bash
# =============================================================================
# TenSafe Full Empirical Validation Setup
# =============================================================================
# This script prepares a machine for complete, real empirical evaluation.
# Run on a GPU instance (A100/H100 recommended) with Ubuntu 22.04+
#
# Usage:
#   # Phase 1: Install everything
#   bash scripts/setup_full_eval.sh install
#
#   # Phase 2: Login to HuggingFace (interactive - needs Meta Llama access)
#   bash scripts/setup_full_eval.sh login
#
#   # Phase 3: Run full evaluation
#   bash scripts/setup_full_eval.sh run
#
#   # All-in-one (non-interactive, skip HF login)
#   bash scripts/setup_full_eval.sh install run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# Dependency checks
# =============================================================================
check_gpu() {
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        log "GPU detected: $GPU_NAME ($GPU_MEM)"
        return 0
    else
        warn "No GPU detected. CPU-only benchmarks will run."
        warn "For full validation, use an NVIDIA A100/H100 instance."
        return 1
    fi
}

check_cuda() {
    python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null
}

# =============================================================================
# Install phase
# =============================================================================
install_system_deps() {
    log "Installing system dependencies..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq 2>/dev/null || true
        apt-get install -y -qq cmake ninja-build g++ git curl 2>/dev/null || true
    fi
}

install_python_ml() {
    log "Installing Python ML stack..."

    if check_gpu && ! check_cuda; then
        log "GPU found but no CUDA PyTorch. Installing CUDA version..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
    elif ! python3 -c "import torch" 2>/dev/null; then
        log "No GPU, installing CPU PyTorch..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
    else
        log "PyTorch already installed."
    fi

    pip install transformers peft accelerate datasets bitsandbytes 2>&1 | tail -3
    log "ML stack installed."
}

install_vllm() {
    if check_gpu; then
        log "Installing vLLM (GPU detected)..."
        pip install vllm 2>&1 | tail -3
        log "vLLM installed."
    else
        warn "Skipping vLLM (requires GPU)."
    fi
}

install_tenseal() {
    log "Installing TenSEAL (real CKKS FHE)..."
    pip install tenseal 2>&1 | tail -3

    python3 -c "
import tenseal as ts
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2**40
enc = ts.ckks_vector(ctx, [1.0, 2.0, 3.0])
dec = enc.decrypt()
print(f'TenSEAL CKKS verified: max_error={max(abs(a-b) for a,b in zip([1,2,3], dec)):.2e}')
" 2>&1
    log "TenSEAL installed and verified."
}

install_liboqs() {
    log "Installing liboqs (post-quantum crypto)..."

    if python3 -c "import oqs" 2>/dev/null; then
        log "liboqs already installed."
        return 0
    fi

    LIBOQS_DIR="/tmp/liboqs"
    if [ ! -d "$LIBOQS_DIR" ]; then
        git clone --depth 1 https://github.com/open-quantum-safe/liboqs.git "$LIBOQS_DIR" 2>&1 | tail -3
    fi

    cd "$LIBOQS_DIR"
    mkdir -p build && cd build
    cmake -GNinja -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local .. 2>&1 | tail -3
    ninja 2>&1 | tail -3
    ninja install 2>&1 | tail -3
    ldconfig

    pip install liboqs-python 2>&1 | tail -3

    python3 -c "
import oqs
sig = oqs.Signature('ML-DSA-65')
pub = sig.generate_keypair()
msg = b'TenSafe PQC test'
signature = sig.sign(msg)
assert sig.verify(msg, signature, pub)
print(f'ML-DSA-65 (Dilithium3) verified: sig_size={len(signature)} bytes')
" 2>&1

    cd "$PROJECT_ROOT"
    log "liboqs installed and verified."
}

install_project() {
    log "Installing TenSafe project..."
    cd "$PROJECT_ROOT"
    pip install -e ".[dev,bench]" 2>&1 | tail -3
    log "TenSafe installed."
}

do_install() {
    log "=== Starting full installation ==="
    install_system_deps
    install_python_ml
    install_tenseal
    install_liboqs
    install_vllm
    install_project

    log ""
    log "=== Installation Summary ==="
    python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
except: print('PyTorch: NOT INSTALLED')

try:
    import transformers; print(f'Transformers: {transformers.__version__}')
except: print('Transformers: NOT INSTALLED')

try:
    import peft; print(f'PEFT: {peft.__version__}')
except: print('PEFT: NOT INSTALLED')

try:
    import tenseal; print(f'TenSEAL: {tenseal.__version__} (REAL CKKS FHE)')
except: print('TenSEAL: NOT INSTALLED')

try:
    import oqs; print(f'liboqs: {oqs.oqs_version()} (REAL PQC)')
except: print('liboqs: NOT INSTALLED')

try:
    import vllm; print(f'vLLM: {vllm.__version__}')
except: print('vLLM: NOT INSTALLED (needs GPU)')
" 2>&1
    log "=== Installation complete ==="
}

# =============================================================================
# HuggingFace login
# =============================================================================
do_login() {
    log "=== HuggingFace Login ==="
    log "You need a HuggingFace token with read access."
    log "1. Go to https://huggingface.co/settings/tokens"
    log "2. Create a token with 'read' permissions"
    log "3. Accept Meta Llama license at https://huggingface.co/meta-llama/Meta-Llama-3-8B"
    log ""
    huggingface-cli login
}

# =============================================================================
# Run evaluation
# =============================================================================
do_run() {
    log "=== Running Full Empirical Evaluation ==="
    cd "$PROJECT_ROOT"

    export PYTHONPATH="src:${PYTHONPATH:-}"
    export TENSAFE_ENV=local
    export TENSAFE_TOY_HE=1

    log "Running evaluate_real.py..."
    python3 scripts/evaluate_real.py 2>&1

    log ""
    log "=== Evaluation Complete ==="
    log "Reports saved to: reports/real_evaluation/"
}

# =============================================================================
# Main
# =============================================================================
if [ $# -eq 0 ]; then
    echo "Usage: $0 {install|login|run}"
    echo ""
    echo "  install  - Install all dependencies"
    echo "  login    - Login to HuggingFace (for Llama-3-8B access)"
    echo "  run      - Run full evaluation suite"
    echo ""
    echo "Example: $0 install login run"
    exit 0
fi

for cmd in "$@"; do
    case "$cmd" in
        install) do_install ;;
        login)   do_login ;;
        run)     do_run ;;
        *)       err "Unknown command: $cmd"; exit 1 ;;
    esac
done
