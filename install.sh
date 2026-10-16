#!/usr/bin/env bash
# Astra Backend Installer
# Usage: ./install.sh [--voice]

set -euo pipefail

VOICE=false
if [[ "${1:-}" == "--voice" ]]; then
    VOICE=true
fi

PYTHON="${PYTHON:-python3}"
MIN_PY="3.11"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[astra]${NC} $*"; }
warn() { echo -e "${YELLOW}[astra]${NC} $*"; }
err() { echo -e "${RED}[astra]${NC} $*" >&2; }

# --- Python version check ---
log "Checking Python version..."
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
if ! $PYTHON -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    err "Python $MIN_PY+ is required, found $PY_VERSION"
    exit 1
fi
log "Found Python $PY_VERSION"

# --- Create venv ---
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment in $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
else
    warn "Virtual environment already exists, reusing."
fi

# --- Activate venv ---
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
log "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- Install core deps ---
log "Installing core requirements..."
pip install -r requirements.txt

# --- Optional voice deps ---
if $VOICE; then
    log "Installing voice/WebRTC requirements (this may take a while)..."
    pip install -r requirements-voice.txt
fi

# --- Create config files from examples ---
if [[ ! -f "config/config.toml" ]]; then
    log "Creating config/config.toml from example..."
    cp config/config.example.toml config/config.toml
else
    warn "config/config.toml already exists, not overwriting."
fi

if [[ ! -f ".env" ]]; then
    log "Creating .env from example..."
    cp .env.example .env
    warn "Edit .env and add your API key before running the server."
else
    warn ".env already exists, not overwriting."
fi

# --- Verify install ---
log "Running backend tests..."
python -m pytest tests -q

log "Install complete."
echo
log "Next steps:"
echo "  1. Edit .env with your LLM API key"
echo "  2. Review config/config.toml"
if $VOICE; then
    echo "  3. Run: python api/server.py"
    echo "     (voice/WebRTC deps installed)"
else
    echo "  3. Run: python api/server.py"
    echo "     (for voice calls, re-run: ./install.sh --voice)"
fi
