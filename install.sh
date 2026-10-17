#!/usr/bin/env bash
# Astra Backend Installer
# Usage: ./install.sh [--voice] [--system-deps] [--skip-config] [--test]

set -euo pipefail

VOICE=false
SYSTEM_DEPS=false
SKIP_CONFIG=false
RUN_TESTS=false
for arg in "$@"; do
    case "$arg" in
        --voice) VOICE=true ;;
        --system-deps) SYSTEM_DEPS=true ;;
        --skip-config) SKIP_CONFIG=true ;;
        --test) RUN_TESTS=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python3}"
MIN_PY="3.11"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[astra]${NC} $*"; }
warn() { echo -e "${YELLOW}[astra]${NC} $*"; }
err() { echo -e "${RED}[astra]${NC} $*" >&2; }

if $RUN_TESTS; then
    if [[ ! -d ".venv" ]]; then
        err "Virtual environment not found. Run ./install.sh first."
        exit 1
    fi
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
    log "Running tests..."
    python -m pytest tests -q
    exit 0
fi

# --- Install system dependencies ---
install_system_deps() {
    if command -v apt-get >/dev/null 2>&1; then
        log "Installing system dependencies with apt..."
        sudo apt-get update
        sudo apt-get install -y \
            python3 python3-venv python3-dev python3-pip \
            build-essential \
            libopus-dev \
            portaudio19-dev \
            libffi-dev \
            libssl-dev
    elif command -v dnf >/dev/null 2>&1; then
        log "Installing system dependencies with dnf..."
        sudo dnf install -y \
            python3 python3-devel python3-pip \
            gcc gcc-c++ make \
            opus-devel \
            portaudio-devel \
            libffi-devel \
            openssl-devel
    elif command -v pacman >/dev/null 2>&1; then
        log "Installing system dependencies with pacman..."
        sudo pacman -Sy --needed \
            python python-pip \
            base-devel \
            opus \
            portaudio \
            libffi \
            openssl
    elif command -v brew >/dev/null 2>&1; then
        log "Installing system dependencies with brew..."
        brew install python opus portaudio libffi openssl
    else
        err "Could not detect a supported package manager."
        err "Install Python $MIN_PY+, a C compiler, and opus/portaudio headers manually."
        exit 1
    fi
}

if $SYSTEM_DEPS; then
    install_system_deps
fi

# --- Python check / install ---
log "Checking Python..."
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    if $SYSTEM_DEPS; then
        log "Python not found; system deps were installed, trying again..."
        PYTHON="python3"
        if ! command -v "$PYTHON" >/dev/null 2>&1; then
            err "Python is still not available after installing system deps."
            err "Please install Python $MIN_PY+ manually and re-run."
            exit 1
        fi
    else
        err "Python is not installed. Re-run with --system-deps to install it automatically,"
        err "or install Python $MIN_PY+ manually."
        exit 1
    fi
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
if ! $PYTHON -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    err "Python $MIN_PY+ is required, found $PY_VERSION"
    if ! $SYSTEM_DEPS; then
        err "Re-run with --system-deps to attempt an upgrade."
    fi
    exit 1
fi
log "Found Python $PY_VERSION"

# --- Ensure venv module is available ---
if ! $PYTHON -m venv --help >/dev/null 2>&1; then
    if $SYSTEM_DEPS; then
        PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
        if command -v apt-get >/dev/null 2>&1; then
            log "Installing python${PY_MAJOR}.${PY_MINOR}-venv package..."
            apt-get update
            apt-get install -y "python${PY_MAJOR}.${PY_MINOR}-venv"
        else
            err "The Python 'venv' module is missing, but this distro is not Debian/Ubuntu."
            err "Install the equivalent 'python3-venv' package for your distro."
            exit 1
        fi
    else
        err "The Python 'venv' module is missing. On Debian/Ubuntu install it with:"
        PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
        err "  apt install python${PY_MAJOR}.${PY_MINOR}-venv"
        err "Or re-run with --system-deps to install it automatically."
        exit 1
    fi
fi

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

# --- Interactive configuration ---
if $SKIP_CONFIG; then
    log "Skipping interactive configuration (--skip-config)."
    if [[ ! -f "config/config.toml" ]]; then
        cp config/config.example.toml config/config.toml
    fi
    if [[ ! -f ".env" ]]; then
        cp .env.example .env
    fi
else
    log "Starting interactive configuration..."
    python scripts/configure.py
fi

# --- Verify install ---
log "Running backend tests..."
python -m pytest tests -q

log "Install complete."
echo
if $SKIP_CONFIG; then
    log "Next steps:"
    echo "  1. Edit .env with your LLM API key"
    echo "  2. Review config/config.toml"
else
    log "You can now start the server:"
fi
if $VOICE; then
    echo "  source .venv/bin/activate"
    echo "  python api/server.py"
    echo "  (voice/WebRTC deps installed)"
else
    echo "  source .venv/bin/activate"
    echo "  python api/server.py"
    echo "  (for voice calls, re-run: ./install.sh --voice --system-deps)"
fi
