# Astra Backend Installer
# Usage: .\install.ps1 [-Voice]

param(
    [switch]$Voice
)

$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$MinPy = [version]"3.11"

function Log($msg) { Write-Host "[astra] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[astra] $msg" -ForegroundColor Yellow }
function Err($msg) { Write-Host "[astra] $msg" -ForegroundColor Red }

# --- Python version check ---
Log "Checking Python version..."
try {
    $pyVersionString = & $Python --version 2>&1
    $pyVersion = [version]($pyVersionString -replace "Python ", "")
} catch {
    Err "Could not determine Python version. Is Python installed and in PATH?"
    exit 1
}

if ($pyVersion -lt $MinPy) {
    Err "Python $MinPy+ is required, found $pyVersion"
    exit 1
}
Log "Found Python $pyVersion"

# --- Create venv ---
$venvDir = ".venv"
if (-not (Test-Path $venvDir)) {
    Log "Creating virtual environment in $venvDir..."
    & $Python -m venv $venvDir
} else {
    Warn "Virtual environment already exists, reusing."
}

# --- Activate venv ---
$activate = Join-Path $venvDir "Scripts\Activate.ps1"
. $activate

Log "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- Install core deps ---
Log "Installing core requirements..."
pip install -r requirements.txt

# --- Optional voice deps ---
if ($Voice) {
    Log "Installing voice/WebRTC requirements (this may take a while)..."
    pip install -r requirements-voice.txt
}

# --- Create config files from examples ---
if (-not (Test-Path "config\config.toml")) {
    Log "Creating config\config.toml from example..."
    Copy-Item "config\config.example.toml" "config\config.toml"
} else {
    Warn "config\config.toml already exists, not overwriting."
}

if (-not (Test-Path ".env")) {
    Log "Creating .env from example..."
    Copy-Item ".env.example" ".env"
    Warn "Edit .env and add your API key before running the server."
} else {
    Warn ".env already exists, not overwriting."
}

# --- Verify install ---
Log "Running backend tests..."
python -m pytest tests -q

Log "Install complete."
Write-Host
Log "Next steps:"
Write-Host "  1. Edit .env with your LLM API key"
Write-Host "  2. Review config\config.toml"
if ($Voice) {
    Write-Host "  3. Run: python api/server.py"
    Write-Host "     (voice/WebRTC deps installed)"
} else {
    Write-Host "  3. Run: python api/server.py"
    Write-Host "     (for voice calls, re-run: .\install.ps1 -Voice)"
}
