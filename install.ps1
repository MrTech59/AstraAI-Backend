# Astra Backend Installer
# Usage: .\install.ps1 [-Voice] [-SystemDeps]

param(
    [switch]$Voice,
    [switch]$SystemDeps,
    [switch]$SkipConfig
)

$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$MinPy = [version]"3.11"

function Log($msg) { Write-Host "[astra] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[astra] $msg" -ForegroundColor Yellow }
function Err($msg) { Write-Host "[astra] $msg" -ForegroundColor Red }

# --- Install system dependencies on Windows ---
function Install-SystemDeps {
    Log "Attempting to install Python via winget..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Err "winget not found. Please install Python $MinPy+ manually from https://www.python.org/downloads/"
        exit 1
    }
    winget install --id Python.Python.3.12 --scope machine --accept-source-agreements --accept-package-agreements
    Log "Python installed. You may need to restart your terminal and re-run this installer."
}

if ($SystemDeps) {
    Install-SystemDeps
}

# --- Python version check ---
Log "Checking Python..."
$pyVersion = $null
try {
    $pyVersionString = & $Python --version 2>&1
    $pyVersion = [version]($pyVersionString -replace "Python ", "")
} catch {
    if (-not $SystemDeps) {
        Err "Python is not installed. Re-run with -SystemDeps to install it automatically,"
        Err "or install Python $MinPy+ manually from https://www.python.org/downloads/"
        exit 1
    } else {
        Err "Python is still not available after installing system deps."
        Err "Restart your terminal and re-run this installer."
        exit 1
    }
}

if ($pyVersion -lt $MinPy) {
    Err "Python $MinPy+ is required, found $pyVersion"
    if (-not $SystemDeps) {
        Err "Re-run with -SystemDeps to attempt an upgrade."
    }
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

# --- Interactive configuration ---
if ($SkipConfig) {
    Log "Skipping interactive configuration (-SkipConfig)."
    if (-not (Test-Path "config\config.toml")) {
        Copy-Item "config\config.example.toml" "config\config.toml"
    }
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
    }
} else {
    Log "Starting interactive configuration..."
    python scripts\configure.py
}

# --- Verify install ---
Log "Running backend tests..."
python -m pytest tests -q

Log "Install complete."
Write-Host
if ($SkipConfig) {
    Log "Next steps:"
    Write-Host "  1. Edit .env with your LLM API key"
    Write-Host "  2. Review config\config.toml"
} else {
    Log "You can now start the server:"
}
if ($Voice) {
    Write-Host "  .venv\Scripts\activate"
    Write-Host "  python api/server.py"
    Write-Host "  (voice/WebRTC deps installed)"
} else {
    Write-Host "  .venv\Scripts\activate"
    Write-Host "  python api/server.py"
    Write-Host "  (for voice calls, re-run: .\install.ps1 -Voice -SystemDeps)"
}
