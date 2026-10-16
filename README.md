# Astra Backend

Public, cloneable backend for the Astra Core AI agent.  This repo contains
the FastAPI server, agent core, tools, memory, CLI, and WebRTC voice-gateway
scaffold.  It does **not** contain the Flutter app.

## What you get

- HTTP API (`/health`, `/api/chat`, `/api/memory`)
- WebSocket endpoint (`/ws`) for real-time chat and WebRTC signaling
- SQLite memory, tool registry, filesystem + shell tools, MCP client stub
- CLI for interactive use
- pytest suite (runs without API keys)

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.12 |
| RAM (core only) | 512 MB | 1 GB |
| RAM (voice/WebRTC) | 2 GB | 4 GB+ |
| Disk (core only) | 200 MB | 1 GB |
| Disk (voice stack) | 2 GB+ | 5 GB+ |
| OS | Linux, Windows, macOS | Ubuntu 22.04/24.04, Debian 12 |
| Network | localhost:8321 | HTTPS/WSS on port 443 |

### Notes

- The core API (text chat + HTTP endpoints) is lightweight and runs on a
  Raspberry Pi or small VPS.
- `requirements-voice.txt` includes `aiortc`, `faster-whisper`, `piper-tts`, and
  `llama-cpp-python`. These are heavy and may require a C++ compiler and
  development headers to build wheels.
- On Debian/Ubuntu, install build tools before `--voice`:

  ```bash
  sudo apt update
  sudo apt install -y build-essential python3-dev libopus-dev portaudio19-dev
  ```

- WebRTC voice calls require a public HTTPS/WSS endpoint. The backend itself
  runs plain HTTP on `localhost:8321`; terminate TLS with Caddy, Nginx, or a
  reverse proxy.

## Quick start

### Linux / macOS

One-shot install (Python + build headers + Astra core):

```bash
git clone https://github.com/MrTech59/AstraAI-Backend.git
cd AstraAI-Backend
./install.sh --system-deps
```

For voice calls, also install the heavy voice/WebRTC deps:

```bash
./install.sh --voice --system-deps
```

If Python is already installed, omit `--system-deps`:

```bash
./install.sh
```

### Windows

One-shot install (Python + Astra core) from an elevated PowerShell:

```powershell
git clone https://github.com/MrTech59/AstraAI-Backend.git
cd AstraAI-Backend
.\install.ps1 -SystemDeps
```

For voice calls:

```powershell
.\install.ps1 -Voice -SystemDeps
```

> If the installer installs Python for the first time, close and reopen
> PowerShell before re-running the command.

## Configure

The installer now runs an interactive wizard that writes both configuration
files for you. Just answer the prompts for provider, model, API key, and bind
host/port.

```bash
./install.sh --system-deps
```

The wizard creates:

- `.env` — your API key (never commit this).
- `config/config.toml` — runtime settings.

To skip the wizard (e.g. for CI/automation):

```bash
./install.sh --skip-config
```

Then edit the files by hand.

### Supported providers

| Number | Provider | Default model |
|---|---|---|
| 1 | OpenRouter | `openrouter/free` |
| 2 | NVIDIA Build | `z-ai/glm-5.3-flash` |
| 3 | Ollama (local) | `qwen3` |
| 4 | OpenAI-compatible | `gpt-4o-mini` |

## Run

```bash
source .venv/bin/activate
python api/server.py
```

Then open the Flutter app pointing at your backend:

```bash
flutter run --dart-define=ASTRA_BACKEND_URL=wss://your-domain.com
```

For local development:

```bash
flutter run --dart-define=ASTRA_BACKEND_URL=ws://localhost:8321
```

## Hosting on a domain

WebRTC requires HTTPS/WSS.  Put a reverse proxy in front of `api/server.py`:

### Caddy

```text
your-domain.com {
    reverse_proxy localhost:8321
}
```

### Nginx

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8321;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Voice calls

Voice calls are a scaffold in this version.  The signaling path (offer/answer/ICE)
works over `/ws`, but the audio pipeline (STT → AgentLoop → TTS) needs to be
wired.  Install `requirements-voice.txt` (`aiortc`) to enable the WebRTC peer
connection on the server.

## Tests

```bash
python -m pytest tests -q
```

## Security notes

- Never commit `.env` or `config/config.toml`.
- Rotate any API key that has ever appeared in source history before making
  this repo public.
- Shell execution is disabled by default; enable only with an explicit allowlist.
