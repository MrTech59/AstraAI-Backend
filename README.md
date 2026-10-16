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

## Quick start

### Linux / macOS

```bash
git clone https://github.com/YOUR_USERNAME/astra-backend.git
cd astra-backend
./install.sh
```

For voice calls, also install the heavy voice/WebRTC deps:

```bash
./install.sh --voice
```

### Windows

```powershell
git clone https://github.com/YOUR_USERNAME/astra-backend.git
cd astra-backend
.\install.ps1
```

```powershell
.\install.ps1 -Voice
```

## Configure

The installer creates two files from examples if they don't exist:

- `.env` — put your LLM API key here (never commit this).
- `config/config.toml` — runtime settings.

Edit `.env`:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-...
# or
NVIDIA_API_KEY=nvapi-...
# or
OLLAMA_API_KEY=...
```

Edit `config/config.toml` to choose a provider:

```toml
[llm]
provider = "ollama"   # or "openrouter", "nvidia_build", "openai-compatible"
model = "qwen3"
# api_key is read from .env automatically
```

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
