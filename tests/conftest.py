import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.asyncio

# Real-API smoke scripts run manually, never during pytest.
collect_ignore = ["smoke_e2e.py", "smoke_nvidia.py", "smoke_nvidia_tools.py"]


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    """Run the app with lifespan (startup wiring) using Ollama (no key)."""
    import api.server as server
    from astra.config import Settings

    def fake_load():
        s = Settings()
        s.llm.provider = "ollama"
        s.llm.model = "qwen3"
        return s

    monkeypatch.setattr(server, "load_settings", fake_load)
    monkeypatch.setattr(server, "memory_db_path", lambda s: tmp_path / "test.db")
    with TestClient(server.app) as test_client:
        yield test_client
