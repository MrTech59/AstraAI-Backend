"""API tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.server as server


class TestHealth:
    def test_health(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestInfo:
    def test_info(self, client) -> None:
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["provider"] == "ollama"


class TestMemory:
    def test_memory_empty(self, client) -> None:
        response = client.get("/api/memory?conversation_id=test&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "test"
        assert data["messages"] == []
