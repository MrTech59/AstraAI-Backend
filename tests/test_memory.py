"""Memory tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from astra.memory import Memory, MemoryStore, Message


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "test.db")


class TestMessageStore:
    def test_save_and_get(self, store: MemoryStore) -> None:
        store.save_message(Message(role="user", content="hi", conversation_id="c1"))
        store.save_message(Message(role="assistant", content="hello", conversation_id="c1"))
        messages = store.get_recent_messages("c1", 10)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].content == "hello"

    def test_limit(self, store: MemoryStore) -> None:
        for i in range(5):
            store.save_message(Message(role="user", content=str(i), conversation_id="c1"))
        messages = store.get_recent_messages("c1", 3)
        assert len(messages) == 3
        assert messages[-1].content == "4"

    def test_delete_conversation(self, store: MemoryStore) -> None:
        store.save_message(Message(role="user", content="hi", conversation_id="c1"))
        store.save_message(Message(role="user", content="bye", conversation_id="c2"))
        assert store.delete_conversation("c1") == 1
        assert len(store.get_recent_messages("c1")) == 0
        assert len(store.get_recent_messages("c2")) == 1


class TestPersistentMemory:
    def test_save_and_search(self, store: MemoryStore) -> None:
        store.save_memory(Memory(content="The sky is blue"))
        store.save_memory(Memory(content="Grass is green"))
        results = store.search_memory("sky")
        assert len(results) == 1
        assert "sky" in results[0].content.lower()

    def test_list(self, store: MemoryStore) -> None:
        store.save_memory(Memory(content="a"))
        store.save_memory(Memory(content="b"))
        assert len(store.list_memories()) == 2

    def test_delete(self, store: MemoryStore) -> None:
        memory = store.save_memory(Memory(content="x"))
        assert store.delete_memory(memory.id or 0)
        assert not store.delete_memory(9999)
