"""SQLite-backed memory store.

Two kinds of state:

* conversation history (per conversation_id, time-ordered messages)
* persistent memories (key/value-ish notes, LIKE-searchable)

Independent of the LLM provider. No secrets are ever stored here.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .models import Memory, Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_call_id TEXT,
    name TEXT,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._connection = sqlite3.connect(str(db_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    # ----- conversation history -----

    def save_message(self, message: Message) -> Message:
        cursor = self._connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, tool_call_id, name, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message.conversation_id,
                message.role,
                message.content,
                message.tool_call_id,
                message.name,
                message.timestamp or time.time(),
            ),
        )
        self._connection.commit()
        message.id = cursor.lastrowid
        return message

    def get_recent_messages(self, conversation_id: str = "default", limit: int = 30) -> list[Message]:
        rows = self._connection.execute(
            """
            SELECT * FROM (
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC LIMIT ?
            ) ORDER BY id ASC
            """,
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> int:
        cursor = self._connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        self._connection.commit()
        return cursor.rowcount

    # ----- persistent memories -----

    def save_memory(self, memory: Memory) -> Memory:
        cursor = self._connection.execute(
            "INSERT INTO memories (key, content, created_at) VALUES (?, ?, ?)",
            (memory.key, memory.content, memory.created_at or time.time()),
        )
        self._connection.commit()
        memory.id = cursor.lastrowid
        return memory

    def search_memory(self, query: str, limit: int = 10) -> list[Memory]:
        like = f"%{query}%"
        rows = self._connection.execute(
            """
            SELECT * FROM memories
            WHERE content LIKE ? OR key LIKE ?
            ORDER BY id DESC LIMIT ?
            """,
            (like, like, limit),
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def list_memories(self, limit: int = 100) -> list[Memory]:
        rows = self._connection.execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def delete_memory(self, memory_id: int) -> bool:
        cursor = self._connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._connection.commit()
        return cursor.rowcount > 0

    # ----- row mappers -----

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            tool_call_id=row["tool_call_id"],
            name=row["name"],
            timestamp=row["timestamp"],
        )

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            key=row["key"],
            content=row["content"],
            created_at=row["created_at"],
        )
