import sqlite3
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a single chat message stored in the database."""
    id: int | None
    session_id: str
    role: str
    content: str
    created_at: datetime
    sources: str | None = None


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all required database tables if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
            content     TEXT    NOT NULL,
            sources     TEXT,
            created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            title       TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON chat_messages(session_id)
    """)
    conn.commit()