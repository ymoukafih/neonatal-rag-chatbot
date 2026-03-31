import sqlite3
import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from src.config.settings import get_settings
from src.database.models import ChatMessage, create_tables


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that provides a database connection and auto-closes it."""
    settings = get_settings()
    conn = sqlite3.connect(settings.database_url)
    conn.row_factory = sqlite3.Row
    try:
        create_tables(conn)
        yield conn
    finally:
        conn.close()


def create_session(title: str = "New Chat") -> str:
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title) VALUES (?, ?)",
            (session_id, title),
        )
        conn.commit()
    return session_id


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> None:
    """Persist a chat message to the database."""
    sources_json = json.dumps(sources) if sources else None
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO chat_messages (session_id, role, content, sources)
               VALUES (?, ?, ?, ?)""",
            (session_id, role, content, sources_json),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), session_id),
        )
        conn.commit()


def get_session_history(session_id: str) -> list[ChatMessage]:
    """Retrieve all messages for a given session ordered by time."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, session_id, role, content, sources, created_at
               FROM chat_messages WHERE session_id = ?
               ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
    return [
        ChatMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            sources=row["sources"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]


def get_all_sessions() -> list[dict]:
    """Return all sessions ordered by most recently updated."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages."""
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()