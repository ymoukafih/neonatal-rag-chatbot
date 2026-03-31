"""Tests for src/database/crud.py — uses a temporary in-memory DB."""
import pytest
from unittest.mock import patch

from src.database import crud


@pytest.fixture
def temp_db(tmp_path):
    """Override database_url to use a temp file for each test."""
    db_path = str(tmp_path / "test.db")
    with patch("src.database.crud.get_settings") as mock:
        mock.return_value.database_url = db_path
        yield db_path


def test_create_session_returns_string(temp_db):
    session_id = crud.create_session("Test Session")
    assert isinstance(session_id, str)
    assert len(session_id) == 36  # UUID format


def test_save_and_retrieve_message(temp_db):
    session_id = crud.create_session()
    crud.save_message(session_id, "user", "What is RDS?")
    crud.save_message(session_id, "assistant", "RDS stands for...")

    history = crud.get_session_history(session_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"


def test_save_message_with_sources(temp_db):
    session_id = crud.create_session()
    sources = [{"content": "Some neonatal info", "metadata": {"source": "PubMed"}}]
    crud.save_message(session_id, "assistant", "Answer", sources=sources)

    history = crud.get_session_history(session_id)
    assert history[0].sources is not None


def test_delete_session(temp_db):
    session_id = crud.create_session()
    crud.save_message(session_id, "user", "Hello")
    crud.delete_session(session_id)

    history = crud.get_session_history(session_id)
    assert history == []


def test_get_all_sessions(temp_db):
    crud.create_session("Session A")
    crud.create_session("Session B")
    sessions = crud.get_all_sessions()
    assert len(sessions) >= 2