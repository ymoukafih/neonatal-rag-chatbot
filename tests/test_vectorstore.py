"""Tests for src/vectorstore/store.py"""
from unittest.mock import patch, MagicMock
from src.vectorstore.store import vectorstore_exists, get_embeddings


def test_vectorstore_exists_returns_false_when_missing(tmp_path):
    with patch("src.vectorstore.store.get_settings") as mock:
        mock.return_value.chroma_persist_dir = str(tmp_path / "nonexistent")
        assert vectorstore_exists() is False


def test_vectorstore_exists_returns_true_when_present(tmp_path):
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    with patch("src.vectorstore.store.get_settings") as mock:
        mock.return_value.chroma_persist_dir = str(chroma_dir)
        assert vectorstore_exists() is True


def test_get_embeddings_returns_ollama_embeddings():
    from langchain_ollama import OllamaEmbeddings
    embeddings = get_embeddings()
    assert isinstance(embeddings, OllamaEmbeddings)