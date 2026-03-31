"""Tests for config/settings.py"""
from src.config.settings import get_settings, Settings


def test_settings_returns_settings_instance():
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_settings_default_model():
    settings = get_settings()
    assert settings.ollama_model == "llama3.2"


def test_settings_chunk_size_positive():
    settings = get_settings()
    assert settings.chunk_size > 0


def test_settings_top_k_positive():
    settings = get_settings()
    assert settings.top_k_results > 0


def test_settings_cached(  ):
    """get_settings() should return the same object every call."""
    assert get_settings() is get_settings()