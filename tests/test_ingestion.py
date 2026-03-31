"""Tests for src/ingestion/loader.py"""
import json
import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.loader import (
    records_to_documents,
    chunk_documents,
)


SAMPLE_RECORDS = [
    {
        "title": "Neonatal Respiratory Distress",
        "description": "RDS is a common condition in preterm infants.",
        "treatment": "Surfactant therapy is the primary treatment.",
    },
    {
        "title": "Neonatal Jaundice",
        "description": "Hyperbilirubinemia occurs in up to 60% of newborns.",
        "treatment": "Phototherapy is the standard treatment.",
    },
]


def test_records_to_documents_returns_documents():
    docs = records_to_documents(SAMPLE_RECORDS)
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)


def test_records_to_documents_content_not_empty():
    docs = records_to_documents(SAMPLE_RECORDS)
    assert all(len(d.page_content) > 0 for d in docs)


def test_records_to_documents_has_metadata():
    docs = records_to_documents(SAMPLE_RECORDS)
    assert all("source" in d.metadata for d in docs)
    assert all("record_index" in d.metadata for d in docs)


def test_chunk_documents_splits_correctly():
    docs = records_to_documents(SAMPLE_RECORDS)
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)


def test_empty_records_returns_empty():
    docs = records_to_documents([])
    assert docs == []


def test_records_with_no_text_skipped():
    records = [{"count": 42, "active": True}]  # no string fields
    docs = records_to_documents(records)
    assert docs == []