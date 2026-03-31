import json
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def load_json_data(path: str | None = None) -> list[dict]:
    """Load raw records from the JSON data file."""
    settings = get_settings()
    data_path = Path(path or settings.data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("data") or data.get("records") or list(data.values())
    else:
        raise ValueError(f"Unexpected JSON structure in {data_path}")

    logger.info("Loaded %d records from %s", len(records), data_path)
    return records


def records_to_documents(records: list[dict]) -> list[Document]:
    """Convert raw JSON records into LangChain Document objects with metadata."""
    documents = []
    for i, record in enumerate(records):
        content_parts = []
        metadata = {"source": "output.json", "record_index": i}

        for key, value in record.items():
            if isinstance(value, str) and value.strip():
                content_parts.append(f"{key}: {value}")
                if len(value) < 200:
                    metadata[key] = value
            elif isinstance(value, (int, float, bool)):
                metadata[key] = value

        if not content_parts:
            continue

        documents.append(
            Document(
                page_content="\n".join(content_parts),
                metadata=metadata,
            )
        )

    logger.info("Converted %d records to documents", len(documents))
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def load_and_chunk(path: str | None = None) -> list[Document]:
    """Full pipeline: load JSON → convert to documents → chunk."""
    records = load_json_data(path)
    documents = records_to_documents(records)
    return chunk_documents(documents)