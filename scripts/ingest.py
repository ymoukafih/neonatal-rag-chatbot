"""
Run once before starting the app to embed your data.
Usage: python scripts/ingest.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_settings
from src.ingestion.loader import load_and_chunk
from src.vectorstore.store import build_vectorstore

logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    settings = get_settings()
    logger.info("Starting ingestion pipeline...")
    logger.info("Data source: %s", settings.data_path)

    chunks = load_and_chunk()
    logger.info("Total chunks ready for embedding: %d", len(chunks))

    build_vectorstore(chunks)
    logger.info("✅ Done. Vector store saved to: %s", settings.chroma_persist_dir)


if __name__ == "__main__":
    main()