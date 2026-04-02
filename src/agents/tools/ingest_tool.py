import hashlib
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.vectorstore.store import load_vectorstore, build_vectorstore, vectorstore_exists

logger = logging.getLogger(__name__)


def _make_doc_id(title: str, uid: str) -> str:
    """Generate a stable deduplication ID from PubMed UID or title hash."""
    key = uid.strip() if uid.strip() else title.strip().lower()
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def _get_existing_ids() -> set[str]:
    """Fetch all doc_ids already stored in Qdrant via scroll API."""
    settings = get_settings()
    try:
        client = QdrantClient(path=settings.qdrant_path)
        existing_ids: set[str] = set()
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=settings.qdrant_collection_name,
                with_payload=True,
                limit=1000,
                offset=offset,
            )
            for point in batch:
                doc_id = point.payload.get("metadata", {}).get("doc_id")
                if doc_id:
                    existing_ids.add(doc_id)
            if offset is None:
                break
        return existing_ids
    except Exception as e:
        logger.warning("Could not fetch existing IDs: %s", e)
        return set()


def ingest_pubmed_results(results: list[dict]) -> tuple[int, int]:
    """
    Chunk, deduplicate, and embed PubMed results into Qdrant.
    Creates the vector store on first run if it does not exist yet.
    """
    if not results:
        return 0, 0

    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    new_docs: list[Document] = []
    skipped = 0

    # Load existing IDs only if store already exists
    existing_ids: set[str] = _get_existing_ids() if vectorstore_exists() else set()
    vectorstore = load_vectorstore() if vectorstore_exists() else None

    for record in results:
        doc_id = _make_doc_id(record["title"], record["uid"])

        if doc_id in existing_ids:
            logger.debug("Duplicate skipped: %s", record["title"][:60])
            skipped += 1
            continue

        content = (
            f"Title: {record['title']}\n\n"
            f"Abstract: {record['abstract']}\n\n"
            f"Published: {record['published']}\n"
            f"Topic: {record['query']}"
        )

        base_doc = Document(
            page_content=content,
            metadata={
                "source": "PubMed",
                "title": record["title"],
                "published": record["published"],
                "uid": record["uid"],
                "query": record["query"],
                "doc_id": doc_id,
            },
        )

        chunks = splitter.split_documents([base_doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)

        new_docs.extend(chunks)
        existing_ids.add(doc_id)

    if new_docs:
        if vectorstore is None:
            build_vectorstore(new_docs)
            logger.info("✅ Vector store created with %d chunks.", len(new_docs))
        else:
            vectorstore.add_documents(new_docs)
            logger.info("✅ Added %d new chunks to existing store.", len(new_docs))

    return len(new_docs), skipped