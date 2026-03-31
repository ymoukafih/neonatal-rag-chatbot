import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def get_embeddings() -> OllamaEmbeddings:
    """Return the configured Ollama embedding model."""
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def build_vectorstore(documents: list[Document]) -> Chroma:
    """Embed documents and persist them to ChromaDB."""
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building vector store with %d chunks → %s", len(documents), persist_dir)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=settings.chroma_collection_name,
        persist_directory=str(persist_dir),
    )
    logger.info("Vector store built and persisted successfully.")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing persisted ChromaDB vector store from disk."""
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. "
            "Run ingestion first: python scripts/ingest.py"
        )

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def get_retriever(vectorstore: Chroma | None = None):
    """Return a LangChain retriever from the vector store."""
    settings = get_settings()
    vs = vectorstore or load_vectorstore()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.top_k_results,
            "fetch_k": settings.top_k_results * 3,
        },
    )


def vectorstore_exists() -> bool:
    """Check whether the vector store has already been built."""
    settings = get_settings()
    return Path(settings.chroma_persist_dir).exists()