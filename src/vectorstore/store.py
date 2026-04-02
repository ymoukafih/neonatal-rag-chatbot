import logging
import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import ConfigDict
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_BM25_INDEX_PATH = "./data/bm25_index.pkl"
_BM25_DOCS_PATH  = "./data/bm25_docs.pkl"


# ── Embeddings ────────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return BGE-M3 multilingual embedding model (Arabic, French, English, Darija)."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Qdrant ────────────────────────────────────────────────────────────────────

def _get_qdrant_client() -> QdrantClient:
    """Return local persistent Qdrant client (no server required)."""
    settings = get_settings()
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=settings.qdrant_path)


def build_vectorstore(documents: list[Document]) -> QdrantVectorStore:
    """Embed documents, persist to Qdrant, and build BM25 index."""
    settings = get_settings()
    embeddings = get_embeddings()

    logger.info("Building Qdrant vector store with %d chunks", len(documents))
    vectorstore = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection_name,
    )

    _build_bm25_index(documents)
    logger.info("Qdrant vector store and BM25 index built successfully.")
    return vectorstore


def load_vectorstore() -> QdrantVectorStore:
    """Load existing Qdrant vector store from disk."""
    settings = get_settings()
    if not Path(settings.qdrant_path).exists():
        raise FileNotFoundError(
            f"Vector store not found at {settings.qdrant_path}. "
            "Run ingestion first: uv run python scripts/run_agent.py"
        )
    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=settings.qdrant_collection_name,
        embedding=get_embeddings(),
    )


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _build_bm25_index(documents: list[Document]) -> None:
    """Build and persist a BM25 index from documents."""
    corpus = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(corpus)
    with open(_BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(_BM25_DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    logger.info("BM25 index persisted to %s", _BM25_INDEX_PATH)


def _load_bm25_index() -> tuple[BM25Okapi, list[Document]]:
    """Load the persisted BM25 index and its document corpus."""
    if not Path(_BM25_INDEX_PATH).exists():
        raise FileNotFoundError(
            "BM25 index not found. "
            "Run ingestion first: uv run python scripts/run_agent.py"
        )
    with open(_BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(_BM25_DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return bm25, docs


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever: BM25 (lexical) + Qdrant (semantic) → cross-encoder reranking.

    Pipeline:
        1. BM25 retrieves top fetch_k candidates  (keyword matching)
        2. Qdrant retrieves top fetch_k candidates (semantic similarity)
        3. Results are merged and deduplicated
        4. BGE-reranker-v2-m3 reranks all candidates
        5. Top k results are returned to the RAG chain
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Any
    bm25: Any
    bm25_docs: list[Document]
    reranker: Any
    top_k: int = 5
    fetch_k: int = 20

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Run hybrid retrieval + reranking for a given query."""

        # Handle dict input when called from RunnableParallel
        if isinstance(query, dict):
            query = query.get("question", str(query))

        # 1. BM25 — lexical retrieval
        tokenized = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized)
        top_bm25_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[: self.fetch_k]
        bm25_results = [self.bm25_docs[i] for i in top_bm25_idx]

        # 2. Qdrant — semantic retrieval
        qdrant_results = self.vectorstore.similarity_search(query, k=self.fetch_k)

        # 3. Merge + deduplicate on first 100 chars of content
        seen: set[str] = set()
        candidates: list[Document] = []
        for doc in bm25_results + qdrant_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

        # 4. Cross-encoder reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(
            zip(scores, candidates), key=lambda x: x[0], reverse=True
        )

        logger.debug(
            "Hybrid retrieval: %d BM25 + %d Qdrant → %d unique → top %d reranked",
            len(bm25_results), len(qdrant_results), len(candidates), self.top_k,
        )
        return [doc for _, doc in ranked[: self.top_k]]


# ── Public API ────────────────────────────────────────────────────────────────

def get_retriever(vectorstore: QdrantVectorStore | None = None) -> HybridRetriever:
    """Return the hybrid BM25 + Qdrant + reranker retriever."""
    settings = get_settings()
    vs = vectorstore or load_vectorstore()
    bm25, bm25_docs = _load_bm25_index()
    reranker = CrossEncoder(settings.reranker_model)

    return HybridRetriever(
        vectorstore=vs,
        bm25=bm25,
        bm25_docs=bm25_docs,
        reranker=reranker,
        top_k=settings.top_k_results,
        fetch_k=settings.reranker_fetch_k,
    )


def vectorstore_exists() -> bool:
    """Check whether both Qdrant and BM25 stores have been built."""
    settings = get_settings()
    return (
        Path(settings.qdrant_path).exists()
        and Path(_BM25_INDEX_PATH).exists()
    )