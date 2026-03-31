import time
import logging
from langchain_community.utilities import PubMedAPIWrapper
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_RATE_LIMIT_DELAY = 0.4

_pubmed = PubMedAPIWrapper(
    top_k_results=10,
    doc_content_chars_max=3000,
)


def _extract_record(doc, query: str) -> dict | None:
    """
    Safely extract fields from a PubMed result.
    Handles both Document objects and plain dicts.
    """
    try:
        # Case 1 — LangChain Document object
        if hasattr(doc, "metadata"):
            title    = doc.metadata.get("Title", "").strip()
            abstract = doc.page_content.strip()
            pub_date = doc.metadata.get("Published", "Unknown")
            uid      = doc.metadata.get("uid", "")

        # Case 2 — plain dict (newer langchain-community versions)
        elif isinstance(doc, dict):
            title    = doc.get("Title", doc.get("title", "")).strip()
            abstract = doc.get("Summary", doc.get("abstract", doc.get("page_content", ""))).strip()
            pub_date = doc.get("Published", doc.get("published", "Unknown"))
            uid      = doc.get("uid", "")

        else:
            logger.warning("Unknown PubMed result type: %s", type(doc))
            return None

        # Skip completely empty records
        if not title and not abstract:
            return None

        return {
            "title":     title or "Untitled",
            "abstract":  abstract,
            "published": pub_date,
            "uid":       uid,
            "source":    "PubMed",
            "query":     query,
        }

    except Exception as e:
        logger.warning("Failed to parse PubMed record: %s", e)
        return None


@tool
def search_pubmed(query: str) -> list[dict]:
    """
    Search PubMed for biomedical literature using the free NCBI Entrez API.

    Args:
        query: A neonatal medical topic string.

    Returns:
        List of dicts with keys: title, abstract, published, uid, source, query.
    """
    logger.info("PubMed search: '%s'", query)
    time.sleep(_RATE_LIMIT_DELAY)

    try:
        docs = _pubmed.load(query)
    except Exception as e:
        logger.error("PubMed API error for '%s': %s", query, e)
        return []

    results = [_extract_record(doc, query) for doc in docs]
    results = [r for r in results if r is not None]  # filter failed parses

    logger.info("Found %d valid results for '%s'", len(results), query)
    return results