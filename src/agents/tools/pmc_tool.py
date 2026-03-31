"""
PMC Full-Text Fetcher — uses NCBI BioC REST API.
Fetches full paper text for Open Access articles using PubMed ID.
~40% of PubMed papers have free full text via PMC Open Access.
"""
import time
import logging
import requests

logger = logging.getLogger(__name__)

# NCBI BioC API — free, no key required, works directly with PubMed IDs
_BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{uid}/unicode"
_TIMEOUT  = 10
_DELAY    = 0.5   # be polite to NCBI servers


def fetch_full_text(pubmed_uid: str) -> str | None:
    """
    Fetch full text of an Open Access paper from PMC using its PubMed ID.

    Args:
        pubmed_uid: The PubMed ID string (e.g. '34567890')

    Returns:
        Full text as a single string, or None if paper is not Open Access.
    """
    if not pubmed_uid or not pubmed_uid.strip():
        return None

    url = _BIOC_URL.format(uid=pubmed_uid.strip())

    try:
        time.sleep(_DELAY)
        response = requests.get(url, timeout=_TIMEOUT)

        # 404 = paper exists but is NOT open access (paywalled)
        if response.status_code == 404:
            logger.debug("Not Open Access: PubMed UID %s", pubmed_uid)
            return None

        response.raise_for_status()
        data = response.json()

        return _extract_text_from_bioc(data)

    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching full text for UID %s", pubmed_uid)
        return None
    except requests.exceptions.RequestException as e:
        logger.warning("Request error for UID %s: %s", pubmed_uid, e)
        return None
    except Exception as e:
        logger.warning("Unexpected error for UID %s: %s", pubmed_uid, e)
        return None


def _extract_text_from_bioc(data: dict) -> str | None:
    """
    Parse BioC JSON format and extract all passage text.
    BioC groups text into passages (title, abstract, intro, methods, results, etc.)
    """
    try:
        documents = data.get("documents", [data]) if isinstance(data, dict) else []

        if not documents:
            return None

        sections: list[str] = []

        for doc in documents:
            passages = doc.get("passages", [])
            for passage in passages:
                infons   = passage.get("infons", {})
                section  = infons.get("section_type", infons.get("type", ""))
                text     = passage.get("text", "").strip()

                if not text:
                    continue

                # Skip references, figure captions, tables — not useful for RAG
                skip_sections = {"REF", "FIGURE", "TABLE", "SUPPL", "ABBR", "COMP_INT"}
                if section.upper() in skip_sections:
                    continue

                # Add section label for better RAG context
                if section:
                    sections.append(f"[{section}] {text}")
                else:
                    sections.append(text)

        if not sections:
            return None

        full_text = "\n\n".join(sections)
        logger.debug("Extracted %d characters of full text", len(full_text))
        return full_text

    except Exception as e:
        logger.warning("BioC parsing error: %s", e)
        return None


def enrich_with_fulltext(records: list[dict]) -> tuple[list[dict], int]:
    """
    Attempt to replace abstracts with full text for Open Access papers.

    Args:
        records: List of paper dicts from search_pubmed tool.

    Returns:
        Tuple of (enriched_records, fulltext_count)
    """
    enriched   = 0
    total      = len(records)

    for i, record in enumerate(records):
        uid = record.get("uid", "").strip()
        if not uid:
            continue

        logger.info(
            "Fetching full text [%d/%d]: %s",
            i + 1, total, record["title"][:60]
        )

        full_text = fetch_full_text(uid)

        if full_text:
            # Replace abstract with full text — keep abstract as fallback note
            record["abstract"] = (
                f"[FULL TEXT AVAILABLE]\n\n"
                f"{full_text}\n\n"
                f"--- Abstract ---\n{record['abstract']}"
            )
            record["has_full_text"] = True
            enriched += 1
        else:
            record["has_full_text"] = False

    logger.info(
        "Full text enrichment: %d/%d papers had Open Access full text",
        enriched, total
    )
    return records, enriched