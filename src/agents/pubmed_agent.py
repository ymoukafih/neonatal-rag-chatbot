import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

NEONATAL_QUERIES = [
    '(neonatal[tiab] OR newborn[tiab] OR neonate[tiab]) AND ("practice guideline"[pt] OR "guideline"[pt])',
    '(neonatal[tiab] OR newborn[tiab] OR neonate[tiab]) AND "systematic review"[pt]',
    '(neonatal[tiab] OR newborn[tiab] OR neonate[tiab]) AND "meta-analysis"[pt]',
    '(neonatal jaundice[tiab] OR neonatal hyperbilirubinemia[tiab]) AND "randomized controlled trial"[pt]',
    '(neonatal sepsis[tiab] OR neonatal infection[tiab]) AND ("systematic review"[pt] OR "guideline"[pt])',
    '(neonatal nutrition[tiab] OR breastfeeding[tiab] OR infant feeding[tiab]) AND ("systematic review"[pt] OR "guideline"[pt])',
    '(neonatal respiratory[tiab] OR respiratory distress syndrome[tiab]) AND ("systematic review"[pt] OR "guideline"[pt])',
    '(neonatal screening[tiab] OR newborn screening[tiab]) AND ("guideline"[pt] OR "systematic review"[pt])',
    '(neonatal resuscitation[tiab]) AND ("guideline"[pt] OR "systematic review"[pt])',
    '(sudden infant death[tiab] OR SIDS[tiab] OR safe sleep[tiab]) AND ("guideline"[pt] OR "systematic review"[pt])',
    '(neonatal pain[tiab] OR neonatal care[tiab]) AND ("systematic review"[pt] OR "guideline"[pt])',
    '(preterm infant[tiab] OR premature infant[tiab]) AND ("systematic review"[pt] OR "guideline"[pt])',
    '(neonatal vaccination[tiab] OR neonatal immunization[tiab]) AND ("guideline"[pt])',
    '(developmental screening[tiab] OR infant development[tiab]) AND ("guideline"[pt] OR "systematic review"[pt])',
]

HIGH_QUALITY_JOURNALS = {
    "pediatrics",
    "jama pediatrics",
    "the lancet",
    "lancet child adolescent health",
    "journal of perinatology",
    "neonatology",
    "early human development",
    "archives of disease in childhood",
    "acta paediatrica",
    "bmc pediatrics",
    "frontiers in pediatrics",
    "american journal of obstetrics and gynecology",
    "journal of pediatrics",
    "european journal of pediatrics",
    "seminars in neonatology",
    "seminars in perinatology",
    "clinics in perinatology",
}


def _sleep():
    time.sleep(1.0 / settings.pubmed_requests_per_sec)


@retry(
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get(url: str, params: dict) -> requests.Response:
    params["api_key"] = settings.ncbi_api_key
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    _sleep()
    return r


def _load_checkpoint() -> set[str]:
    path = Path(settings.checkpoint_file)
    if path.exists():
        try:
            content = path.read_text(encoding="utf-8").strip()
            if content:
                data = json.loads(content)
                logger.info(f"Resuming — {len(data)} PMIDs already processed.")
                return set(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Checkpoint corrupted ({e}) — starting fresh.")
    return set()


def _save_checkpoint(processed: set[str]):
    Path(settings.checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.checkpoint_file).write_text(
        json.dumps(list(processed)), encoding="utf-8"
    )


def search_pubmed(query: str, max_results: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query + ' AND ("2019"[pdat] : "2026"[pdat])',
        "retmax": max_results,
        "sort": "relevance",
        "retmode": "json",
    }
    try:
        r = _get(BASE_URL + "esearch.fcgi", params)
        return r.json()["esearchresult"].get("idlist", [])
    except Exception as e:
        logger.error(f"Search failed for query '{query[:60]}...': {e}")
        return []


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    try:
        r = _get(BASE_URL + "efetch.fcgi", params)
    except Exception as e:
        logger.error(f"Fetch failed for {len(pmids)} PMIDs: {e}")
        return []

    articles = []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID", default="")
            title = article.findtext(".//ArticleTitle", default="").strip()
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (p.attrib.get("Label", "") + ": " if p.attrib.get("Label") else "") + (p.text or "")
                for p in abstract_parts
            ).strip()
            journal = article.findtext(".//Journal/Title", default="").strip().lower()
            pub_year = article.findtext(".//PubDate/Year", default="")
            pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
            authors = [
                f"{a.findtext('LastName', '')} {a.findtext('ForeName', '')}".strip()
                for a in article.findall(".//Author")[:3]
            ]

            if not abstract or not title:
                continue

            if journal and not any(q in journal for q in HIGH_QUALITY_JOURNALS):
                continue

            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": pub_year,
                "publication_types": pub_types,
                "authors": authors,
                "source": f"PubMed PMID:{pmid}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            continue

    return articles


def run_pubmed_agent() -> list[dict]:
    processed_pmids = _load_checkpoint()
    output_path = Path(settings.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict] = []
    if output_path.exists():
        try:
            content = output_path.read_text(encoding="utf-8").strip()
            if content:
                all_articles = json.loads(content)
                logger.info(f"Loaded {len(all_articles)} existing articles from {output_path}")
            else:
                logger.warning("output.json exists but is empty — starting fresh.")
        except json.JSONDecodeError as e:
            logger.warning(f"output.json is corrupted ({e}) — starting fresh.")
            output_path.unlink()

    for i, query in enumerate(NEONATAL_QUERIES, 1):
        logger.info(f"[{i}/{len(NEONATAL_QUERIES)}] Searching: {query[:80]}...")
        pmids = search_pubmed(query, settings.pubmed_max_results)
        new_pmids = [p for p in pmids if p not in processed_pmids]
        logger.info(f"  → {len(pmids)} results, {len(new_pmids)} new PMIDs to fetch")

        for batch_start in range(0, len(new_pmids), 100):
            batch = new_pmids[batch_start: batch_start + 100]
            articles = fetch_abstracts(batch)
            all_articles.extend(articles)
            processed_pmids.update(batch)

            _save_checkpoint(processed_pmids)
            output_path.write_text(
                json.dumps(all_articles, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

            logger.info(
                f"  Batch {batch_start // 100 + 1}: fetched {len(articles)} quality articles"
                f" | Total so far: {len(all_articles)}"
            )

    logger.info(f"\n✅ Done. {len(all_articles)} high-quality articles saved to {output_path}")
    return all_articles