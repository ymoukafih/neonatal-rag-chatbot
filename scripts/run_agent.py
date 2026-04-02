import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.pubmed_agent import run_pubmed_agent
from src.config.settings import get_settings

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

if not settings.ncbi_api_key:
    print("❌ NCBI_API_KEY is missing in your .env file!")
    print("   Get yours at: https://www.ncbi.nlm.nih.gov/myncbi/")
    sys.exit(1)

if __name__ == "__main__":
    print("🔬 Starting PubMed neonatal data collection...")
    print(f"   Rate limit : {settings.pubmed_requests_per_sec} req/sec")
    print(f"   Max results: {settings.pubmed_max_results} per query")
    print(f"   Queries    : 14 high-evidence neonatal topics")
    print(f"   Checkpoint : {settings.checkpoint_file}\n")

    articles = run_pubmed_agent()
    print(f"\n✅ Collection complete: {len(articles)} articles saved to {settings.output_file}")