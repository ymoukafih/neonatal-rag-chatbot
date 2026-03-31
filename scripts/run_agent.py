"""
Automatically fetches and embeds ALL neonatal topics into ChromaDB.
Usage: uv run python scripts/run_agent.py
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path so 'src' is always found
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.logging_config import setup_logging
setup_logging("INFO")

from src.agents.pubmed_agent import pubmed_agent
from src.agents.state import AgentState

NEONATAL_TOPICS = [
    "neonatal respiratory distress syndrome",
    "neonatal jaundice hyperbilirubinemia",
    "neonatal sepsis diagnosis treatment",
    "preterm infant care oxygen therapy",
    "neonatal hypoxic ischemic encephalopathy",
]


def run(topic: str) -> None:
    initial_state = AgentState(topic=topic)
    final_state = pubmed_agent.invoke(initial_state)
    print("\n" + "─" * 60)
    print(final_state["summary"])
    print("─" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run with custom topic from command line
        run(" ".join(sys.argv[1:]))
    else:
        # Run all default neonatal topics
        print(f"Running agent for {len(NEONATAL_TOPICS)} neonatal topics...\n")
        for topic in NEONATAL_TOPICS:
            run(topic)