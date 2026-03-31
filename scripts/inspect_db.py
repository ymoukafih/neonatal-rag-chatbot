"""
Inspect the ChromaDB vector store.
Usage: python scripts/inspect_db.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vectorstore.store import load_vectorstore


def main() -> None:
    vs = load_vectorstore()
    data = vs.get(include=["metadatas"])

    ids       = data["ids"]
    metadatas = data["metadatas"]

    total = len(ids)

    # Count by source
    sources: dict[str, int] = {}
    for m in metadatas:
        src = m.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    # Count unique papers (by doc_id)
    unique_papers = len({m.get("doc_id", "") for m in metadatas if m.get("doc_id")})

    # Count by topic/query
    topics: dict[str, int] = {}
    for m in metadatas:
        q = m.get("query", "")
        if q:
            topics[q] = topics.get(q, 0) + 1

    print(f"\n{'═' * 55}")
    print(f"  📊 ChromaDB Vector Store Report")
    print(f"{'═' * 55}")
    print(f"  Total chunks      : {total}")
    print(f"  Unique papers     : {unique_papers}")
    print(f"\n  By source:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    • {src:<20} {count} chunks")

    print(f"\n  By topic (top 10):")
    for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:10]:
        print(f"    • {topic[:45]:<45} {count}")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    main()