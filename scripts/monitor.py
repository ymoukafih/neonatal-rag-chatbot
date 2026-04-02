import time
import pickle
import sys
from pathlib import Path

BM25_DOCS  = Path("./data/bm25_docs.pkl")
BM25_INDEX = Path("./data/bm25_index.pkl")
QDRANT_DIR = Path("./data/qdrant")
JSON_FILE  = Path("./data/output.json")

def fmt(size):
    for unit in ["B", "KB", "MB"]:
        if size < 1024: return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"

def check():
    print("\033[2J\033[H", end="")
    print("=" * 50)
    print("  📡 RAG Ingestion Monitor  —  Ctrl+C to stop")
    print("=" * 50)

    if JSON_FILE.exists():
        print(f"\n✅ output.json     {fmt(JSON_FILE.stat().st_size)}")
    else:
        print(f"\n⏳ output.json     not yet created")

    if QDRANT_DIR.exists():
        total = sum(f.stat().st_size for f in QDRANT_DIR.rglob("*") if f.is_file())
        files = sum(1 for f in QDRANT_DIR.rglob("*") if f.is_file())
        print(f"✅ qdrant/         {fmt(total)}  ({files} files)")
    else:
        print(f"⏳ qdrant/         not yet created")

    if BM25_INDEX.exists():
        print(f"✅ bm25_index.pkl  {fmt(BM25_INDEX.stat().st_size)}")
    else:
        print(f"⏳ bm25_index.pkl  not yet created")

    if BM25_DOCS.exists():
        print(f"✅ bm25_docs.pkl   {fmt(BM25_DOCS.stat().st_size)}")
        try:
            with open(BM25_DOCS, "rb") as f:
                docs = pickle.load(f)
            print(f"\n   📄 Documents indexed : {len(docs)}")
            print(f"   📝 Preview: {docs[0].page_content[:120]}...")
        except Exception:
            print("   ⏳ Still writing...")
    else:
        print(f"⏳ bm25_docs.pkl   not yet created")

    print("\n" + "=" * 50)
    print("⏳  Watching for changes... Ctrl+C to stop")
    print("=" * 50)

# ← Always loop forever, never auto-exit
try:
    while True:
        check()
        time.sleep(5)
except KeyboardInterrupt:
    print("\n\nMonitor stopped.")