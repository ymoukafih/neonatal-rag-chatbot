🩺 Neonatal RAG Chatbot
A local-first, privacy-preserving Retrieval-Augmented Generation (RAG) chatbot for neonatal clinical decision support. Built with Ollama LLMs, Qdrant vector search, hybrid BM25 retrieval, cross-encoder reranking, and a Gradio web interface.
________________________________________
✨ Features
•	Hybrid Retrieval — BM25 (lexical) + Qdrant (semantic) search combined for maximum recall
•	Cross-Encoder Reranking — BGE-reranker-v2-m3 reranks candidates for precision
•	Local-first — All models run locally via Ollama; no data leaves your machine
•	Multilingual — BGE-M3 embeddings support Arabic, French, English, and Darija
•	Session History — Conversations persisted to SQLite with source tracking
•	Gradio UI — Clean chat interface with multi-session support
________________________________________
🏗️ Architecture
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Gradio UI (port 7860)   │
└────────────────┬────────────────┘
                 │
    ┌────────────▼────────────┐
    │      RAG Chain          │
    │   (LangChain + Ollama)  │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │    HybridRetriever      │
    │  BM25 + Qdrant + BGE    │
    │     Reranker v2-m3      │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │   Qdrant (local disk)   │
    │   BM25 index (pickle)   │
    └─────────────────────────┘

________________________________________
📋 Requirements
•	Python 3.12+
•	uv package manager
•	Ollama running locally
________________________________________
🚀 Quick Start
1. Clone & Install
git clone https://github.com/your-org/neonatal-rag-chatbot.git
cd neonatal-rag-chatbot
uv sync
uv pip install -e .

2. Start Ollama & Pull Models
# Pull the LLM
ollama pull llama3.2

# Pull the embedding model
ollama pull nomic-embed-text

3. Configure Environment (Optional)
Copy and edit the environment file:
copy .env.example .env

Key settings in .env:
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
TOP_K_RESULTS=5
RERANKER_FETCH_K=20
APP_PORT=7860

4. Run the Ingestion Agent
This scrapes the neonatal knowledge base, embeds it, and builds the vector stores:
uv run python scripts/run_agent.py

⏳ First run only — downloads BGE-M3 (~2.3 GB) and builds Qdrant + BM25 indexes. Takes 5–15 minutes on CPU. Subsequent runs are instant.
5. Launch the App
uv run python src/ui/app.py

Open your browser at http://localhost:7860
________________________________________
📁 Project Structure
neonatal-rag-chatbot/
├── src/
│   ├── config/
│   │   └── settings.py          # Pydantic settings (reads .env)
│   ├── database/
│   │   └── crud.py              # SQLite session & message persistence
│   ├── ingestion/
│   │   └── loader.py            # Document loading & chunking
│   ├── rag/
│   │   ├── chain.py             # RAG chain (LangChain + Ollama)
│   │   └── prompts.py           # System & RAG prompt templates
│   ├── vectorstore/
│   │   └── store.py             # Qdrant + BM25 + HybridRetriever
│   └── ui/
│       └── app.py               # Gradio chat interface
├── scripts/
│   ├── run_agent.py             # Ingestion agent (scrape → embed → index)
│   ├── monitor.py               # Live ingestion monitor (second terminal)
│   └── inspect_db.py            # Inspect SQLite chat history
├── data/                        # Auto-created by run_agent.py
│   ├── qdrant/                  # Qdrant vector store (local disk)
│   ├── bm25_index.pkl           # BM25 index
│   ├── bm25_docs.pkl            # BM25 document corpus
│   └── chatbot.db               # SQLite session history
├── .env.example                 # Environment variable template
├── pyproject.toml
└── README.md

________________________________________
🔍 Retrieval Pipeline
The HybridRetriever runs a 4-step pipeline on every query:
Step	Method	Purpose
1	BM25 (Okapi)	Lexical keyword matching — top fetch_k candidates
2	Qdrant HNSW	Semantic similarity search — top fetch_k candidates
3	Deduplication	Merge results on first 100 chars, remove duplicates
4	BGE Reranker	Cross-encoder reranking → return top k results

Default values: fetch_k=20, top_k=5 (configurable in .env).
________________________________________
🩺 Monitoring Ingestion
While run_agent.py is running in Terminal 1, open Terminal 2 and watch progress live:
uv run python scripts/monitor.py

It refreshes every 5 seconds showing file sizes, document count, and a chunk preview.
________________________________________
🗄️ Inspect Chat History
uv run python scripts/inspect_db.py

________________________________________
⚙️ Configuration Reference
All settings are in src/config/settings.py and can be overridden via .env:
Variable	Default	Description
OLLAMA_MODEL	llama3.2	LLM used for answer generation
OLLAMA_BASE_URL	http://localhost:11434	Ollama server URL
OLLAMA_TEMPERATURE	0.1	LLM sampling temperature
EMBEDDING_MODEL	BAAI/bge-m3	HuggingFace embedding model
RERANKER_MODEL	BAAI/bge-reranker-v2-m3	Cross-encoder reranker
QDRANT_PATH	./data/qdrant	Local Qdrant storage path
QDRANT_COLLECTION_NAME	neonatal_knowledge	Qdrant collection name
TOP_K_RESULTS	5	Final documents returned to LLM
RERANKER_FETCH_K	20	Candidates fetched before reranking
CHUNK_SIZE	512	Token chunk size for splitting
CHUNK_OVERLAP	64	Token overlap between chunks
APP_PORT	7860	Gradio server port
LOG_LEVEL	INFO	Logging verbosity

________________________________________
🛠️ Development
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .

________________________________________
📊 Data Folder (Auto-generated)
The ./data/ folder is never committed to Git — it is created automatically by the ingestion script. Add it to .gitignore:
data/

________________________________________
⚠️ Windows Notes
•	The msvcrt warning from Qdrant on shutdown is harmless — a known Windows file-lock cleanup issue
•	The HuggingFace symlink warning is harmless — cache works in degraded mode using slightly more disk space
•	Use $env:PYTHONPATH = "." if you get ModuleNotFoundError: No module named 'src' (fixed permanently by uv pip install -e .)
________________________________________
📄 License
MIT