# 🩺 Neonatal RAG Chatbot

A **local-first** Retrieval-Augmented Generation (RAG) chatbot for neonatal clinical decision support. Combines Ollama LLMs with ChromaDB vector search and LangChain orchestration for privacy-preserving medical Q&A.

---

## 🏗️ Architecture
PubMed Agent (LangGraph)
↓ fetches + embeds 60 neonatal topics
ChromaDB Vector Store (local, persistent)
↓ semantic search (MMR retrieval)
LangChain RAG Chain
↓ generates grounded answer
Gradio Chat UI → http://localhost:7860

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| **LLM** | Ollama (local) | 100% private, no API cost, works offline |
| **Embeddings** | nomic-embed-text | Best open-source medical embeddings |
| **Vector Store** | ChromaDB | Local persistence, zero cloud dependency |
| **Orchestration** | LangChain + LangGraph | Industry standard, conversation memory |
| **Data Agent** | LangGraph + PubMed API | Auto-builds knowledge base from literature |
| **Full Text** | NCBI BioC API | Fetches open-access full papers (~40% of results) |
| **UI** | Gradio | Native ChatInterface, zero frontend code |
| **Config** | Pydantic Settings | Typed, validated, .env-based |
| **Database** | SQLite | Lightweight chat history persistence |
| **Package Manager** | uv | 10–100x faster than pip |
| **Linting** | ruff | Fastest Python linter |
| **Testing** | pytest + pytest-cov | Full test coverage |

---

## ⚡ Quick Start

### Prerequisites

```bash
# 1. Install Ollama from https://ollama.com/download

# 2. Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Installation

```bash
# Clone the repo
git clone https://github.com/youness-moukafih/neonatal-rag-chatbot
cd neonatal-rag-chatbot

# Create virtual environment
uv venv --python 3.12

# Activate it
.venv\Scripts\Activate.ps1    # Windows PowerShell
source .venv/bin/activate      # Mac / Linux

# Install all dependencies
uv sync

# Copy environment config
cp .env.example .env
```

### Build the Knowledge Base

```bash
# Fetch 60 neonatal topics from PubMed + PMC full text
# Run once — takes ~10 minutes, persists to data/chroma/
uv run python scripts/run_agent.py
```

### Launch the Chatbot

```bash
uv run python main.py
# Open your browser at http://localhost:7860
```



## 🧠 How It Works

### 1. Knowledge Building (run once)
The LangGraph agent automatically:
- Searches PubMed for **60 predefined neonatal topics**
- Fetches **full text** for Open Access papers via PMC BioC API (~40% of results)
- **Deduplicates** papers by PubMed UID across all topic runs
- **Chunks and embeds** everything into ChromaDB using `nomic-embed-text`

### 2. Question Answering (at runtime)
For every user question:
1. **MMR retrieval** fetches the top-5 most relevant chunks from ChromaDB
2. Retrieved context is injected into the **clinical system prompt**
3. Ollama LLM generates an answer **strictly grounded in the retrieved context**
4. Conversation history is passed for **multi-turn coherence**
5. The full exchange is **persisted to SQLite** for session review

---

## 🔧 Configuration

All configuration lives in `.env`:

```env
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
CHROMA_COLLECTION_NAME=neonatal_knowledge
TOP_K_RESULTS=5
CHUNK_SIZE=512
CHUNK_OVERLAP=64
APP_PORT=7860
LOG_LEVEL=INFO
```

---

## 🧪 Testing

```bash
# Run full test suite with coverage
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Run linter
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/
```

---

## 📊 Inspect Your Knowledge Base

```bash
uv run python scripts/inspect_db.py
```

Output example:
═══════════════════════════════════════════════════════
📊 ChromaDB Vector Store Report
═══════════════════════════════════════════════════════
Total chunks : 1842
Unique papers : 312

By source:
- PubMed 1842 chunks

By topic (top 5):
- preterm infant care oxygen therapy 142
- neonatal sepsis diagnosis treatment 128
- neonatal hypoxic ischemic encephalopathy 118


---

## ⚠️ Medical Disclaimer

This tool is intended to **support** clinical decision-making, not replace it. Always apply professional clinical judgment and follow your institution's protocols. Never use AI-generated medical information without verification from a qualified clinician.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Youness Moukafih**
Built with LangChain, LangGraph, ChromaDB, Ollama, and Gradio.