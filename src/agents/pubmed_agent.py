import logging
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from src.agents.state import AgentState
from src.agents.tools.pubmed_tool import search_pubmed
from src.agents.tools.pmc_tool import enrich_with_fulltext
from src.agents.tools.ingest_tool import ingest_pubmed_results
from src.vectorstore.store import vectorstore_exists

logger = logging.getLogger(__name__)


# ─── Node 1: Validate ────────────────────────────────────────────────────────

def validate_input(state: AgentState) -> AgentState:
    """Check topic is non-empty. Vector store is created automatically if needed."""
    if not state.topic.strip():
        state.error = "Topic is empty."
        return state

    state.messages.append(
        AIMessage(content=f"🔍 Searching PubMed: '{state.topic}'")
    )
    return state


# ─── Node 2: Search PubMed ───────────────────────────────────────────────────

def fetch_from_pubmed(state: AgentState) -> AgentState:
    """Search PubMed and store results in state."""
    if state.error:
        return state

    results = search_pubmed.invoke({"query": state.topic})

    if not results:
        state.error = f"No PubMed results for: '{state.topic}'"
        return state

    state.raw_results = results
    state.papers_found = len(results)
    state.messages.append(
        AIMessage(content=f"📄 {len(results)} papers retrieved.")
    )
    return state


# ─── Node 3: Enrich With Full Text ───────────────────────────────────────────

def enrich_with_pmc_fulltext(state: AgentState) -> AgentState:
    """
    Try to replace abstracts with full text for Open Access papers via PMC.
    Non-OA papers silently fall back to abstract — no errors raised.
    """
    if state.error:
        return state

    enriched_results, fulltext_count = enrich_with_fulltext(state.raw_results)
    state.raw_results    = enriched_results
    state.fulltext_count = fulltext_count

    state.messages.append(
        AIMessage(
            content=(
                f"📚 Full text: {fulltext_count}/{state.papers_found} papers "
                f"had Open Access full text."
            )
        )
    )
    return state


# ─── Node 4: Ingest into ChromaDB ────────────────────────────────────────────

def ingest_into_vectorstore(state: AgentState) -> AgentState:
    """Deduplicate and embed enriched results into ChromaDB."""
    if state.error:
        return state

    try:
        added, skipped = ingest_pubmed_results(state.raw_results)
        state.added_count   = added
        state.skipped_count = skipped
        state.messages.append(
            AIMessage(
                content=(
                    f"✅ Added {added} chunks | "
                    f"Skipped {skipped} duplicates."
                )
            )
        )
    except Exception as e:
        state.error = f"Ingestion failed: {e}"
        logger.error("Ingestion error for '%s': %s", state.topic, e)

    return state


# ─── Node 5: Summarize ───────────────────────────────────────────────────────

def build_summary(state: AgentState) -> AgentState:
    """Produce the final summary string for this topic run."""
    if state.error:
        state.summary = f"⚠️  '{state.topic}' → {state.error}"
    else:
        top_titles = "\n".join(
            f"    {i+1}. {'[FULL]' if p.get('has_full_text') else '[ABS] '} "
            f"{p['title'][:65]} ({p['published']})"
            for i, p in enumerate(state.raw_results[:3])
        )
        state.summary = (
            f"✅ '{state.topic}'\n"
            f"   Papers: {state.papers_found} | "
            f"Full text: {state.fulltext_count} | "
            f"New chunks: {state.added_count} | "
            f"Skipped: {state.skipped_count}\n"
            f"   Top papers:\n{top_titles}"
        )

    state.messages.append(AIMessage(content=state.summary))
    return state


# ─── Routing ─────────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    """Skip remaining nodes and go straight to summary on error."""
    return "end" if state.error else "continue"


# ─── Build & Compile Graph ───────────────────────────────────────────────────

def build_pubmed_agent():
    """Compile and return the LangGraph PubMed ingestion agent."""
    graph = StateGraph(AgentState)

    graph.add_node("validate",  validate_input)
    graph.add_node("search",    fetch_from_pubmed)
    graph.add_node("enrich",    enrich_with_pmc_fulltext)    # ← NEW
    graph.add_node("ingest",    ingest_into_vectorstore)
    graph.add_node("summarize", build_summary)

    graph.add_edge(START, "validate")

    graph.add_conditional_edges(
        "validate",
        route,
        {"continue": "search", "end": "summarize"},
    )
    graph.add_conditional_edges(
        "search",
        route,
        {"continue": "enrich", "end": "summarize"},   # ← goes to enrich now
    )
    graph.add_conditional_edges(
        "enrich",
        route,
        {"continue": "ingest", "end": "summarize"},
    )
    graph.add_conditional_edges(
        "ingest",
        route,
        {"continue": "summarize", "end": "summarize"},
    )

    graph.add_edge("summarize", END)
    return graph.compile()


pubmed_agent = build_pubmed_agent()