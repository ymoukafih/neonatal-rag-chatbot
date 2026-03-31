from dataclasses import dataclass, field
from typing import Annotated
from langgraph.graph.message import add_messages


@dataclass
class AgentState:
    """State that flows through every node of the PubMed agent graph."""

    # The neonatal topic being processed
    topic: str = ""

    # Raw paper dicts returned from PubMed (enriched with full text if available)
    raw_results: list[dict] = field(default_factory=list)

    # New chunks added to ChromaDB in this run
    added_count: int = 0

    # Duplicate chunks skipped
    skipped_count: int = 0

    # Total papers fetched from PubMed
    papers_found: int = 0

    # How many papers got full text from PMC Open Access
    fulltext_count: int = 0

    # Final human-readable summary for this topic
    summary: str = ""

    # Error message — empty string means no error
    error: str = ""

    # LangGraph message log (supports streaming)
    messages: Annotated[list, add_messages] = field(default_factory=list)