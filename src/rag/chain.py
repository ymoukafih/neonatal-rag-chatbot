import logging

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from src.config.settings import get_settings
from src.rag.prompts import NEONATAL_SYSTEM_PROMPT
from src.vectorstore.store import get_retriever

logger = logging.getLogger(__name__)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("title", doc.metadata.get("source", "unknown"))
        parts.append(f"[{i}] {doc.page_content}\n(Source: {source})")
    return "\n\n".join(parts)


def get_llm() -> ChatOllama:
    """Return configured Ollama LLM instance."""
    settings = get_settings()
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
    )


def build_rag_chain():
    """Build and return the full RAG chain and retriever."""
    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", NEONATAL_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ])

    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def format_history(history: list[dict]) -> list:
    """Convert stored chat history dicts into LangChain message objects."""
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages