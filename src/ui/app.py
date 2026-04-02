import logging
import gradio as gr

from src.config.settings import get_settings
from src.database.crud import create_session, save_message
from src.rag.chain import build_rag_chain, format_history
from src.vectorstore.store import vectorstore_exists


logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)


if not vectorstore_exists():
    raise RuntimeError(
        "Vector store not found. Run ingestion first:\n"
        "  python scripts/run_agent.py"
    )


def launch() -> None:
    """Build and launch the Gradio interface."""
    settings = get_settings()
    session_id = create_session()

    rag_chain, retriever = build_rag_chain()

    def chat(message: str, history: list, session_id: str):
        """Process one user message and return the assistant response."""
        if not message or not message.strip():
            return "", history

        history = history or []

        # Convert Gradio chat history into LangChain history
        flat_history = []
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                flat_history.append(
                    {"role": msg["role"], "content": str(msg["content"])}
                )

        chat_history = format_history(flat_history)

        source_docs = retriever.invoke(message)
        sources = [
            {"content": doc.page_content[:200], "metadata": doc.metadata}
            for doc in source_docs
        ]

        response = rag_chain.invoke({
            "question": message,
            "chat_history": chat_history,
        })

        save_message(session_id, "user", message)
        save_message(session_id, "assistant", response)

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

        return "", history

    with gr.Blocks(title=settings.app_name) as demo:
        gr.Markdown(f"# 🩺 {settings.app_name}")
        gr.Markdown(
            "Ask questions about neonatal care based on our clinical knowledge base."
        )

        session_state = gr.State(session_id)

        chatbot = gr.Chatbot(
            height=500,
            value=[],
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask a neonatal care question...",
                label="Your question",
                scale=9,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            new_session_btn = gr.Button("➕ New Session", variant="secondary")

        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, session_state],
            outputs=[msg_input, chatbot],
        )

        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, session_state],
            outputs=[msg_input, chatbot],
        )

        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot],
        )

        new_session_btn.click(
            fn=lambda: (create_session(), []),
            outputs=[session_state, chatbot],
        )

    demo.launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    launch()