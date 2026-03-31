from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_temperature: float = Field(default=0.1)

    # Embeddings
    embedding_model: str = Field(default="nomic-embed-text")

    # Vector Store
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="neonatal_knowledge")

    # Data
    data_path: str = Field(default="./data/output.json")

    # Database
    database_url: str = Field(default="./data/chatbot.db")

    # App
    app_name: str = Field(default="Neonatal RAG Chatbot")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=7860)
    log_level: str = Field(default="INFO")

    # RAG
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    top_k_results: int = Field(default=5)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance — reads .env only once."""
    return Settings()