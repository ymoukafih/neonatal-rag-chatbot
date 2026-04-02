from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — Ollama (local)
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # NCBI / PubMed
    ncbi_api_key: str = ""
    pubmed_max_results: int = 500
    pubmed_requests_per_sec: float = 8.0

    # Vector store
    qdrant_path: str = "./data/qdrant"
    collection_name: str = "neonatal_kb"

    # Checkpointing
    checkpoint_file: str = "./data/pubmed_checkpoint.json"
    output_file: str = "./data/output.json"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 7860
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()