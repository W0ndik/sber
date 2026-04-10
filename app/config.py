from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "ilyagusev/saiga_llama3"
    ollama_embed_model: str = "bge-m3"

    qdrant_path: str = "./data/qdrant"
    qdrant_collection: str = "bank_support_kb"
    sqlite_path: str = "./data/app.db"
    knowledge_base_dir: str = "./knowledge_base"

    retrieval_limit: int = 5
    min_relevance_score: float = 0.35

    chat_keep_alive: str = "10m"
    embed_keep_alive: str = "10m"


@lru_cache
def get_settings() -> Settings:
    return Settings()