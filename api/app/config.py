"""Application configuration loaded from environment variables.

Uses pydantic-settings so every config value is validated and typed.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (OpenAI-compatible; KKU IntelSphere in this workspace) ---
    openai_api_key: str = ""
    openai_base_url: str | None = None

    # Measured on the KKU proxy: qwen3.7-max is a reasoning model and takes
    # ~18s before it emits a single visible token, which is unusable for a
    # streaming chat UI. This one starts streaming in under a second.
    # claude-haiku-4.5 is the documented alternative.
    chat_model: str = "qwen3-next-80b-a3b-instruct"

    # --- Storage ---
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"

    # --- Embeddings ---
    # Default is the fast English pair so `docker compose up` is quick. For a
    # Thai or mixed-language corpus switch to intfloat/multilingual-e5-large
    # (dim 1024) — see the README; it is a re-index, not a config edit.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Chunking ---
    # Not the 500-1000 the brief asks for: every multilingual model fastembed
    # offers caps at 512 input tokens, and anything past that is silently
    # truncated rather than rejected. 400 leaves room for the title + heading
    # prefix that gets prepended to the embedding input.
    chunk_tokens: int = 400
    chunk_overlap: int = 80

    # --- Retrieval ---
    candidates_per_leg: int = 50  # from each of the vector and keyword legs
    fusion_keep: int = 20  # survivors of RRF that go to the re-ranker
    top_k: int = 5  # chunks that actually enter the prompt
    rrf_k: int = 60  # the constant in 1 / (k + rank)

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
