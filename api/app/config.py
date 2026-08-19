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

    # --- Sources ---
    # Optional. Without it the GitHub API allows 60 requests an hour, which one
    # medium repository exhausts; with it, 5000.
    github_token: str = ""

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

    # --- Uploads ---
    # Per file, and across one request. `UploadFile.read()` with no argument
    # pulls the whole file into memory before anything is in a position to
    # object to its size, so without a ceiling one request decides how much
    # memory this process uses. Neither number is a judgement about what a
    # document should weigh — they are the point past which the answer is 413
    # rather than an out-of-memory kill.
    max_upload_bytes: int = 20 * 1024 * 1024
    max_upload_total_bytes: int = 100 * 1024 * 1024

    # --- Retrieval ---
    candidates_per_leg: int = 50  # from each of the vector and keyword legs
    fusion_keep: int = 20  # survivors of RRF that go to the re-ranker
    top_k: int = 5  # chunks that actually enter the prompt
    rrf_k: int = 60  # the constant in 1 / (k + rank)
    # HNSW trades recall for speed through this knob, and its default (40) gives
    # up more than a corpus this size can afford. Applied per connection.
    hnsw_ef_search: int = 100
    # The cross-encoder's relevance probability, below which retrieval does not
    # claim to have found anything relevant. It marks the result low-confidence;
    # it does not delete rows. That distinction is measured, not stylistic:
    #
    #   this corpus, 30 golden questions   weakest correct answer  0.00248
    #   12 questions it answers nothing of strongest wrong answer  0.00011
    #   "how do I start everything on my laptop?"
    #     -> "one compose command brings up ..."                   0.0000154
    #
    # The third line is the one that matters. It is a correct retrieval — a
    # paraphrase with no shared vocabulary, which is the case the vector leg
    # exists for — and the cross-encoder scores it *below* every off-topic
    # question in the set. ms-marco is bimodal: it recognises lexical overlap
    # at ~0.99 and reports everything else at ~1e-5, relevant or not. So no
    # threshold separates "irrelevant" from "relevant but reworded", and a
    # floor that dropped rows would trade confident wrong answers for silently
    # missing right ones — the worse of the two, because a dropped chunk
    # leaves nothing on screen to notice.
    #
    # Re-measure after changing RERANK_MODEL. This is a property of that
    # model's output scale, not a constant.
    min_rerank_score: float = 0.0005

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
