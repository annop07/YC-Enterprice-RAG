"""Pydantic models — the server half of the contract in
`frontend/src/lib/types.ts`. Field names must match it exactly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["markdown", "pdf", "github"]


class Locator(BaseModel):
    line_start: int | None = None
    line_end: int | None = None
    page: int | None = None


class RetrievalTrace(BaseModel):
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class Source(BaseModel):
    n: int
    chunk_id: str
    document_id: str
    title: str
    source_type: SourceType
    path: str
    url: str | None = None
    heading_path: str | None = None
    locator: Locator
    retrieval: RetrievalTrace
    snippet: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# --- SSE event payloads ---------------------------------------------------


class SessionEvent(BaseModel):
    session_id: str
    title: str


class SourcesEvent(BaseModel):
    sources: list[Source]
    candidates_considered: int
    retrieval_ms: int


class TokenEvent(BaseModel):
    text: str


class DoneEvent(BaseModel):
    message_id: str
    latency_ms: int
    usage: Usage
    dropped_citations: int
    model: str


class ErrorEvent(BaseModel):
    detail: str


# --- Requests / REST responses --------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    top_k: int | None = None


class DocumentSummary(BaseModel):
    id: str
    title: str
    path: str
    source_type: SourceType
    url: str | None = None
    chunk_count: int
    indexed_at: str


class DocumentText(BaseModel):
    id: str
    path: str
    text: str


class CorpusStats(BaseModel):
    documents: int
    chunks: int
    embed_model: str
    demo: bool = False


class Health(BaseModel):
    status: str
    database: bool
    pgvector: str | None = None
    llm_configured: bool
    chat_model: str
    embed_model: str
    embed_dim: int
    documents: int
    chunks: int
