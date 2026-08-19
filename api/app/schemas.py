"""Pydantic models — the server half of the contract in
`frontend/src/lib/types.ts`. Field names must match it exactly.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    #: Why this result set is the shape it is, when the shape needs explaining —
    #: currently only "the embedding model could not read the question". Null on
    #: a normal turn. A short result list is otherwise indistinguishable from a
    #: thin corpus, and the two have completely different fixes.
    notice: str | None = None


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


#: Strip before validating, so `min_length` sees the text and not the padding
#: around it. Without this a message of spaces or a newline satisfies
#: `min_length=1`, and the turn runs end to end: retrieval finds nothing, the
#: prompt says "(nothing was retrieved for this question)", the LLM is billed
#: for answering it, and an empty question is stored in the transcript. The
#: cost of a blank submit should be a 422, not a round trip.
_STRIPPED = ConfigDict(str_strip_whitespace=True)


class ChatRequest(BaseModel):
    model_config = _STRIPPED

    message: str = Field(min_length=1)
    session_id: str | None = None
    #: Bounded, unlike the bare `int | None` this used to be. `top_k` reaches a
    #: Python slice, where a negative number is a valid instruction to count
    #: from the end: `top_k=-1` sent nineteen chunks into the prompt instead of
    #: five, and `top_k=-100` sent none at all, so the model answered "there is
    #: nothing in the documents" over a corpus that had the answer. The ceiling
    #: is FUSION_KEEP, because nothing beyond it survives re-ranking to be sent.
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchRequest(BaseModel):
    model_config = _STRIPPED

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class GitHubIngestRequest(BaseModel):
    model_config = _STRIPPED

    #: "owner/name". The old pattern was "anything without a slash or space,
    #: twice", which accepted `../..` — and since the value is interpolated
    #: into an api.github.com path, that addressed the API root rather than a
    #: repository.
    #:
    #: This restricts the character set; it is deliberately not the whole
    #: rule. `GitHubConnector.__init__` is the authority — it also rejects a
    #: segment that is only dots — because the connector is reachable from the
    #: CLI and from tests without ever passing through this model, and a
    #: validation rule with two homes drifts. The endpoint turns the
    #: connector's `ValueError` into a 400.
    repo: str = Field(pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
    #: Branch, tag or SHA. Defaults to the repository's default branch.
    #: Slashes are allowed — `release/2026-08` is a branch — but `..` is not.
    ref: str | None = Field(default=None, max_length=255)
    #: Restrict to a subtree, e.g. "docs".
    path_prefix: str = Field(default="", max_length=255)
    force: bool = False


class IngestDocumentResult(BaseModel):
    path: str
    #: "created" | "updated" | "unchanged" | "failed"
    status: str
    chunks: int
    #: Why this one document could not be indexed, when it could not be. The
    #: rest of the batch is unaffected — a corrupt PDF among twelve files used
    #: to fail the whole request and index none of them.
    error: str | None = None


class IngestResponse(BaseModel):
    documents: int
    written: int
    unchanged: int
    failed: int = 0
    chunks: int
    chunk_budget: int
    results: list[IngestDocumentResult]


class IngestJob(BaseModel):
    """A unit of indexing work that outlives the request that asked for it."""

    id: str
    #: "files" | "github"
    source: str
    #: What is being indexed, readable in a list: "4 files", "pgvector/pgvector".
    label: str
    #: "running" | "done" | "failed"
    status: str
    #: Which stage a running job is in — reading, fetching, indexing.
    phase: str | None = None
    #: Null while unknown: a connector discovers its documents as it goes.
    total: int | None = None
    done: int = 0
    #: The document being worked on right now.
    current: str | None = None
    #: Present once the job finishes: exactly the payload the endpoint used to
    #: return synchronously.
    report: IngestResponse | None = None
    #: Set only when the job as a whole failed, never for a single document.
    error: str | None = None
    created_at: str
    updated_at: str


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


class SessionSummary(BaseModel):
    id: str
    title: str
    updated_at: str


class StoredMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] = []
    meta: dict | None = None


class SessionDetail(BaseModel):
    id: str
    title: str
    updated_at: str
    messages: list[StoredMessage]


class CorpusStats(BaseModel):
    documents: int
    chunks: int
    embed_model: str
    demo: bool = False


class Health(BaseModel):
    status: str
    database: bool
    pgvector: str | None = None
    #: False until the embedding and re-ranking models have finished loading.
    #: The first question after startup used to load them on the event loop,
    #: which stopped the whole server — this endpoint included — for as long as
    #: it took, with nothing to show that was what was happening.
    models_ready: bool = False
    llm_configured: bool
    chat_model: str
    embed_model: str
    embed_dim: int
    documents: int
    chunks: int
