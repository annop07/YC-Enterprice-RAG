"""FastAPI application: corpus endpoints today, retrieval and chat next."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import db
from app.chat import store
from app.chat.service import stream_chat
from app.config import get_settings
from app.ingest.github import GitHubConnector, GitHubError
from app.ingest.pipeline import IngestReport, ingest
from app.ingest.uploads import document_from_upload
from app.retrieval.search import hybrid_search
from app.schemas import (
    ChatRequest,
    CorpusStats,
    DocumentSummary,
    DocumentText,
    GitHubIngestRequest,
    Health,
    IngestDocumentResult,
    IngestResponse,
    SearchRequest,
    SessionDetail,
    SessionSummary,
    SourcesEvent,
    StoredMessage,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.bootstrap()
    await db.open_pool()
    try:
        yield
    finally:
        await db.close_pool()


app = FastAPI(
    title="Enterprise RAG API",
    description="Ingest internal documents, ask questions, get line-exact citations.",
    version="0.1.0",
    lifespan=lifespan,
)

# The Next.js app runs on its own origin and talks to this over plain HTTP.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3100",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3100",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=Health)
async def health() -> Health:
    settings = get_settings()
    database = True
    pgvector = None
    counts = {"documents": 0, "chunks": 0}

    try:
        row = await db.fetch_one(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        pgvector = row[0] if row else None
        counts = await db.corpus_stats()
    except Exception:  # noqa: BLE001 — health must report failure, not raise
        database = False

    return Health(
        status="ok" if database else "degraded",
        database=database,
        pgvector=pgvector,
        llm_configured=settings.llm_configured,
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        embed_dim=settings.embed_dim,
        **counts,
    )


@app.get("/corpus", response_model=CorpusStats)
async def corpus() -> CorpusStats:
    settings = get_settings()
    counts = await db.corpus_stats()
    return CorpusStats(embed_model=settings.embed_model, demo=False, **counts)


@app.get("/documents", response_model=list[DocumentSummary])
async def documents() -> list[DocumentSummary]:
    rows = await db.fetch_all(
        """
        SELECT d.id, d.title, d.path, d.source_type, d.url,
               count(c.id) AS chunk_count, d.indexed_at
        FROM document d
        LEFT JOIN chunk c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.path
        """
    )
    return [
        DocumentSummary(
            id=r[0],
            title=r[1],
            path=r[2],
            source_type=r[3],
            url=r[4],
            chunk_count=int(r[5]),
            indexed_at=r[6].isoformat(),
        )
        for r in rows
    ]


def _response(report: IngestReport) -> IngestResponse:
    return IngestResponse(
        documents=report.documents,
        written=report.written,
        unchanged=report.unchanged,
        chunks=report.chunks,
        chunk_budget=report.chunk_budget,
        results=[
            IngestDocumentResult(path=r.path, status=r.status, chunks=r.chunks)
            for r in report.results
        ],
    )


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(files: list[UploadFile], force: bool = False) -> IngestResponse:
    """Index uploaded Markdown and PDF files."""
    docs = []
    for upload in files:
        data = await upload.read()
        try:
            # PDF text extraction is CPU work; keep it off the event loop.
            docs.append(
                await asyncio.to_thread(
                    document_from_upload, upload.filename or "upload", data
                )
            )
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e)) from e

    return _response(await ingest(docs, force=force))


@app.post("/ingest/github", response_model=IngestResponse)
async def ingest_github(request: GitHubIngestRequest) -> IngestResponse:
    """Index the Markdown in a GitHub repository at one commit."""
    connector = GitHubConnector(
        request.repo,
        ref=request.ref,
        path_prefix=request.path_prefix,
        token=get_settings().github_token or None,
    )
    try:
        # httpx here is synchronous, and a repo is many round trips.
        docs = await asyncio.to_thread(lambda: list(connector.load()))
    except GitHubError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return _response(await ingest(docs, force=request.force))


@app.post("/search", response_model=SourcesEvent)
async def search(request: SearchRequest) -> SourcesEvent:
    """Hybrid retrieval on its own.

    Returns exactly the payload the chat stream sends as its `sources` event,
    so the retrieval layer can be inspected and evaluated without generating
    an answer over it.
    """
    result = await hybrid_search(request.query, top_k=request.top_k)
    return SourcesEvent(
        sources=result.sources,
        candidates_considered=result.candidates_considered,
        retrieval_ms=result.retrieval_ms,
    )


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Answer a question over the corpus, streamed as SSE.

    Events arrive as session → sources → token* → done, and the order is the
    point: sources are flushed before the first token so the citation cards are
    readable while the answer is still being written.
    """
    session_id = request.session_id or store.new_id("s")
    return StreamingResponse(
        stream_chat(
            question=request.message, session_id=session_id, top_k=request.top_k
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Without this nginx buffers the whole body and the answer arrives
            # in one lump. The stream still works, it just stops looking like one.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions", response_model=list[SessionSummary])
async def sessions() -> list[SessionSummary]:
    return [
        SessionSummary(id=i, title=t, updated_at=u) for i, t, u in await store.list_sessions()
    ]


@app.get("/sessions/{session_id}", response_model=SessionDetail)
async def session_detail(session_id: str) -> SessionDetail:
    row = await db.fetch_one(
        "SELECT id, title, updated_at FROM chat_session WHERE id = %s", (session_id,)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="not found")

    return SessionDetail(
        id=row[0],
        title=row[1],
        updated_at=row[2].isoformat(),
        messages=[StoredMessage(**m) for m in await store.session_messages(session_id)],
    )


@app.delete("/sessions/{session_id}", status_code=204)
async def remove_session(session_id: str) -> None:
    if not await store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="not found")


@app.get("/documents/{document_id}", response_model=DocumentText)
async def document_text(document_id: str) -> DocumentText:
    row = await db.fetch_one(
        "SELECT id, path, text FROM document WHERE id = %s", (document_id,)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return DocumentText(id=row[0], path=row[1], text=row[2])
