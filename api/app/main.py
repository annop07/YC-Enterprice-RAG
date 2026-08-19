"""FastAPI application: ingestion, hybrid retrieval and the streaming chat
endpoint, over one Postgres."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app import db
from app.chat import store
from app.chat.service import close_client, stream_chat
from app.config import get_settings
from app.ingest.embedder import get_embedder
from app.ingest import jobs
from app.ingest.github import GitHubConnector, GitHubError
from app.ingest.pipeline import DocumentResult, ingest
from app.ingest.uploads import document_from_upload, duplicate_names, upload_name
from app.retrieval.reranker import get_reranker
from app.retrieval.search import hybrid_search, notice_for
from app.schemas import (
    ChatRequest,
    CorpusStats,
    DocumentSummary,
    DocumentText,
    GitHubIngestRequest,
    Health,
    IngestJob,
    SearchRequest,
    SessionDetail,
    SessionSummary,
    SourcesEvent,
    StoredMessage,
)


log = logging.getLogger(__name__)

#: Whether the embedding and re-ranking models have finished loading. Read by
#: `/health`; see `_warm_models`.
_models_ready = False


async def _warm_models() -> None:
    """Load the ONNX models at startup, on a worker thread.

    Both getters are synchronous and take seconds the first time — longer if
    the model still has to be downloaded. Left to the first question they ran
    *on the event loop*, so the entire server stopped answering, `/health`
    included, for as long as it took, with nothing to say that was what was
    happening. Two questions arriving together loaded two copies.

    Started rather than awaited: the port opens immediately and `/health`
    answers honestly with `models_ready: false` while this runs. A question
    that arrives mid-load blocks on the same lock the warm-up holds and gets
    that instance rather than building a second one.
    """
    global _models_ready
    try:
        await asyncio.to_thread(get_embedder)
        await asyncio.to_thread(get_reranker)
        _models_ready = True
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — startup continues; the retry is on demand
        log.exception("model warm-up failed; the first question will try again")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.bootstrap()
    await db.open_pool()
    warm = asyncio.create_task(_warm_models())
    # A job's task lives in one process's event loop. Anything still marked
    # running belongs to a process that is gone, and would otherwise sit at
    # "indexing 12 of 40" for ever.
    if abandoned := await jobs.abandon_running():
        log.warning("closed %d ingest job(s) left running by a previous process", abandoned)
    try:
        yield
    finally:
        await jobs.cancel_all()
        warm.cancel()
        await asyncio.gather(warm, return_exceptions=True)
        # The LLM client holds an httpx pool of its own; the connections are
        # closed here rather than left for the interpreter to reclaim.
        await close_client()
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
        models_ready=_models_ready,
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




#: One megabyte at a time. Small enough that the limit below is enforced on the
#: way in rather than after the fact, large enough not to make a syscall per
#: kilobyte.
_UPLOAD_CHUNK = 1024 * 1024


async def _read_capped(upload: UploadFile, name: str, budget: int) -> bytes:
    """Read one upload, refusing as soon as it goes over `budget`.

    Chunked rather than a bare `await upload.read()`, which has already
    materialised the whole file — any file, at any size — in this process's
    memory by the time there is anything to check.
    """
    settings = get_settings()
    chunks: list[bytes] = []
    size = 0

    while data := await upload.read(_UPLOAD_CHUNK):
        size += len(data)
        if size > budget:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{name} is over the size limit for this request "
                    f"(MAX_UPLOAD_BYTES={settings.max_upload_bytes} per file, "
                    f"MAX_UPLOAD_TOTAL_BYTES={settings.max_upload_total_bytes} "
                    f"per request)"
                ),
            )
        chunks.append(data)

    return b"".join(chunks)


async def _settle(
    job_id: str, task: asyncio.Task, *, wait: bool, response: Response
) -> IngestJob:
    """Answer with the job — either as accepted, or as finished.

    The endpoints answer with a job because indexing outlasts a request. A
    script does not always want that — reproducing the evaluation corpus is
    one command in the README, and making it two so a browser can draw a
    progress bar is a bad trade. So the wait is available and opt-in, and it
    is the caller who accepts the timeout risk by asking for it.

    The status code follows what actually happened rather than the route's
    declared default: 202 says "accepted, not finished", which is a lie about
    a job that is finished by the time it is described.
    """
    if not wait:
        return await _job_or_404(job_id)

    await asyncio.gather(task, return_exceptions=True)
    response.status_code = 200
    return await _job_or_404(job_id)


async def _job_or_404(job_id: str) -> IngestJob:
    row = await jobs.get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return IngestJob(**row)


@app.post("/ingest/files", response_model=IngestJob, status_code=202)
async def ingest_files(
    files: list[UploadFile],
    response: Response,
    force: bool = False,
    wait: bool = False,
) -> IngestJob:
    """Index uploaded Markdown and PDF files.

    Answers with a job rather than a report: chunking and embedding a batch of
    PDFs takes longer than a browser or a proxy will hold a request open, and
    a request that times out mid-ingest leaves the work running with nobody
    holding its result. Poll `GET /jobs/{id}`, or pass `?wait=true` to have
    this call block until it is finished.
    """
    settings = get_settings()
    remaining = settings.max_upload_total_bytes

    # Before a single byte is read: two files that reduce to the same name are
    # one document to the corpus, so ingesting both would report two written
    # and keep the second. Refused here rather than resolved by guessing —
    # nothing in the request says which of the two the user meant to keep, and
    # the fix is theirs to make (rename one, or send them separately).
    if collisions := duplicate_names([u.filename or "upload" for u in files]):
        raise HTTPException(
            status_code=409,
            detail=(
                "two or more files in this request are named "
                f"{', '.join(collisions)} — a document is identified by its "
                "name, so indexing them together would keep only the last of "
                "each. Rename them or upload them separately."
            ),
        )

    # Read now, index later. An `UploadFile` is a handle on the request body
    # and stops existing when the response is sent, so the bytes have to be in
    # hand before the job can outlive the request that started it. The size
    # caps are what make that safe to hold.
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        name = upload_name(upload.filename or "upload")
        data = await _read_capped(upload, name, min(settings.max_upload_bytes, remaining))
        remaining -= len(data)
        payloads.append((name, data))

    label = f"{len(payloads)} file" + ("" if len(payloads) == 1 else "s")
    job_id = await jobs.create(
        source="files", label=label, total=len(payloads), phase="reading"
    )

    async def work(progress):
        # Extraction is per file and so is its failure: a PDF whose bytes are
        # not a PDF used to 415 the whole request and index none of the others.
        failed: list[DocumentResult] = []
        docs = []
        for name, data in payloads:
            try:
                # PDF text extraction is CPU work; keep it off the event loop.
                docs.append(await asyncio.to_thread(document_from_upload, name, data))
            except Exception as e:  # noqa: BLE001 — one file, not the batch
                failed.append(DocumentResult(name, "failed", error=str(e)))
                await jobs.advance(job_id, done=len(failed), current=name)

        await jobs.set_phase(job_id, "indexing")

        async def offset(result: DocumentResult, done: int) -> None:
            await progress(result, done + len(failed))

        report = await ingest(docs, force=force, on_progress=offset)
        report.results = failed + report.results
        return report

    task = jobs.start(job_id, work)
    return await _settle(job_id, task, wait=wait, response=response)


async def _known_blob_shas(repo: str) -> dict[str, str]:
    """path -> the git blob sha already indexed for it, for one repository.

    A blob sha is a hash of the file's contents and the tree listing hands one
    over per file, so a match means the connector can skip fetching that blob
    entirely — see `GitHubConnector.known_blob_shas`.
    """
    rows = await db.fetch_all(
        """
        SELECT source_id, meta->>'blob_sha'
        FROM document
        WHERE source_type = 'github'
          -- Compared, not pattern-matched. This used to be a LIKE against
          -- the repository name with a trailing wildcard, and an underscore
          -- is a LIKE wildcard of its own: `my_repo` also matched `myXrepo`,
          -- which could hand this repository another one's blob shas and so
          -- skip a fetch and leave the wrong text indexed. A repository name
          -- cannot contain an at sign, so everything before the first one is
          -- exactly the repository.
          --
          -- (Written without a per-cent sign on purpose: psycopg scans the
          -- whole statement for placeholders, comments included, so one in
          -- here is read as a malformed parameter and the query never runs.)
          AND split_part(source_id, '@', 1) = %s
          AND meta->>'blob_sha' IS NOT NULL
        """,
        (repo,),
    )
    return {source_id.split("@", 1)[1]: sha for source_id, sha in rows}


@app.post("/ingest/github", response_model=IngestJob, status_code=202)
async def ingest_github(
    request: GitHubIngestRequest, response: Response, wait: bool = False
) -> IngestJob:
    """Index the Markdown in a GitHub repository at one commit.

    A repository is one HTTP round trip per file before any of it is embedded,
    so this is the endpoint the job model exists for. The connector is built
    here rather than in the job: a malformed `repo` or `ref` is the caller's
    mistake and belongs in the response to their request, not in a job row
    they have to fetch to discover it.
    """
    try:
        connector = GitHubConnector(
            request.repo,
            ref=request.ref,
            path_prefix=request.path_prefix,
            token=get_settings().github_token or None,
            # `force` means re-read everything, so nothing is treated as known.
            known_blob_shas={} if request.force else await _known_blob_shas(request.repo),
        )
    except ValueError as e:
        # The connector owns the rules for `repo` and `ref`; a request that
        # breaks them is the caller's to fix, so it is a 400 naming the value
        # rather than the 500 an uncaught ValueError used to produce.
        raise HTTPException(status_code=400, detail=str(e)) from e

    label = request.repo + (f"/{request.path_prefix}" if request.path_prefix else "")
    job_id = await jobs.create(source="github", label=label, total=None, phase="fetching")

    async def work(progress):
        try:
            # httpx here is synchronous, and a repo is many round trips.
            docs = await asyncio.to_thread(lambda: list(connector.load()))
        except httpx.HTTPError as e:
            # `_get` maps everything it sees; this catches anything raised
            # outside it, so a transport failure reads as a GitHub problem
            # rather than as an unexplained crash.
            raise GitHubError(f"could not reach GitHub: {e}") from e

        # Only now is the count known — the tree listing is what discovers it.
        await jobs.set_phase(
            job_id, "indexing", total=len(docs) + len(connector.skipped)
        )
        report = await ingest(docs, force=request.force, on_progress=progress)
        # Files whose blob sha had not moved were never fetched, so the pipeline
        # never saw them. They are unchanged all the same, and leaving them out of
        # the report would read as "these documents disappeared from the repo".
        report.results.extend(
            DocumentResult(path, "unchanged") for path in connector.skipped
        )
        return report

    task = jobs.start(job_id, work)
    return await _settle(job_id, task, wait=wait, response=response)


@app.get("/jobs", response_model=list[IngestJob])
async def list_jobs(limit: int = 20) -> list[IngestJob]:
    return [IngestJob(**row) for row in await jobs.recent(min(max(limit, 1), 100))]


@app.get("/jobs/{job_id}", response_model=IngestJob)
async def job_detail(job_id: str) -> IngestJob:
    return await _job_or_404(job_id)


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
        notice=notice_for(result),
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


@app.delete("/documents/{document_id}", status_code=204)
async def remove_document(document_id: str) -> None:
    """Drop a document and its chunks.

    Answers that cited it keep their citations: `message_citation` holds a
    snapshot of the source, and its link to the chunk is nulled rather than
    cascaded away.
    """
    async with db.pool().connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM document WHERE id = %s", (document_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="not found")


@app.get("/documents/{document_id}", response_model=DocumentText)
async def document_text(document_id: str) -> DocumentText:
    row = await db.fetch_one(
        "SELECT id, path, text FROM document WHERE id = %s", (document_id,)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return DocumentText(id=row[0], path=row[1], text=row[2])
