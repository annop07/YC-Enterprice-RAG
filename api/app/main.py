"""FastAPI application: corpus endpoints today, retrieval and chat next."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.config import get_settings
from app.schemas import CorpusStats, DocumentSummary, DocumentText, Health


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


@app.get("/documents/{document_id}", response_model=DocumentText)
async def document_text(document_id: str) -> DocumentText:
    row = await db.fetch_one(
        "SELECT id, path, text FROM document WHERE id = %s", (document_id,)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return DocumentText(id=row[0], path=row[1], text=row[2])
