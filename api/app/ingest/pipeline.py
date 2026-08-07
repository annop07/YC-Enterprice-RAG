"""Connector output -> chunks -> embeddings -> Postgres."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Iterable

from psycopg.types.json import Json

from app import db
from app.config import get_settings
from app.ingest.chunker import build_embed_input, chunk_markdown, chunk_pages
from app.ingest.connectors import RawDocument
from app.ingest.embedder import get_embedder

log = logging.getLogger(__name__)

#: Room left for the title/heading breadcrumb that `build_embed_input` puts in
#: front of every chunk, plus the tokenizer's own [CLS]/[SEP].
PREFIX_RESERVE_TOKENS = 64


@dataclass
class DocumentResult:
    path: str
    status: str  # "created" | "updated" | "unchanged"
    chunks: int = 0


@dataclass
class IngestReport:
    results: list[DocumentResult] = field(default_factory=list)
    chunk_budget: int = 0

    @property
    def documents(self) -> int:
        return len(self.results)

    @property
    def written(self) -> int:
        return sum(1 for r in self.results if r.status != "unchanged")

    @property
    def unchanged(self) -> int:
        return sum(1 for r in self.results if r.status == "unchanged")

    @property
    def chunks(self) -> int:
        return sum(r.chunks for r in self.results)


def effective_chunk_budget() -> int:
    """Chunk size the model can actually take, not the one that was asked for.

    CHUNK_TOKENS is a request; the embedding model's input ceiling is the law.
    Exceeding it does not raise — the tail is dropped during encoding — so the
    budget is clamped here and the clamp is logged rather than left implicit.
    """
    settings = get_settings()
    ceiling = get_embedder().max_input_tokens - PREFIX_RESERVE_TOKENS
    if settings.chunk_tokens > ceiling:
        log.warning(
            "CHUNK_TOKENS=%d exceeds what %s can encode; clamped to %d "
            "(model max %d, %d reserved for the heading prefix)",
            settings.chunk_tokens,
            settings.embed_model,
            ceiling,
            get_embedder().max_input_tokens,
            PREFIX_RESERVE_TOKENS,
        )
        return ceiling
    return settings.chunk_tokens


async def ingest(docs: Iterable[RawDocument], *, force: bool = False) -> IngestReport:
    settings = get_settings()
    embedder = get_embedder()
    budget = effective_chunk_budget()
    report = IngestReport(chunk_budget=budget)

    for doc in docs:
        existing = await db.fetch_one(
            "SELECT content_hash FROM document WHERE id = %s", (doc.id,)
        )
        if existing and existing[0] == doc.content_hash and not force:
            report.results.append(DocumentResult(doc.path, "unchanged"))
            continue

        split = dict(
            max_tokens=budget,
            overlap_tokens=settings.chunk_overlap,
            count_tokens=embedder.count_tokens,
            token_offsets=embedder.token_offsets,
        )
        # A paginated source chunks per page so every chunk can name one page.
        chunks = (
            chunk_pages(doc.pages, **split)
            if doc.pages is not None
            else chunk_markdown(doc.text, **split)
        )
        if not chunks:
            report.results.append(DocumentResult(doc.path, "unchanged"))
            continue

        embed_inputs = [
            build_embed_input(doc.title, c.heading_path, c.content) for c in chunks
        ]
        # fastembed is synchronous CPU work; off the event loop so the API can
        # keep serving while a large ingest runs.
        vectors = await asyncio.to_thread(embedder.embed_passages, embed_inputs)

        async with db.pool().connection() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO document
                        (id, source_type, source_id, title, path, url,
                         content_hash, text, meta, indexed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET
                        title = EXCLUDED.title,
                        path = EXCLUDED.path,
                        url = EXCLUDED.url,
                        content_hash = EXCLUDED.content_hash,
                        text = EXCLUDED.text,
                        meta = EXCLUDED.meta,
                        indexed_at = now()
                    """,
                    (
                        doc.id,
                        doc.source_type,
                        doc.source_id,
                        doc.title,
                        doc.path,
                        doc.url,
                        doc.content_hash,
                        doc.text,
                        Json(doc.meta),
                    ),
                )
                # Delete-then-insert inside one transaction: a partially
                # re-indexed document is never visible to a search.
                await conn.execute("DELETE FROM chunk WHERE document_id = %s", (doc.id,))
                await conn.cursor().executemany(
                    """
                    INSERT INTO chunk
                        (document_id, ordinal, content, embed_input, heading_path,
                         line_start, line_end, page, token_count, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            doc.id,
                            c.ordinal,
                            c.content,
                            embed_input,
                            c.heading_path,
                            c.line_start,
                            c.line_end,
                            c.page,
                            c.token_count,
                            vector,
                        )
                        for c, embed_input, vector in zip(chunks, embed_inputs, vectors)
                    ],
                )

        report.results.append(
            DocumentResult(doc.path, "updated" if existing else "created", len(chunks))
        )

    return report
