"""Ingestion as a background job.

Indexing is slow in a way that a request cannot politely absorb: a repository
costs one HTTP round trip per file, and every chunk is embedded on the CPU. It
used to run inside the request that asked for it, which had three consequences
and no upside — the caller could only be told "working" until it was over, a
proxy or a browser was free to time the request out and leave the work running
with nobody holding its result, and closing the tab killed it half way.

So the request now writes a row, starts a task, and hands back an id. The row
is the thing that knows how far it has got, and it outlives both the request
and the reader.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Awaitable, Callable

from psycopg.types.json import Json

from app import db
from app.ingest.pipeline import DocumentResult, IngestReport
from app.schemas import IngestDocumentResult, IngestResponse

log = logging.getLogger(__name__)

#: Live tasks, held so the event loop does not collect a job mid-run —
#: `asyncio` keeps only a weak reference to a task nobody is awaiting, and a
#: garbage-collected one simply stops.
_tasks: set[asyncio.Task] = set()


def new_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


def response(report: IngestReport) -> IngestResponse:
    """The wire shape of a finished report."""
    return IngestResponse(
        documents=report.documents,
        written=report.written,
        unchanged=report.unchanged,
        failed=report.failed,
        chunks=report.chunks,
        chunk_budget=report.chunk_budget,
        results=[
            IngestDocumentResult(
                path=r.path, status=r.status, chunks=r.chunks, error=r.error
            )
            for r in report.results
        ],
    )


# --- the row --------------------------------------------------------------


async def create(*, source: str, label: str, total: int | None, phase: str) -> str:
    job_id = new_id()
    async with db.pool().connection() as conn:
        await conn.execute(
            """
            INSERT INTO ingest_job (id, source, label, status, phase, total)
            VALUES (%s, %s, %s, 'running', %s, %s)
            """,
            (job_id, source, label, phase, total),
        )
    return job_id


async def set_phase(job_id: str, phase: str, *, total: int | None = None) -> None:
    async with db.pool().connection() as conn:
        await conn.execute(
            """
            UPDATE ingest_job
               SET phase = %s,
                   total = COALESCE(%s, total),
                   updated_at = now()
             WHERE id = %s
            """,
            (phase, total, job_id),
        )


async def advance(job_id: str, *, done: int, current: str) -> None:
    async with db.pool().connection() as conn:
        await conn.execute(
            "UPDATE ingest_job SET done = %s, current = %s, updated_at = now() WHERE id = %s",
            (done, current, job_id),
        )


async def finish(job_id: str, report: IngestReport) -> None:
    async with db.pool().connection() as conn:
        await conn.execute(
            """
            UPDATE ingest_job
               SET status = 'done', phase = NULL, current = NULL,
                   done = %s, total = %s,
                   report = %s, updated_at = now()
             WHERE id = %s
            """,
            (
                report.documents,
                report.documents,
                Json(response(report).model_dump()),
                job_id,
            ),
        )


async def fail(job_id: str, error: str) -> None:
    """The job as a whole could not run.

    Not the same as a document that could not be read: that is one `failed`
    row inside the report, and the rest of the batch is still indexed.
    """
    async with db.pool().connection() as conn:
        await conn.execute(
            """
            UPDATE ingest_job
               SET status = 'failed', phase = NULL, current = NULL,
                   error = %s, updated_at = now()
             WHERE id = %s
            """,
            (error[:2000], job_id),
        )


COLUMNS = (
    "id, source, label, status, phase, total, done, current, report, error, "
    "created_at, updated_at"
)


def _row(r: tuple) -> dict:
    return {
        "id": r[0],
        "source": r[1],
        "label": r[2],
        "status": r[3],
        "phase": r[4],
        "total": r[5],
        "done": r[6],
        "current": r[7],
        "report": r[8],
        "error": r[9],
        "created_at": r[10].isoformat(),
        "updated_at": r[11].isoformat(),
    }


async def get(job_id: str) -> dict | None:
    row = await db.fetch_one(f"SELECT {COLUMNS} FROM ingest_job WHERE id = %s", (job_id,))
    return _row(row) if row else None


async def recent(limit: int = 20) -> list[dict]:
    rows = await db.fetch_all(
        f"SELECT {COLUMNS} FROM ingest_job ORDER BY created_at DESC LIMIT %s", (limit,)
    )
    return [_row(r) for r in rows]


async def abandon_running() -> int:
    """Close out jobs left `running` by a process that is no longer here.

    A job's task lives in one process's event loop and nothing survives its
    restart, so a `running` row at startup is a job that will never move again.
    Left alone it would sit at "indexing 12 of 40" forever, which reads as
    working rather than as gone.
    """
    async with db.pool().connection() as conn:
        cursor = await conn.execute(
            """
            UPDATE ingest_job
               SET status = 'failed', phase = NULL, current = NULL,
                   error = 'the API restarted while this job was running',
                   updated_at = now()
             WHERE status = 'running'
            """
        )
        return cursor.rowcount


# --- running it -----------------------------------------------------------

#: Given a progress reporter, do the work and return what happened.
Work = Callable[[Callable[[DocumentResult, int], Awaitable[None]]], Awaitable[IngestReport]]


def progress_for(job_id: str) -> Callable[[DocumentResult, int], Awaitable[None]]:
    async def report(result: DocumentResult, done: int) -> None:
        await advance(job_id, done=done, current=result.path)

    return report


async def _run(job_id: str, work: Work) -> None:
    try:
        await finish(job_id, await work(progress_for(job_id)))
    except asyncio.CancelledError:
        # The API is shutting down. Say so in the row rather than leaving it
        # `running` for `abandon_running` to find on the way back up.
        await fail(job_id, "the API shut down while this job was running")
        raise
    except Exception as e:  # noqa: BLE001 — the row is where a job reports failure
        log.exception("ingest job %s failed", job_id)
        await fail(job_id, f"{type(e).__name__}: {e}")


def start(job_id: str, work: Work) -> asyncio.Task:
    task = asyncio.create_task(_run(job_id, work))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


async def cancel_all() -> None:
    """Stop every running job. Called from the application lifespan."""
    for task in list(_tasks):
        task.cancel()
    if _tasks:
        await asyncio.gather(*list(_tasks), return_exceptions=True)
