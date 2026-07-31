"""Postgres access: schema bootstrap and a shared async connection pool."""
from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: AsyncConnectionPool | None = None


async def bootstrap() -> None:
    """Create the extension, tables and indexes if they are not there yet.

    Runs on a standalone connection rather than through the pool: the pool
    registers pgvector's type adapters on every connection, and that lookup
    fails until `CREATE EXTENSION vector` has actually run once.
    """
    settings = get_settings()
    sql = SCHEMA_PATH.read_text().replace("__EMBED_DIM__", str(settings.embed_dim))

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql)  # type: ignore[arg-type]
        await conn.commit()


async def open_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=8,
            configure=register_vector_async,
            open=False,
        )
        await _pool.open(wait=True, timeout=15)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("connection pool is not open — call open_pool() first")
    return _pool


async def fetch_one(sql: str, params: tuple | dict | None = None) -> tuple | None:
    async with pool().connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)  # type: ignore[arg-type]
        return await cur.fetchone()


async def fetch_all(sql: str, params: tuple | dict | None = None) -> list[tuple]:
    async with pool().connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)  # type: ignore[arg-type]
        return await cur.fetchall()


async def corpus_stats() -> dict[str, int]:
    row = await fetch_one(
        "SELECT (SELECT count(*) FROM document), (SELECT count(*) FROM chunk)"
    )
    documents, chunks = row if row else (0, 0)
    return {"documents": int(documents), "chunks": int(chunks)}
