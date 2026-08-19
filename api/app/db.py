"""Postgres access: schema bootstrap and a shared async connection pool."""
from __future__ import annotations

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: AsyncConnectionPool | None = None


#: pgvector stores the declared width in `atttypmod` directly — 384 for
#: `vector(384)` — and -1 for an unconstrained `vector`.
EMBEDDING_DIM_SQL = """
SELECT a.atttypmod
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relname = 'chunk'
  AND a.attname = 'embedding'
  AND a.attnum > 0
  AND NOT a.attisdropped
  AND n.nspname = current_schema()
"""


class SchemaMismatch(RuntimeError):
    """The database was built for a different embedding model."""


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
            await cur.execute(EMBEDDING_DIM_SQL)
            row = await cur.fetchone()
        await conn.commit()

    # `CREATE TABLE IF NOT EXISTS` is a no-op against an existing table, so a
    # database built for a 384-dimension model stays 384 after EMBED_MODEL is
    # changed to a 1024-dimension one — and startup succeeds, and `get_embedder`
    # succeeds because it only checks the model against settings. The failure
    # then surfaces on the first insert or search, as a pgvector type error that
    # says nothing about what to do. Say it here instead, once, at startup.
    if row is not None and row[0] != -1 and row[0] != settings.embed_dim:
        raise SchemaMismatch(
            f"EMBED_DIM={settings.embed_dim} but this database's "
            f"chunk.embedding column is vector({row[0]}). The column width is "
            f"fixed when the table is created, so switching embedding models is "
            f"a re-index, not a config change:\n"
            f"    psql \"$DATABASE_URL\" -c 'DROP TABLE chunk, document CASCADE'\n"
            f"then start the API again and re-ingest. Answers already in the "
            f"transcript keep their citations — `message_citation` holds its own "
            f"snapshot of every source."
        )


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)
    # Session-level, so it is set once per pooled connection rather than on
    # every query. Interpolated because Postgres does not take parameters in
    # SET; the value is an int from settings, not user input.
    await conn.execute(f"SET hnsw.ef_search = {int(get_settings().hnsw_ef_search)}")
    # The pool requires the callback to hand the connection back outside a
    # transaction. Without this commit every connection is discarded as
    # "left in status INTRANS" and the pool never fills — it just times out.
    await conn.commit()


async def open_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=8,
            configure=_configure,
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
