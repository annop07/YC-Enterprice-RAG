"""Chat history in Postgres."""
from __future__ import annotations

import uuid

from psycopg.types.json import Json

from app import db
from app.schemas import Source

#: Enough context for the rewriter to resolve "that" or "the second one",
#: without dragging a long conversation into every request.
HISTORY_TURNS = 6


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def derive_title(first_message: str) -> str:
    line = first_message.strip().split("\n")[0]
    return f"{line[:56]}…" if len(line) > 56 else (line or "New chat")


async def recent_turns(session_id: str, limit: int = HISTORY_TURNS) -> list[tuple[str, str]]:
    rows = await db.fetch_all(
        """
        SELECT role, content FROM chat_message
        WHERE session_id = %s ORDER BY seq DESC LIMIT %s
        """,
        (session_id, limit),
    )
    return [(r[0], r[1]) for r in reversed(rows)]


async def save_turn(
    *,
    session_id: str,
    title: str,
    question: str,
    answer: str,
    sources: list[Source],
    meta: dict,
) -> str:
    """Persist one question/answer pair and its citations. Returns message id."""
    user_id = new_id("m")
    assistant_id = new_id("m")

    async with db.pool().connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO chat_session (id, title) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET updated_at = now()
                """,
                (session_id, title),
            )
            await conn.cursor().executemany(
                """
                INSERT INTO chat_message (id, session_id, role, content, meta)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (user_id, session_id, "user", question, Json({})),
                    (assistant_id, session_id, "assistant", answer, Json(meta)),
                ],
            )
            if sources:
                await conn.cursor().executemany(
                    """
                    INSERT INTO message_citation (message_id, n, chunk_id, source)
                    -- The subquery resolves to NULL when the chunk is already
                    -- gone. A re-index that lands between retrieval and this
                    -- write would otherwise fail the foreign key and take the
                    -- whole answer down with it — over a link that is only a
                    -- convenience, since `source` below is the durable record.
                    VALUES (%s, %s, (SELECT id FROM chunk WHERE id = %s), %s)
                    """,
                    [
                        (
                            assistant_id,
                            source.n,
                            # Soft reference: re-indexing the document replaces
                            # this chunk, and the answer must not lose its
                            # citation because of it. The snapshot below is what
                            # history actually renders from.
                            int(source.chunk_id) if source.chunk_id.isdigit() else None,
                            Json(source.model_dump()),
                        )
                        for source in sources
                    ],
                )

    return assistant_id


async def list_sessions() -> list[tuple[str, str, str]]:
    rows = await db.fetch_all(
        "SELECT id, title, updated_at FROM chat_session ORDER BY updated_at DESC LIMIT 100"
    )
    return [(r[0], r[1], r[2].isoformat()) for r in rows]


async def session_messages(session_id: str) -> list[dict]:
    rows = await db.fetch_all(
        """
        SELECT m.id, m.role, m.content, m.meta,
               COALESCE(
                   jsonb_agg(c.source ORDER BY c.n) FILTER (WHERE c.source IS NOT NULL),
                   '[]'::jsonb
               ) AS sources
        FROM chat_message m
        LEFT JOIN message_citation c ON c.message_id = m.id
        WHERE m.session_id = %s
        GROUP BY m.id
        ORDER BY m.seq
        """,
        (session_id,),
    )
    return [
        {"id": r[0], "role": r[1], "content": r[2], "meta": r[3] or None, "sources": r[4]}
        for r in rows
    ]


async def delete_session(session_id: str) -> bool:
    async with db.pool().connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM chat_session WHERE id = %s", (session_id,)
        )
        return cursor.rowcount > 0
