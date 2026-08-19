"""Chat history in Postgres."""
from __future__ import annotations

import uuid

from psycopg.types.json import Json

from app import db
from app.schemas import Source

#: Enough context for the rewriter to resolve "that" or "the second one",
#: without dragging a long conversation into every request.
#:
#: Messages, not turns: `LIMIT` counts rows, and a turn is two rows. Six here is
#: the last three exchanges. The name used to say "turns", which read as twice
#: the history the rewriter actually got.
HISTORY_MESSAGES = 6


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def derive_title(first_message: str) -> str:
    line = first_message.strip().split("\n")[0]
    return f"{line[:56]}…" if len(line) > 56 else (line or "New chat")


async def recent_turns(
    session_id: str, limit: int = HISTORY_MESSAGES
) -> list[tuple[str, str]]:
    rows = await db.fetch_all(
        """
        SELECT role, content FROM chat_message
        WHERE session_id = %s ORDER BY seq DESC LIMIT %s
        """,
        (session_id, limit),
    )
    return [(r[0], r[1]) for r in reversed(rows)]


async def start_turn(*, session_id: str, title: str, question: str) -> str:
    """Open a turn: create or touch the session, write the question.

    Written before retrieval rather than after generation, which is where the
    whole turn used to be written. Everything between those two points can
    fail — the LLM can error, the reader can press Stop, a laptop lid can
    close — and until this existed, all of it ended the same way: nothing was
    saved. The question the user typed was gone on reload, and if it was the
    first of a conversation the session itself never appeared in the sidebar,
    so there was no thread to go back to and no evidence the turn had ever
    happened.

    Returns the user message id.
    """
    user_id = new_id("m")

    async with db.pool().connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO chat_session (id, title) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET updated_at = now()
                """,
                (session_id, title),
            )
            await conn.execute(
                """
                INSERT INTO chat_message (id, session_id, role, content, meta)
                VALUES (%s, %s, 'user', %s, %s)
                """,
                (user_id, session_id, question, Json({})),
            )

    return user_id


async def finish_turn(
    *,
    session_id: str,
    answer: str,
    sources: list[Source],
    meta: dict,
) -> str:
    """Close a turn: write the answer and the citations it stands on.

    Separate transaction from `start_turn`, which is what lets a turn be
    persisted in the state it actually reached — including a half-written
    answer that was interrupted. Ordering does not depend on the split:
    `chat_message.seq` is assigned at insert time and the question is always
    inserted first.

    Returns the assistant message id.
    """
    assistant_id = new_id("m")

    async with db.pool().connection() as conn:
        async with conn.transaction():
            # The answer lands after the question, sometimes much later, and
            # the sidebar orders by this.
            await conn.execute(
                "UPDATE chat_session SET updated_at = now() WHERE id = %s",
                (session_id,),
            )
            await conn.execute(
                """
                INSERT INTO chat_message (id, session_id, role, content, meta)
                VALUES (%s, %s, 'assistant', %s, %s)
                """,
                (assistant_id, session_id, answer, Json(meta)),
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


async def save_turn(
    *,
    session_id: str,
    title: str,
    question: str,
    answer: str,
    sources: list[Source],
    meta: dict,
) -> str:
    """One finished question/answer pair, written in one call.

    The streaming endpoint does not use this — it opens the turn before it
    retrieves and closes it when the answer is whole, which is the point of
    the split above. This is for callers that already have both halves, and
    for the tests that exercise the read path.

    Returns the assistant message id.
    """
    await start_turn(session_id=session_id, title=title, question=question)
    return await finish_turn(
        session_id=session_id, answer=answer, sources=sources, meta=meta
    )


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
