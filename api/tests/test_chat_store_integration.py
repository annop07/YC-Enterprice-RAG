"""Chat persistence against a real Postgres.

No LLM here — `save_turn` and the readers are what these exercise. The
generation path is covered by the prompt tests and by running it for real.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from app import db
from app.chat import store
from app.schemas import Locator, RetrievalTrace, Source
from tests.conftest import requires_db

pytestmark = [requires_db, pytest.mark.asyncio]


def source(n: int) -> Source:
    return Source(
        n=n,
        chunk_id=str(1000 + n),
        document_id="doc-1",
        title="Ingestion Pipeline",
        source_type="markdown",
        path="docs/ingestion.md",
        url=None,
        heading_path="Ingestion Pipeline > Chunking",
        locator=Locator(line_start=18, line_end=24, page=None),
        retrieval=RetrievalTrace(vector_rank=n, keyword_rank=None, rrf_score=0.03),
        snippet=f"snippet {n}",
    )


@pytest_asyncio.fixture(loop_scope="session")
async def session_id(pool):
    sid = store.new_id("s")
    yield sid
    async with db.pool().connection() as conn:
        await conn.execute("DELETE FROM chat_session WHERE id = %s", (sid,))


async def test_a_transcript_comes_back_in_the_order_it_was_written(session_id):
    """Regression: the answer came back above the question.

    Both rows are inserted in one transaction, and `now()` is transaction time
    — so `created_at` is identical for the pair and cannot order them. The
    tiebreaker was the id, which is random. `seq` is the fix.
    """
    await store.save_turn(
        session_id=session_id,
        title="Chunking",
        question="What chunk size?",
        answer="400 tokens [1].",
        sources=[source(1)],
        meta={"model": "test"},
    )
    await store.save_turn(
        session_id=session_id,
        title="Chunking",
        question="And the overlap?",
        answer="80 tokens [1].",
        sources=[source(1)],
        meta={"model": "test"},
    )

    messages = await store.session_messages(session_id)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "What chunk size?"
    assert messages[2]["content"] == "And the overlap?"


async def test_a_citation_survives_the_chunk_it_points_at_disappearing(session_id):
    """A re-index between retrieval and this write must not fail the turn.

    The chunk link is a convenience; the stored source payload is the record
    history renders from, so an unresolvable link becomes NULL rather than a
    foreign key violation that loses the whole answer.
    """
    await store.save_turn(
        session_id=session_id,
        title="t",
        question="q",
        answer="a [1]",
        sources=[source(1)],  # chunk_id 1001 — deliberately not in `chunk`
        meta={},
    )

    row = await db.fetch_one(
        """
        SELECT c.chunk_id, c.source->>'path'
        FROM message_citation c JOIN chat_message m ON m.id = c.message_id
        WHERE m.session_id = %s
        """,
        (session_id,),
    )
    assert row[0] is None, "the dangling link is nulled"
    assert row[1] == "docs/ingestion.md", "the snapshot still renders"


async def test_citations_are_stored_with_the_answer_and_come_back_whole(session_id):
    await store.save_turn(
        session_id=session_id,
        title="Chunking",
        question="q",
        answer="a [1][2].",
        sources=[source(1), source(2)],
        meta={"model": "test", "dropped_citations": 0},
    )

    assistant = (await store.session_messages(session_id))[-1]
    assert [s["n"] for s in assistant["sources"]] == [1, 2]
    assert assistant["sources"][0]["locator"]["line_start"] == 18
    assert assistant["sources"][0]["retrieval"]["vector_rank"] == 1
    assert assistant["meta"]["model"] == "test"


async def test_history_is_oldest_first_and_bounded(session_id):
    """The rewriter needs turns in order; a long chat must not be sent whole."""
    for i in range(5):
        await store.save_turn(
            session_id=session_id,
            title="t",
            question=f"question {i}",
            answer=f"answer {i}",
            sources=[],
            meta={},
        )

    turns = await store.recent_turns(session_id, limit=4)
    assert len(turns) == 4
    assert turns == [
        ("user", "question 3"),
        ("assistant", "answer 3"),
        ("user", "question 4"),
        ("assistant", "answer 4"),
    ]


async def test_deleting_a_session_takes_its_messages_and_citations_with_it(session_id):
    await store.save_turn(
        session_id=session_id,
        title="t",
        question="q",
        answer="a [1]",
        sources=[source(1)],
        meta={},
    )

    assert await store.delete_session(session_id) is True
    assert await store.session_messages(session_id) == []
    orphans = await db.fetch_one(
        """
        SELECT count(*) FROM message_citation c
        LEFT JOIN chat_message m ON m.id = c.message_id WHERE m.id IS NULL
        """
    )
    assert orphans[0] == 0
    assert await store.delete_session(session_id) is False


async def test_titles_are_derived_from_the_opening_question():
    assert store.derive_title("What chunk size?\nmore") == "What chunk size?"
    assert store.derive_title("x" * 80).endswith("…")
    assert store.derive_title("   ") == "New chat"
