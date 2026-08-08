"""What happens when the models cannot read the question.

`BAAI/bge-small-en-v1.5` and `Xenova/ms-marco-MiniLM-L-6-v2` are both
English-only, and both fail silently on text they cannot tokenise: the embedder
maps every Thai question to the same vector, and the cross-encoder then hands
0.99 to whatever that vector happened to land on. Measured on the development
corpus, two unrelated Thai questions returned an identical top five headed by
the same irrelevant chunk at 0.9905.

These pin the fix — the blind stages are switched off and the keyword leg is
left to carry the query — and, just as importantly, pin that English and mixed
questions are untouched by it.
"""
from __future__ import annotations

import json

import pytest
import pytest_asyncio

from app import db
from app.chat import service, store
from app.ingest.connectors import RawDocument
from app.ingest.pipeline import ingest
from app.retrieval import search
from app.retrieval.search import hybrid_search
from tests.conftest import requires_db

pytestmark = [requires_db, pytest.mark.asyncio]

THAI_QUESTION = "ใครเป็นคนดูแลระบบนี้"

CORPUS = [
    RawDocument(
        source_type="markdown",
        source_id="test/tuning.md",
        title="Tuning",
        path="test/tuning.md",
        text=(
            "# Tuning\n\n"
            "## Index parameters\n\n"
            "Set hnsw.ef_search to 100 at query time. The default trades away\n"
            "more recall than this workload can afford.\n"
        ),
    ),
    RawDocument(
        source_type="markdown",
        source_id="test/running.md",
        title="Running",
        path="test/running.md",
        text=(
            "# Running\n\n"
            "## Local environment\n\n"
            "One compose command brings up the database, the API and the web\n"
            "application together with seeded data.\n"
        ),
    ),
]


@pytest_asyncio.fixture(scope="module", loop_scope="session", autouse=True)
async def corpus(pool):
    async with db.pool().connection() as conn:
        await conn.execute("TRUNCATE document CASCADE")
    await ingest(CORPUS, force=True)
    yield


@pytest.fixture
def reranker_calls(monkeypatch):
    """Counts calls rather than inspecting scores.

    An unreadable question usually retrieves nothing on this English corpus, so
    "no source carries a re-rank score" would pass for free. Whether
    `get_reranker` was reached at all is true either way.
    """
    calls: list[str] = []
    real = search.get_reranker

    def counting_get_reranker():
        calls.append("called")
        return real()

    monkeypatch.setattr(search, "get_reranker", counting_get_reranker)
    return calls


async def test_a_thai_question_never_reaches_the_reranker(reranker_calls):
    result = await hybrid_search(THAI_QUESTION, top_k=5)

    assert result.unreadable_query is True
    assert reranker_calls == [], "0.99 on an irrelevant chunk comes from here"
    for source in result.sources:
        assert source.retrieval.vector_rank is None, "the blind leg should be off"
        assert source.retrieval.rerank_score is None


async def test_an_english_question_is_unchanged_in_every_respect(reranker_calls):
    result = await hybrid_search("hnsw.ef_search", top_k=3)

    assert result.unreadable_query is False
    assert reranker_calls == ["called"]
    assert result.sources[0].path == "test/tuning.md"
    assert result.sources[0].retrieval.rerank_score is not None
    assert any(s.retrieval.vector_rank is not None for s in result.sources)


async def test_one_english_term_is_enough_to_keep_the_normal_path(reranker_calls):
    """The common real case: a Thai question about an English identifier."""
    result = await hybrid_search("ค่า hnsw.ef_search ควรตั้งเท่าไร", top_k=3)

    assert result.unreadable_query is False
    assert reranker_calls == ["called"]
    assert any(s.retrieval.vector_rank is not None for s in result.sources)


async def test_the_eval_harness_legs_still_work():
    """`legs` is how the evaluation harness measures what each leg contributes."""
    vector_only = await hybrid_search("how do I start everything?", legs="vector")
    keyword_only = await hybrid_search("compose seeded data", legs="keyword")

    assert vector_only.sources and keyword_only.sources
    assert all(s.retrieval.keyword_rank is None for s in vector_only.sources)
    assert all(s.retrieval.vector_rank is None for s in keyword_only.sources)
    assert not vector_only.unreadable_query and not keyword_only.unreadable_query


# --- the chat turn --------------------------------------------------------


def parse(frames: list[str]) -> list[tuple[str, dict]]:
    events = []
    for raw in frames:
        event, data = raw.strip().split("\n", 1)
        events.append(
            (event.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        )
    return events


@pytest_asyncio.fixture(loop_scope="session")
async def session_id(pool):
    sid = store.new_id("s")
    yield sid
    async with db.pool().connection() as conn:
        await conn.execute("DELETE FROM chat_session WHERE id = %s", (sid,))


async def test_the_chat_turn_says_what_went_wrong_without_calling_the_llm(session_id):
    """No LLM is configured for this test and none is needed — that is the point.

    The chat model reads Thai perfectly well, so left to itself it would write a
    fluent, confident answer over whatever the blind stages dredged up. The
    fixed message is the alternative to that.
    """
    events = parse(
        [f async for f in service.stream_chat(question=THAI_QUESTION, session_id=session_id)]
    )
    kinds = [name for name, _ in events]

    assert "error" not in kinds
    assert kinds[0] == "session" and kinds[-1] == "done"
    assert kinds.count("token") == 1

    sources = next(data for name, data in events if name == "sources")
    assert sources["notice"], "the sources card has to explain the empty result"

    answer = next(data for name, data in events if name == "token")["text"]
    assert "อ่านคำถามนี้ไม่ออก" in answer, "must not read as 'no documents found'"
    assert "intfloat/multilingual-e5-large" in answer, "and must offer the way out"

    done = next(data for name, data in events if name == "done")
    assert done["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert done["dropped_citations"] == 0
    assert done["model"]


async def test_the_unanswered_turn_is_still_persisted(session_id):
    """A hole in the transcript is its own bug — history has to render."""
    async for _ in service.stream_chat(question=THAI_QUESTION, session_id=session_id):
        pass

    messages = await store.session_messages(session_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == THAI_QUESTION
    assert messages[1]["meta"]["unreadable_query"] is True
    assert messages[1]["meta"]["usage"]["total_tokens"] == 0


async def test_an_english_turn_is_not_diverted(session_id, monkeypatch):
    """The divert is keyed on readability, not on an empty result set."""
    calls: list[str] = []

    def fail_if_called(*a, **kw):
        calls.append("llm")
        raise RuntimeError("stand-in for the LLM")

    monkeypatch.setattr(service, "client", fail_if_called)

    events = parse(
        [
            f
            async for f in service.stream_chat(
                question="how do I start everything?", session_id=session_id
            )
        ]
    )

    sources = next(data for name, data in events if name == "sources")
    assert sources["notice"] is None
    # It reached the generation path and failed there, which is the proof; the
    # stand-in stands in for an LLM this test has no reason to call for real.
    assert calls == ["llm"]
    assert events[-1][0] == "error"
