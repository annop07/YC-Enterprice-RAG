"""What survives a turn that does not finish.

The whole turn used to be written in one call after the last token, so an LLM
error, a reader pressing Stop, or a closed connection threw away the question
as well as the answer. If it was the first question of a conversation, the
session never existed either: nothing in the sidebar, nothing to go back to,
no evidence the turn had happened at all.
"""
from __future__ import annotations

import pytest

from app import db
from app.chat import store
from tests.conftest import requires_db

pytestmark = [requires_db, pytest.mark.asyncio]


@pytest.fixture
def session_id() -> str:
    return store.new_id("s")


async def test_the_question_is_stored_before_the_answer_exists(pool, session_id):
    await store.start_turn(
        session_id=session_id, title="Chunking", question="What chunk size?"
    )

    messages = await store.session_messages(session_id)
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "What chunk size?")
    ]


async def test_an_opened_turn_puts_the_session_in_the_list(pool, session_id):
    """The sidebar reads `chat_session`. Until the turn finished, nothing wrote it."""
    await store.start_turn(session_id=session_id, title="Chunking", question="q")
    assert session_id in [s[0] for s in await store.list_sessions()]


async def test_an_answer_that_never_arrives_leaves_the_question_readable(
    pool, session_id
):
    """The transcript of an interrupted turn is the question, and nothing else.

    An empty assistant row would be a second, less honest way of saying the
    same thing — this way the reader sees what they asked and that it was not
    answered.
    """
    await store.start_turn(session_id=session_id, title="t", question="unanswered?")

    messages = await store.session_messages(session_id)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


async def test_a_partial_answer_is_kept_with_the_reason_it_stopped(pool, session_id):
    await store.start_turn(session_id=session_id, title="t", question="q")
    await store.finish_turn(
        session_id=session_id,
        answer="The ingestion pipeline uses 400 tok",
        sources=[],
        meta={"model": "m", "aborted": True},
    )

    messages = await store.session_messages(session_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"].endswith("400 tok")
    assert messages[1]["meta"]["aborted"] is True


async def test_the_question_still_comes_back_above_its_answer(pool, session_id):
    """Two transactions now, not one — the ordering must not depend on that.

    `created_at` cannot order a conversation (it is transaction time, and both
    rows used to share one); `seq` is assigned at insert and the question is
    always inserted first.
    """
    await store.start_turn(session_id=session_id, title="t", question="the question")
    await store.finish_turn(
        session_id=session_id, answer="the answer", sources=[], meta={}
    )

    messages = await store.session_messages(session_id)
    assert [m["content"] for m in messages] == ["the question", "the answer"]


async def test_a_second_turn_lands_after_the_first(pool, session_id):
    for n in ("one", "two"):
        await store.start_turn(session_id=session_id, title="t", question=f"q {n}")
        await store.finish_turn(
            session_id=session_id, answer=f"a {n}", sources=[], meta={}
        )

    messages = await store.session_messages(session_id)
    assert [m["content"] for m in messages] == ["q one", "a one", "q two", "a two"]


async def test_opening_a_turn_twice_does_not_duplicate_the_session(pool, session_id):
    await store.start_turn(session_id=session_id, title="first", question="q1")
    await store.start_turn(session_id=session_id, title="first", question="q2")

    row = await db.fetch_one(
        "SELECT count(*), max(title) FROM chat_session WHERE id = %s", (session_id,)
    )
    assert row == (1, "first")


async def test_history_read_before_a_turn_opens_does_not_contain_it(pool, session_id):
    """`stream_chat` reads history first, then opens the turn.

    In the other order the rewriter would be handed the very question it is
    supposed to be resolving against the conversation, and the last user
    message would always be a duplicate of the current one.
    """
    await store.start_turn(session_id=session_id, title="t", question="first")
    await store.finish_turn(session_id=session_id, answer="a", sources=[], meta={})

    history = await store.recent_turns(session_id)
    await store.start_turn(session_id=session_id, title="t", question="second")

    assert [c for _, c in history] == ["first", "a"]


# --- the blob-sha lookup the ingest endpoint depends on -------------------


async def test_the_known_blob_sha_lookup_runs_at_all(pool):
    """A regression test for a comment.

    psycopg scans the whole statement for placeholders and does not skip SQL
    comments, so a per-cent sign written inside one — in a comment explaining
    why a `LIKE 'repo@%'` had been replaced, as it happens — is read as a
    malformed parameter and the query raises before it reaches Postgres. Every
    `POST /ingest/github` returned 500, including the ones that were correct
    in every other way, and no unit test noticed because they all build the
    connector directly.
    """
    from app.main import _known_blob_shas

    assert await _known_blob_shas("owner/name") == {}


async def test_an_underscore_in_a_repository_name_is_not_a_wildcard(pool, session_id):
    """`_` matches any single character in LIKE.

    So the lookup keyed on `my_repo@` also matched `myXrepo@`, and this
    repository could be handed another one's blob shas — which the connector
    trusts as "unchanged" and skips fetching, leaving the wrong text indexed
    under the right path.
    """
    from app.ingest.connectors import RawDocument
    from app.ingest.pipeline import ingest
    from app.main import _known_blob_shas

    await ingest(
        [
            RawDocument(
                source_type="github",
                source_id="myXrepo/x@docs/guide.md",
                title="Guide",
                path="docs/guide.md",
                text="# Guide\n\nThe impostor.\n",
                meta={"blob_sha": "sha-of-the-wrong-repository"},
            )
        ]
    )

    assert await _known_blob_shas("my_repo/x") == {}


# --- the abort path, driven end to end ------------------------------------


def _chunk(text: str | None = None, *, model: str = "stub", usage=None):
    """The shape `stream_chat` reads off an OpenAI streaming response."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model=model,
        usage=usage,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
    )


def stub_llm(monkeypatch, tokens: list[str]):
    """A `client()` whose stream yields `tokens` and then waits forever.

    Never finishing is the point: the turn has to be cut off by the reader
    going away, which is the case that used to discard the answer.
    """
    import asyncio as _asyncio
    from types import SimpleNamespace

    from app.chat import service

    async def stream():
        for token in tokens:
            yield _chunk(token)
        await _asyncio.Event().wait()  # the model is still writing

    async def create(**kwargs):
        create.kwargs = kwargs
        return stream()

    monkeypatch.setattr(
        service,
        "client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    return create


async def test_a_stream_cut_off_mid_answer_keeps_what_had_arrived(
    pool, session_id, monkeypatch
):
    """Closing the generator is what a reader pressing Stop does.

    The first version of this fix looked right and saved nothing: the write
    went through `asyncio.shield`, `asyncio` was never imported in that
    module, and the `except Exception` around the write logged the resulting
    NameError and carried on. Nothing above this level could tell the
    difference between that and a turn with no tokens in it.
    """
    from app.chat import service

    stub_llm(monkeypatch, ["Retrieval ", "runs ", "two ", "legs"])

    frames = []
    agen = service.stream_chat(question="how does retrieval work?", session_id=session_id)
    async for frame in agen:
        frames.append(frame)
        if sum(1 for f in frames if "event: token" in f) >= 4:
            break
    await agen.aclose()

    messages = await store.session_messages(session_id)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "Retrieval runs two legs"
    assert messages[1]["meta"]["aborted"] is True


async def test_the_conversation_is_handed_to_the_answering_model(
    pool, session_id, monkeypatch
):
    """B-05 at the level that matters: what actually goes over the wire."""
    from app.chat import service

    await store.start_turn(session_id=session_id, title="t", question="What chunk size?")
    await store.finish_turn(
        session_id=session_id, answer="400 tokens [1].", sources=[], meta={"model": "m"}
    )

    create = stub_llm(monkeypatch, ["because ", "512 ", "is ", "the ", "cap"])
    agen = service.stream_chat(question="why not larger?", session_id=session_id)
    frames = []
    async for frame in agen:
        frames.append(frame)
        if sum(1 for f in frames if "event: token" in f) >= 5:
            break
    await agen.aclose()

    roles = [m["role"] for m in create.kwargs["messages"]]
    contents = [m["content"] for m in create.kwargs["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1] == "What chunk size?"
    # The earlier answer is there, without the citation number that pointed at
    # a chunk this turn never retrieved.
    assert contents[2] == "400 tokens."
    assert "why not larger?" in contents[3]
