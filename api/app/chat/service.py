"""The chat stream: rewrite, retrieve, generate, guard, persist."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.chat import store
from app.chat.prompt import (
    build_messages,
    build_rewrite_messages,
    strip_unsupported_citations,
)
from app.config import get_settings
from app.retrieval.search import SearchResult, hybrid_search, notice_for
from app.schemas import Source

log = logging.getLogger(__name__)


def unreadable_query_message(embed_model: str, rerank_model: str) -> str:
    """The answer sent when retrieval could not read the question.

    It has to say *the model cannot read this*, not *no documents found*. The
    two look identical from the outside and have opposite fixes: one sends the
    user off to check whether the corpus is empty, when the corpus is fine and
    the model is simply the wrong one for this language. So the cause is named,
    the models are named, and both real ways out are given.
    """
    return (
        f"ผมยังตอบคำถามนี้ไม่ได้ครับ — ไม่ใช่เพราะไม่มีเอกสารในระบบ "
        f"แต่เพราะโมเดล embedding ที่ใช้อยู่ (`{embed_model}`) "
        f"อ่านคำถามนี้ไม่ออก\n\n"
        f"โมเดลนี้มีคำศัพท์เฉพาะภาษาอังกฤษ ข้อความที่ไม่ใช่ภาษาอังกฤษจึงกลายเป็น "
        f"token “ไม่รู้จัก” ทั้งหมด และคำถามทุกข้อในภาษานั้นจะได้เวกเตอร์ตัวเดียวกันเป๊ะ "
        f"ระบบจึงปิดการค้นแบบ semantic ไว้ แทนที่จะหยิบเอกสารสุ่มมาตอบอย่างมั่นใจ\n\n"
        f"ทางออก:\n"
        f"- ถามเป็นภาษาอังกฤษ หรือใส่คำสำคัญภาษาอังกฤษลงไปในคำถามด้วย\n"
        f"- หรือเปลี่ยน `EMBED_MODEL` เป็น `intfloat/multilingual-e5-large` "
        f"(`EMBED_DIM=1024`) แล้ว re-index ใหม่ — และเปลี่ยน `RERANK_MODEL` จาก "
        f"`{rerank_model}` เป็น `jinaai/jina-reranker-v2-base-multilingual` ด้วย "
        f"เพราะตัว re-ranker ก็อ่านไม่ออกเหมือนกัน (ดู `.env.example`)\n\n"
        f"_The embedding model in use is English-only and cannot read this "
        f"question, so semantic search was disabled rather than allowed to "
        f"return arbitrary documents. Ask in English, or switch to a "
        f"multilingual embedding and re-ranking model and re-index._"
    )


class LLMNotConfigured(RuntimeError):
    """No API key. Separate from every other failure because it is the one the
    person reading the error can actually fix, so its text is theirs to see."""


_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    """The one OpenAI client for this process.

    Built once rather than per call: every `AsyncOpenAI` owns an httpx
    connection pool, and a per-call client meant two fresh pools per question
    (rewrite, then answer), a TLS handshake in front of every first token, and
    connections left open for the garbage collector to notice.
    """
    global _client
    settings = get_settings()
    if not settings.llm_configured:
        raise LLMNotConfigured(
            "ยังไม่ได้ตั้งค่า LLM — OPENAI_API_KEY is not set; copy .env.example "
            "to .env and restart the API."
        )
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=90.0,
        )
    return _client


async def close_client() -> None:
    """Close the shared client's pool. Called from the application lifespan."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def client_error(exc: Exception, error_id: str) -> str:
    """What the browser is allowed to be told about a failed turn.

    Not `f"{type(e).__name__}: {e}"`, which is what this used to send: a psycopg
    error carries the database host, user and name; an OpenAI error carries the
    proxy URL, the model and a request id. All of it was rendered in a red box
    on the user's screen, and none of it means anything to them. The reference
    is the join back to the log line, which still has everything.
    """
    if isinstance(exc, LLMNotConfigured):
        return str(exc)
    return (
        "ระบบตอบคำถามนี้ไม่สำเร็จ ลองถามใหม่อีกครั้งได้เลยครับ "
        f"(the answer could not be generated · reference {error_id})"
    )


def frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def rewrite_question(question: str, history: list[tuple[str, str]]) -> str:
    """Fold the conversation into a standalone question before retrieving.

    "What about the second one?" retrieves nothing on its own — embeddings and
    keywords both need the subject to be present in the text. This is the
    single biggest quality difference between one-shot RAG and chat RAG.

    A failure here is not fatal: the original question still retrieves, just
    less well, so the error is logged and the turn continues.
    """
    if not history:
        return question
    try:
        response = await client().chat.completions.create(
            model=get_settings().chat_model,
            messages=build_rewrite_messages(question, history),  # type: ignore[arg-type]
            max_tokens=120,
            temperature=0.0,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception as e:  # noqa: BLE001
        log.warning("query rewrite failed, using the question as asked: %s", e)
        return question


def nothing_relevant_message(considered: int) -> str:
    """The answer when retrieval worked and found nothing that answers this.

    Said here rather than left to the model. Handing an LLM five passages that
    the re-ranker scored at zero and trusting it to reply "the documents do
    not cover this" is a policy enforced by politeness: the model can just as
    easily assemble a fluent answer out of five irrelevant paragraphs, cite
    them, and be believed — and the citation guard cannot catch it, because
    the citations are real. The chunks were shown; they simply do not say it.

    The two cases are separated because they have different fixes. Nothing
    retrieved at all usually means an empty index; things retrieved and all of
    them irrelevant means the corpus does not cover the subject.
    """
    if considered == 0:
        return (
            "ไม่พบเอกสารที่เกี่ยวข้องเลยครับ — การค้นหาไม่คืนผลอะไรกลับมาเลย "
            "ซึ่งมักแปลว่ายังไม่ได้ index เอกสารเข้าระบบ\n\n"
            "ลองตรวจที่แผง Corpus ว่ามีเอกสารอยู่หรือไม่ หรือรัน "
            "`python -m app.ingest <path>` เพื่อเพิ่มเอกสารเข้าไป\n\n"
            "_Nothing was retrieved for this question at all, which usually "
            "means the index is empty._"
        )
    return (
        f"ผมไม่พบเนื้อหาในเอกสารที่ตอบคำถามนี้ได้ครับ — ระบบค้นเจอ {considered} "
        "ชิ้นที่ใกล้เคียงที่สุด แต่ตัวจัดอันดับประเมินว่าไม่มีชิ้นไหนเกี่ยวข้องมากพอ "
        "จะตอบจากเนื้อหาเหล่านั้น\n\n"
        "ผมจึงไม่ตอบดีกว่าเดาครับ ถ้าคำถามนี้ควรมีคำตอบอยู่ในระบบ "
        "ลองใช้คำที่ตรงกับที่เอกสารใช้ หรือเพิ่มเอกสารที่เกี่ยวข้องเข้าไปใน corpus\n\n"
        "_The corpus was searched and nothing in it was scored as relevant "
        "enough to answer from, so no answer was generated over it._"
    )


async def _turn_without_the_model(
    *,
    answer: str,
    session_id: str,
    question: str,
    search_query: str,
    sources: list[Source],
    result: SearchResult,
    started: float,
    extra_meta: dict,
) -> AsyncIterator[str]:
    """A turn answered without calling the LLM at all.

    Deliberately shaped like an ordinary turn — `token` frames, then `done`,
    and the turn is persisted — so the client needs no special case to render
    it and the session history does not develop holes where these turns were.
    """
    settings = get_settings()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    yield frame("token", {"text": answer})

    message_id = await store.finish_turn(
        session_id=session_id,
        answer=answer,
        sources=sources,
        meta={
            "model": settings.chat_model,
            "usage": usage,
            "dropped_citations": 0,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "retrieval_ms": result.retrieval_ms,
            "candidates_considered": result.candidates_considered,
            "rewritten_query": search_query if search_query != question else None,
            # So a stored turn explains itself later, rather than reading as a
            # turn where the model inexplicably refused. The notice is snapshot
            # alongside the flag for the same reason the source payload is: a
            # replayed turn has to render what the live turn rendered, without
            # the client keeping its own copy of the sentence.
            "notice": notice_for(result),
            **extra_meta,
        },
    )

    yield frame(
        "done",
        {
            "message_id": message_id,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "usage": usage,
            "dropped_citations": 0,
            "model": settings.chat_model,
        },
    )


async def _save_interrupted(
    *,
    session_id: str,
    parts: list[str],
    sources: list[Source],
    result: SearchResult | None,
    started: float,
    model: str,
    usage: dict,
    reason: dict,
) -> bool:
    """Persist however much of an answer had arrived before the turn was cut.

    Returns whether anything was written.

    Shielded, because the common way to get here is the reader disconnecting,
    which cancels this task: an unshielded `await` in a `finally` on a
    cancelled task raises `CancelledError` at the first suspension point and
    the write never reaches Postgres. The shield lets the write finish while
    the cancellation continues to propagate, which is what it is for.

    With no tokens at all there is nothing worth a row: the question is
    already stored by `start_turn`, and a transcript showing a question with
    no answer is an accurate account of what happened. An empty assistant
    message would only be a second, less honest way of saying the same thing.
    """
    if not parts:
        return False

    cleaned, dropped = strip_unsupported_citations(
        "".join(parts), {s.n for s in sources}
    )
    try:
        await asyncio.shield(
            store.finish_turn(
                session_id=session_id,
                answer=cleaned,
                sources=sources,
                meta={
                    "model": model,
                    "usage": usage,
                    "dropped_citations": dropped,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "retrieval_ms": result.retrieval_ms if result else 0,
                    "candidates_considered": (
                        result.candidates_considered if result else 0
                    ),
                    **reason,
                },
            )
        )
    except Exception:  # noqa: BLE001 — losing the answer is worse than the log line
        log.exception("could not persist the interrupted turn")
        return False
    return True


async def stream_chat(
    *, question: str, session_id: str, top_k: int | None = None
) -> AsyncIterator[str]:
    """Yield SSE frames for one turn.

    Order matters: `sources` is flushed before the first token so the citation
    cards are on screen and readable while the answer is still being written.

    The turn is opened in the store before retrieval and closed when the
    answer is whole. Everything in between can fail, and until it was split
    this way all of it ended the same: the whole turn was written in one call
    after the last token, so an error, a Stop, or a closed laptop lid threw
    away the question as well as the answer.
    """
    settings = get_settings()
    started = time.perf_counter()

    parts: list[str] = []
    sources: list[Source] = []
    result: SearchResult | None = None
    search_query = question
    model_used = settings.chat_model
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    saved = False

    try:
        history = await store.recent_turns(session_id)
        title = store.derive_title(
            next((c for r, c in history if r == "user"), question)
        )
        yield frame("session", {"session_id": session_id, "title": title})

        # Before anything that can fail, and before the history read above is
        # invalidated by this very question joining it.
        await store.start_turn(
            session_id=session_id, title=title, question=question
        )

        search_query = await rewrite_question(question, history)
        result = await hybrid_search(search_query, top_k=top_k)
        sources = result.sources

        yield frame(
            "sources",
            {
                "sources": [s.model_dump() for s in sources],
                "candidates_considered": result.candidates_considered,
                "retrieval_ms": result.retrieval_ms,
                "notice": notice_for(result),
            },
        )

        # Two ways a turn is answered without the model, and they are separate
        # facts: one is "the question could not be read", the other "the
        # corpus does not answer it". Both used to be, or would have been,
        # settled by asking an LLM to be honest about material it was handed.
        explanation: str | None = None
        extra_meta: dict = {}
        if result.unreadable_query:
            # Whatever the keyword leg scraped together, retrieval could not
            # read the question, so an answer generated over it would be a
            # guess wearing citations — and the chat model reads Thai
            # perfectly well, which makes that guess fluent and confident.
            explanation = unreadable_query_message(
                settings.embed_model, settings.rerank_model
            )
            extra_meta = {"unreadable_query": True}
        elif not sources:
            # Retrieval came back with nothing at all — not "nothing good",
            # nothing. There is no context to answer over and no judgement for
            # a model to make about it, so the call is skipped rather than
            # spent on asking an LLM to report an empty list back to us.
            explanation = nothing_relevant_message(result.candidates_considered)
            extra_meta = {"nothing_relevant": True}

        if explanation is not None:
            async for sse in _turn_without_the_model(
                answer=explanation,
                session_id=session_id,
                question=question,
                search_query=search_query,
                sources=sources,
                result=result,
                started=started,
                extra_meta=extra_meta,
            ):
                yield sse
            saved = True
            return

        stream = await client().chat.completions.create(
            model=settings.chat_model,
            # `history` is what makes a follow-up answerable: the rewriter has
            # always had it, and the model that has to write the answer never
            # did.
            messages=build_messages(  # type: ignore[arg-type]
                question, sources, history, low_confidence=result.low_confidence
            ),
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.2,
        )
        async for chunk in stream:
            if chunk.model:
                model_used = chunk.model
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                    "total_tokens": chunk.usage.total_tokens or 0,
                }
            for choice in chunk.choices or []:
                delta = choice.delta.content
                if delta:
                    parts.append(delta)
                    yield frame("token", {"text": delta})

        cleaned, dropped = strip_unsupported_citations(
            "".join(parts), {s.n for s in sources}
        )

        message_id = await store.finish_turn(
            session_id=session_id,
            answer=cleaned,
            sources=sources,
            meta={
                "model": model_used,
                "usage": usage,
                "dropped_citations": dropped,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "retrieval_ms": result.retrieval_ms,
                "candidates_considered": result.candidates_considered,
                "rewritten_query": search_query if search_query != question else None,
                # Snapshot for the same reason the source payload is: a
                # replayed turn has to be able to show what the live one did.
                "notice": notice_for(result),
            },
        )
        saved = True

        yield frame(
            "done",
            {
                "message_id": message_id,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "usage": usage,
                "dropped_citations": dropped,
                "model": model_used,
            },
        )
    except Exception as e:  # noqa: BLE001 — the client gets an error frame, not a dead socket
        error_id = store.new_id("err")
        log.exception("chat turn failed [%s]", error_id)
        if not saved:
            saved = await _save_interrupted(
                session_id=session_id,
                parts=parts,
                sources=sources,
                result=result,
                started=started,
                model=model_used,
                usage=usage,
                reason={"failed": True, "error_id": error_id},
            )
        yield frame("error", {"detail": client_error(e, error_id)})
    finally:
        # Reached when the reader pressed Stop or the connection dropped: the
        # generator is closed at a `yield`, so neither branch above runs and
        # without this the half-written answer is discarded.
        if not saved:
            await _save_interrupted(
                session_id=session_id,
                parts=parts,
                sources=sources,
                result=result,
                started=started,
                model=model_used,
                usage=usage,
                reason={"aborted": True},
            )
