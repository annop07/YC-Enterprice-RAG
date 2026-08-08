"""The chat stream: rewrite, retrieve, generate, guard, persist."""
from __future__ import annotations

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
from app.retrieval.search import hybrid_search, notice_for
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


def client() -> AsyncOpenAI:
    settings = get_settings()
    if not settings.llm_configured:
        raise RuntimeError("OPENAI_API_KEY is not set — copy .env.example to .env.")
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=90.0,
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


async def _explain_instead_of_answering(
    *,
    settings,
    session_id: str,
    history: list[tuple[str, str]],
    question: str,
    search_query: str,
    sources: list[Source],
    result,
    started: float,
) -> AsyncIterator[str]:
    """The unreadable-question turn: the same frames, with no model call.

    Deliberately shaped like an ordinary turn — `token` frames, then `done`,
    and the turn is persisted — so the client needs no special case to render
    it and the session history does not develop holes where these turns were.
    """
    answer = unreadable_query_message(settings.embed_model, settings.rerank_model)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    yield frame("token", {"text": answer})

    message_id = await store.save_turn(
        session_id=session_id,
        title=store.derive_title(
            next((c for r, c in history if r == "user"), question)
        ),
        question=question,
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
            "unreadable_query": True,
            "notice": notice_for(result),
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


async def stream_chat(
    *, question: str, session_id: str, top_k: int | None = None
) -> AsyncIterator[str]:
    """Yield SSE frames for one turn.

    Order matters: `sources` is flushed before the first token so the citation
    cards are on screen and readable while the answer is still being written.
    """
    settings = get_settings()
    started = time.perf_counter()

    history = await store.recent_turns(session_id)
    title = store.derive_title(
        next((c for r, c in history if r == "user"), question)
    )
    yield frame("session", {"session_id": session_id, "title": title})

    try:
        search_query = await rewrite_question(question, history)
        result = await hybrid_search(search_query, top_k=top_k)
        sources: list[Source] = result.sources

        yield frame(
            "sources",
            {
                "sources": [s.model_dump() for s in sources],
                "candidates_considered": result.candidates_considered,
                "retrieval_ms": result.retrieval_ms,
                "notice": notice_for(result),
            },
        )

        parts: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_used = settings.chat_model

        if result.unreadable_query:
            # No call to the LLM at all. Whatever the keyword leg scraped
            # together, retrieval could not read the question, so an answer
            # generated over it would be a guess wearing citations — and the
            # chat model reads Thai perfectly well, which makes that guess
            # fluent and confident. Say what went wrong instead.
            async for sse in _explain_instead_of_answering(
                settings=settings,
                session_id=session_id,
                history=history,
                question=question,
                search_query=search_query,
                sources=sources,
                result=result,
                started=started,
            ):
                yield sse
            return

        stream = await client().chat.completions.create(
            model=settings.chat_model,
            messages=build_messages(question, sources),  # type: ignore[arg-type]
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

        answer = "".join(parts)
        cleaned, dropped = strip_unsupported_citations(answer, {s.n for s in sources})

        message_id = await store.save_turn(
            session_id=session_id,
            title=store.derive_title(
                next((c for r, c in history if r == "user"), question)
            ),
            question=question,
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
            },
        )

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
        log.exception("chat turn failed")
        yield frame("error", {"detail": f"{type(e).__name__}: {e}"})
