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
from app.retrieval.search import hybrid_search
from app.schemas import Source

log = logging.getLogger(__name__)


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
            },
        )

        parts: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model_used = settings.chat_model

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
