"""The chat service's error surface and its LLM client.

Neither needs a database, a model or a network: what is worth pinning here is
what leaves the process — the text of an error frame, and how many connection
pools one conversation opens.
"""
from __future__ import annotations

import pytest

from app.chat import service
from app.chat.service import LLMNotConfigured, client, client_error, close_client, frame


class FakeSettings:
    """Only the four attributes `client()` reads."""

    openai_api_key = "sk-not-a-real-key"
    openai_base_url = "https://gen.example.invalid/api/v1"
    llm_configured = True


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(service, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(service, "_client", None)
    yield
    monkeypatch.setattr(service, "_client", None)


# --- B-13: the browser is not shown the exception ---------------------------


def test_a_failed_turn_reports_a_reference_and_not_the_exception():
    """It used to send `f"{type(e).__name__}: {e}"`. A psycopg failure spells
    out the database host, user and name; an OpenAI failure spells out the
    proxy URL, the model and a request id. All of it was rendered in a red box
    on the user's screen, and none of it was any use to them.
    """
    leaky = RuntimeError(
        "connection to server at 'db.internal' (10.0.0.4), port 5432 failed: "
        "FATAL: password authentication failed for user 'rag'"
    )
    detail = client_error(leaky, "err_9f21c4")

    assert "err_9f21c4" in detail, "the reference is the join back to the log"
    for internal in ["db.internal", "10.0.0.4", "5432", "rag", "RuntimeError"]:
        assert internal not in detail


def test_the_one_error_the_reader_can_fix_still_says_what_to_fix():
    """"Set OPENAI_API_KEY" names nothing internal and is the whole remedy —
    hiding it behind a reference id would make the user open the server log to
    be told to edit their own .env."""
    detail = client_error(LLMNotConfigured("OPENAI_API_KEY is not set"), "err_1")
    assert "OPENAI_API_KEY" in detail
    assert "err_1" not in detail


def test_the_error_frame_is_still_a_well_formed_sse_event():
    sse = frame("error", {"detail": client_error(ValueError("boom"), "err_2")})
    assert sse.startswith("event: error\ndata: {")
    assert sse.endswith("\n\n")
    assert "boom" not in sse


# --- B-19: one client per process, not one per call -------------------------


def test_the_llm_client_is_built_once_and_reused(configured):
    """Each `AsyncOpenAI` owns an httpx pool. Built per call, one question made
    two of them — rewrite, then answer — so every first token waited behind a
    fresh TLS handshake and the pools were left for the garbage collector.
    """
    assert client() is client()


@pytest.mark.asyncio
async def test_closing_the_client_releases_it_and_a_later_call_rebuilds(configured):
    first = client()
    await close_client()

    assert service._client is None
    assert client() is not first


def test_without_a_key_the_client_refuses_in_a_way_the_error_frame_can_read(
    monkeypatch,
):
    class Unconfigured(FakeSettings):
        openai_api_key = ""
        llm_configured = False

    monkeypatch.setattr(service, "get_settings", lambda: Unconfigured())
    monkeypatch.setattr(service, "_client", None)

    with pytest.raises(LLMNotConfigured, match="OPENAI_API_KEY"):
        client()
