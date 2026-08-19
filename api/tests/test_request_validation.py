"""What the request models refuse before any work starts.

These are the cheapest tests in the suite and they guard the most expensive
mistake: a request that is accepted, embedded, searched, sent to an LLM and
stored, when the only honest answer was 422.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest, SearchRequest

#: Every shape of "the user pressed send with nothing in the box". `min_length`
#: alone lets all of them through, because a space is a character.
BLANK = ["", " ", "   ", "\n", "\t", " \n\t "]


@pytest.mark.parametrize("blank", BLANK)
def test_a_blank_chat_message_is_refused(blank: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message=blank)


@pytest.mark.parametrize("blank", BLANK)
def test_a_blank_search_query_is_refused(blank: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=blank)


def test_padding_is_stripped_and_the_question_itself_is_untouched() -> None:
    """Stripping is at the edges only — an inner newline is part of the question."""
    request = ChatRequest(message="  What chunk size\ndoes ingestion use?  ")
    assert request.message == "What chunk size\ndoes ingestion use?"


@pytest.mark.parametrize("out_of_range", [-100, -1, 0, 21, 1000])
def test_a_chat_top_k_outside_the_useful_range_is_refused(out_of_range: int) -> None:
    """`top_k` ends up in a Python slice, where a negative number is a valid
    instruction to count from the end rather than an error. `top_k=-1` sent
    nineteen chunks into the prompt instead of five; `top_k=-100` sent none and
    the model answered "there is nothing in the documents" over a corpus that
    had the answer; `top_k=0` fell back to the default through `x or default`,
    which at least was harmless. The ceiling is FUSION_KEEP — nothing past it
    survives re-ranking to be sent anyway.
    """
    with pytest.raises(ValidationError):
        ChatRequest(message="what is the chunk size?", top_k=out_of_range)


@pytest.mark.parametrize("model", [ChatRequest, SearchRequest])
def test_omitting_top_k_is_still_how_you_ask_for_the_default(model) -> None:
    field = {"message": "x"} if model is ChatRequest else {"query": "x"}
    assert model(**field).top_k is None
    assert model(**field, top_k=5).top_k == 5


def test_a_question_of_punctuation_is_still_a_question() -> None:
    """The rule is "no text", not "no letters".

    `Embedder.reads` deliberately keeps a letterless query like "5433" on the
    normal path, and this must not be the layer that takes it back off.
    """
    assert ChatRequest(message="???").message == "???"
    assert SearchRequest(query="5433").query == "5433"
