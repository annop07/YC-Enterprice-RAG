"""Prompt assembly and the citation guard."""
from __future__ import annotations

from app.chat.prompt import (
    build_context,
    build_messages,
    cited_numbers,
    locator_label,
    strip_unsupported_citations,
)
from app.schemas import Locator, RetrievalTrace, Source


def source(n: int, **overrides) -> Source:
    base = dict(
        n=n,
        chunk_id=str(n),
        document_id="d1",
        title="Ingestion Pipeline",
        source_type="markdown",
        path="docs/ingestion.md",
        url=None,
        heading_path="Ingestion Pipeline > Chunking",
        locator=Locator(line_start=18, line_end=24, page=None),
        retrieval=RetrievalTrace(vector_rank=1, keyword_rank=2, rrf_score=0.03),
        snippet=f"snippet {n}",
    )
    base.update(overrides)
    return Source(**base)  # type: ignore[arg-type]


def test_context_blocks_are_numbered_and_locatable():
    context = build_context([source(1), source(2)])

    assert context.startswith("[1] Ingestion Pipeline — docs/ingestion.md:18-24")
    assert "[2] " in context
    assert "snippet 1" in context and "snippet 2" in context


def test_pdf_blocks_are_labelled_by_page():
    pdf = source(1, path="handbook.pdf", locator=Locator(line_start=9, line_end=16, page=2))
    assert locator_label(pdf) == "handbook.pdf p.2"


def test_the_prompt_says_so_when_retrieval_came_back_empty():
    """Otherwise the model sees an empty context and fills it in from memory."""
    messages = build_messages("anything?", [])
    assert "nothing was retrieved" in messages[1]["content"]


def test_the_system_prompt_forbids_outside_knowledge_and_requires_citations():
    system = build_messages("q", [source(1)])[0]["content"]
    assert "Never use outside knowledge" in system
    assert "[2]" in system  # the citation format is shown, not just described


def test_valid_citations_survive_untouched():
    text = "Chunks are 400 tokens [1]. Overlap is 80 [2]."
    cleaned, dropped = strip_unsupported_citations(text, {1, 2})

    assert cleaned == text
    assert dropped == 0


def test_a_citation_pointing_at_nothing_is_removed_and_counted():
    """The model can invent a number; nothing in the format stops it."""
    text = "Chunks are 400 tokens [1]. The cache is warmed nightly [7]."
    cleaned, dropped = strip_unsupported_citations(text, {1, 2})

    assert dropped == 1
    assert "[7]" not in cleaned
    assert "[1]" in cleaned
    # The sentence must not be left with a gap before its full stop.
    assert cleaned == "Chunks are 400 tokens [1]. The cache is warmed nightly."


def test_several_invented_citations_are_all_counted():
    _, dropped = strip_unsupported_citations("a [4] b [5] c [6]", {1})
    assert dropped == 3


def test_stripping_leaves_ordinary_brackets_alone():
    text = "See the array [0] and the note [1]."
    cleaned, dropped = strip_unsupported_citations(text, {0, 1})
    assert cleaned == text and dropped == 0


def test_cited_numbers_reads_what_the_answer_claims():
    assert cited_numbers("a [1] b [3] c [1]") == {1, 3}
    assert cited_numbers("no citations here") == set()


# --- the conversation the answering model never used to see ---------------


def test_the_conversation_reaches_the_model_that_writes_the_answer():
    """The rewriter had the history; the model answering the question did not.

    So "what about the second one?" arrived at the model as those exact words
    with five chunks attached and nothing saying what the second one was. The
    good case was a lucky guess off the context; the normal case was the model
    asking the user to repeat themselves, over a corpus that had already found
    the answer.
    """
    history = [
        ("user", "What chunk size does ingestion use?"),
        ("assistant", "400 tokens, with 80 of overlap [1]."),
    ]
    messages = build_messages("why not larger?", [source(1)], history)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "What chunk size does ingestion use?"
    assert "why not larger?" in messages[-1]["content"]


def test_an_earlier_answer_loses_its_citation_numbers():
    """`[1]` in the previous turn is not `[1]` in this one.

    Each turn retrieves its own chunks and numbers them from one, so a number
    carried over from an earlier answer points at a different passage. The
    citation guard cannot catch that — the number is valid for this turn, so
    it is kept, and it is wrong.
    """
    history = [("assistant", "It uses 400 tokens [1], with 80 of overlap [2].")]
    carried = build_messages("and the overlap?", [source(1)], history)[1]["content"]

    assert "[1]" not in carried and "[2]" not in carried
    assert carried == "It uses 400 tokens, with 80 of overlap."


def test_a_users_own_words_are_never_rewritten_in_the_history():
    """Only assistant turns are stripped: `[2]` typed by a user is their text."""
    history = [("user", "what does [2] mean?")]
    assert build_messages("q", [source(1)], history)[1]["content"] == "what does [2] mean?"


def test_no_history_is_the_same_two_messages_it_always_was():
    assert len(build_messages("q", [source(1)])) == 2


def test_low_confidence_retrieval_says_so_in_the_prompt():
    """The blocks look identical whether or not anything scored as relevant.

    They are the closest rows in the index either way, so the model has no way
    to tell from the context alone that it is looking at a near-miss set.
    """
    plain = build_messages("q", [source(1)])[-1]["content"]
    warned = build_messages("q", [source(1)], low_confidence=True)[-1]["content"]

    assert "not scored as clearly relevant" not in plain
    assert "clearly relevant" in warned
    assert warned.startswith(plain[:40])


def test_nothing_retrieved_says_so_rather_than_sending_an_empty_context():
    assert "(nothing was retrieved" in build_messages("q", [])[-1]["content"]
