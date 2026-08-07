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
