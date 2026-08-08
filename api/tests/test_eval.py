"""Scoring and golden-set validation.

The measurement itself needs a corpus; what is tested here is that the metrics
are computed correctly and that a mistyped expectation is caught rather than
reported as a retrieval failure.
"""
from __future__ import annotations

import re

from app.eval.runner import (
    Question,
    ReadabilityScore,
    Score,
    as_markdown,
    first_relevant_rank,
    load_golden,
    load_readability,
    validate,
)
from app.schemas import Locator, RetrievalTrace, Source

THAI = re.compile(r"[฀-๿]")


def source(n: int, path: str, snippet: str) -> Source:
    return Source(
        n=n,
        chunk_id=str(n),
        document_id="d",
        title="t",
        source_type="markdown",
        path=path,
        url=None,
        heading_path=None,
        locator=Locator(line_start=1, line_end=2, page=None),
        retrieval=RetrievalTrace(vector_rank=n, keyword_rank=None, rrf_score=0.0),
        snippet=snippet,
    )


QUESTION = Question(
    question="What chunk size?", path="ingestion.md", contains="400 tokens"
)


def test_a_hit_needs_the_right_passage_not_just_the_right_file():
    """A five-chunk document would otherwise score a hit for any of it."""
    wrong_passage = source(1, "ingestion.md", "Connectors yield RawDocument.")
    right_passage = source(2, "ingestion.md", "Chunks are 400 tokens with overlap.")

    assert first_relevant_rank(QUESTION, [wrong_passage]) is None
    assert first_relevant_rank(QUESTION, [wrong_passage, right_passage]) == 2


def test_the_right_text_in_the_wrong_document_is_not_a_hit():
    assert first_relevant_rank(QUESTION, [source(1, "other.md", "400 tokens")]) is None


def test_no_results_is_a_miss_not_a_crash():
    assert first_relevant_rank(QUESTION, []) is None


def test_the_markdown_table_reports_every_configuration():
    table = as_markdown(
        {
            "vector only": Score(0.5, 0.6, 0.7, 0.55, []),
            "hybrid": Score(0.8, 0.9, 1.0, 0.85, ["missed one"]),
        },
        total=30,
    )
    assert "| vector only | 0.50 | 0.60 | 0.70 | 0.550 |" in table
    assert "| hybrid | 0.80 | 0.90 | 1.00 | 0.850 |" in table
    assert "30 questions." in table


def test_validation_catches_an_expectation_no_document_can_satisfy():
    corpus = {"ingestion.md": "Chunks are 400 tokens with overlap."}

    assert validate([QUESTION], corpus) == []
    assert validate(
        [Question("q", "missing.md", "x")], corpus
    ) == ["'q': no document at 'missing.md'"]
    assert validate([Question("q", "ingestion.md", "not here")], corpus) == [
        "'q': 'not here' is not in ingestion.md"
    ]


def test_the_shipped_golden_set_loads_and_is_not_trivially_small():
    questions = load_golden()

    assert len(questions) >= 20
    assert len({q.question for q in questions}) == len(questions), "duplicate questions"
    assert len({q.path for q in questions}) >= 4, "questions cover several documents"
    for q in questions:
        assert q.contains.strip(), f"{q.question!r} has no expected phrase"


# --- readability ----------------------------------------------------------


def test_the_readability_set_pins_the_boundary_from_both_sides():
    """A set of only must-refuse questions passes by refusing everything.

    The failure this file guards against has two directions — going blind and
    going over-cautious — so both have to be represented or the metric can be
    satisfied by a regression.
    """
    cases = load_readability()

    assert len(cases) >= 8
    assert len({c.question for c in cases}) == len(cases), "duplicate questions"

    unreadable = [c for c in cases if c.unreadable]
    readable = [c for c in cases if not c.unreadable]
    assert len(unreadable) >= 4 and len(readable) >= 4

    for case in cases:
        assert case.why.strip(), f"{case.question!r} does not say why it is here"


def test_the_readable_side_covers_what_used_to_slip_through():
    """`?` and digits survive an English vocabulary while carrying no meaning.

    A check that counted them as read would classify every Thai question ending
    in a question mark as readable, which is most of them.
    """
    unreadable = {c.question for c in load_readability() if c.unreadable}

    assert any(q.endswith("?") for q in unreadable), "no Thai question with a ?"
    assert any(re.search(r"\d", q) for q in unreadable), "no Thai question with digits"


def test_the_golden_set_alone_cannot_see_this_axis():
    """Why a second file exists rather than more rows in `golden.json`.

    Every golden question is English, so the recall table can read 1.00 while
    another script returns arbitrary chunks at 0.99 — which is what happened.
    `golden.json` measures ranking; this measures whether the question was read.
    """
    assert not any(THAI.search(q.question) for q in load_golden())
    assert any(THAI.search(c.question) for c in load_readability())


def test_the_table_reports_readability_and_names_what_failed():
    scores = {"hybrid": Score(0.8, 0.9, 1.0, 0.85, [])}

    clean = as_markdown(scores, total=30, readability=ReadabilityScore(10, 10, []))
    assert "**10/10**" in clean

    failed = as_markdown(
        scores,
        total=30,
        readability=ReadabilityScore(10, 9, [("แมวชอบกินอะไร", True)]),
    )
    assert "**9/10**" in failed
    assert "not classified as unreadable: 'แมวชอบกินอะไร'" in failed


def test_the_table_without_readability_is_unchanged():
    """`as_markdown` is called from tests and scripts that do not score it."""
    table = as_markdown({"hybrid": Score(0.8, 0.9, 1.0, 0.85, [])}, total=30)

    assert "readability" not in table.lower()
    assert "30 questions." in table
