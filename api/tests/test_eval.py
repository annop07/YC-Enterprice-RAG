"""Scoring and golden-set validation.

The measurement itself needs a corpus; what is tested here is that the metrics
are computed correctly and that a mistyped expectation is caught rather than
reported as a retrieval failure.
"""
from __future__ import annotations

from app.eval.runner import Question, Score, as_markdown, first_relevant_rank, load_golden, validate
from app.schemas import Locator, RetrievalTrace, Source


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
