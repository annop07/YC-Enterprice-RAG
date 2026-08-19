"""Whether retrieval claims to have found an answer, or only the nearest rows.

Fusion returns its best twenty however bad the best twenty are — rank has no
opinion about whether anything actually answers the question. So a question
this corpus knows nothing about came back as five citation cards with a zeroed
relevance bar, which reads exactly like five good ones, and an LLM was asked
to write over them.

The fix is a caveat, not a filter, and that is measured rather than chosen:
see `MIN_RERANK_SCORE`. On this cross-encoder a correct paraphrase retrieval
scores 1.5e-5 — below every off-topic question in a twelve-question set — so a
floor that dropped rows would delete right answers to hide wrong ones.
"""
from __future__ import annotations

import pytest

from app.config import get_settings
from app.retrieval.search import (
    LOW_CONFIDENCE_NOTICE,
    UNREADABLE_QUERY_NOTICE,
    SearchResult,
    notice_for,
)


def result(top: float | None, sources: list | None = None) -> SearchResult:
    return SearchResult(
        sources=sources if sources is not None else ["a source"],
        candidates_considered=20,
        retrieval_ms=1,
        top_rerank_score=top,
    )


def test_a_confident_hit_is_not_flagged():
    assert result(0.99).low_confidence is False
    assert notice_for(result(0.99)) is None


def test_nothing_scoring_above_the_threshold_is_flagged():
    floor = get_settings().min_rerank_score
    assert result(floor / 10).low_confidence is True
    assert notice_for(result(floor / 10)) == LOW_CONFIDENCE_NOTICE


def test_the_threshold_itself_counts_as_confident():
    assert result(get_settings().min_rerank_score).low_confidence is False


def test_a_run_with_no_re_ranking_makes_no_claim_either_way():
    """The eval harness's un-reranked legs have no score to judge, so they are
    not flagged — a missing signal is not a negative one."""
    assert result(None).low_confidence is False
    assert notice_for(result(None)) is None


def test_low_confidence_keeps_its_sources():
    """The point of the whole design: the caveat is printed *over* the results.

    A floor that emptied the list would turn a visible weak answer into an
    invisible missing one, and the measurement says the score cannot tell a
    reworded right answer from a wrong one.
    """
    weak = result(0.000001)
    assert weak.low_confidence and weak.sources


def test_being_unable_to_read_the_question_outranks_being_unsure():
    """Both are true at once for an unreadable question — nothing was ranked,
    so nothing scored — and the reader needs the one with a fix attached."""
    unreadable = SearchResult(
        sources=[],
        candidates_considered=3,
        retrieval_ms=1,
        unreadable_query=True,
        top_rerank_score=None,
    )
    assert notice_for(unreadable) == UNREADABLE_QUERY_NOTICE
