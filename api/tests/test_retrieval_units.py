"""Retrieval pieces that need neither a database nor a model."""
from __future__ import annotations

import math
import unicodedata

import pytest

from app.retrieval.reranker import relevance_probability
from app.retrieval.search import (
    HYBRID_SQL,
    Candidate,
    anchored_url,
    keyword_tsquery,
    to_source,
)


def make_candidate(**overrides) -> Candidate:
    base = dict(
        chunk_id=41,
        document_id="doc1",
        title="Ingestion Pipeline",
        source_type="github",
        path="docs/ingestion.md",
        doc_url="https://github.com/acme/handbook/blob/7f3a91c/docs/ingestion.md",
        heading_path="Ingestion Pipeline > Chunking",
        line_start=18,
        line_end=24,
        page=None,
        content="Chunks are 400 tokens with 80 tokens of overlap.",
        vector_rank=1,
        keyword_rank=3,
        rrf_score=0.0325,
        rerank_score=0.94,
    )
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


def test_keyword_query_ors_its_terms_rather_than_anding_them():
    """AND means a whole question has to appear verbatim — it never does."""
    assert keyword_tsquery("compose seeded data") == "'compose' | 'seeded' | 'data'"


def test_keyword_query_drops_function_words():
    """`ts_rank_cd` has no IDF, so "the" would score as loudly as "compose"."""
    assert keyword_tsquery("how do I run the whole stack locally?") == (
        "'run' | 'whole' | 'stack' | 'locally'"
    )


def test_keyword_query_keeps_identifiers_intact():
    assert keyword_tsquery("set hnsw.ef_search to 100") == (
        "'set' | 'hnsw.ef_search' | '100'"
    )
    assert keyword_tsquery("the X-Accel-Buffering header") == (
        "'x-accel-buffering' | 'header'"
    )


def test_keyword_query_deduplicates_and_is_case_insensitive():
    assert keyword_tsquery("Compose compose COMPOSE up") == "'compose' | 'up'"


def test_keyword_query_cannot_inject_tsquery_syntax():
    """Operators in the input become nothing; every term ships quoted."""
    assert keyword_tsquery("drop & everything | now !") == (
        "'drop' | 'everything' | 'now'"
    )
    # An apostrophe is not a term character, so it cannot close the quoting
    # around a term — it splits it, and the leftover "s" is too short to keep.
    assert keyword_tsquery("nick's") == "'nick'"
    # `:*` in the input is not an instruction either — it is punctuation, and
    # punctuation is not a term character.
    assert keyword_tsquery("a:*b | drop:*") == "'drop'"


def test_a_query_of_only_function_words_yields_an_empty_tsquery():
    """Which Postgres accepts and matches nothing — the vector leg carries it."""
    assert keyword_tsquery("what is it about?") == ""
    assert keyword_tsquery("   ") == ""


# --- B-08: an unstemmed index, and what did *not* fix it ------------------


def test_terms_are_not_prefix_matched():
    """The obvious fix for an unstemmed index is `'chunk':*`, and it was tried.

    Measured on the 30-question golden set at a minimum length of four, five,
    six and seven characters, every variant made retrieval worse than leaving
    it alone — keyword-only MRR 0.640 -> 0.579 at four, and no threshold got
    back to the baseline. `ts_rank_cd` has no IDF, so the extra words a prefix
    matches score as loudly as the exact term and blur the ranking. Word forms
    are recovered in the SQL instead, by the stemmed `tsv_en` column, which
    cannot outrank an exact match.
    """
    assert keyword_tsquery("chunk") == "'chunk'"
    assert keyword_tsquery("deployment") == "'deployment'"
    assert ":*" not in keyword_tsquery("configuration settings documented")


# --- B-04: marks are part of the word --------------------------------------


def test_thai_words_survive_their_vowel_signs_and_tone_marks():
    """`[^\\W\\d_]` rejects category-Mn characters, which cut every Thai word
    into the fragments between its marks — `['งค', 'าไว', 'เท', 'าไร']` — while
    Postgres keeps the run whole, so nothing could ever match."""
    assert keyword_tsquery("ตั้งค่าไว้เท่าไร") == "'ตั้งค่าไว้เท่าไร'"
    assert keyword_tsquery("chunk size ของ ingestion") == (
        "'chunk' | 'size' | 'ของ' | 'ingestion'"
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tiếng Việt", "'tiếng' | 'việt'"),
        # Two alternatives split a word at the first non-ASCII letter and the
        # one-character tail was then dropped for being too short: "café" used
        # to leave only "caf".
        ("café", "'café'"),
        ("दस्तावेज़", "'दस्तावेज़'"),
    ],
)
def test_a_word_is_never_split_at_its_first_non_ascii_letter(text: str, expected: str):
    assert keyword_tsquery(text) == expected


def test_a_decomposed_word_is_one_term_and_not_one_per_accent():
    """The same text in NFD is a letter followed by two combining marks per
    vowel. The lexeme stays in the form it arrived in — normalising is the
    index's business, not this function's — but it stays *whole*, which is the
    property that was broken.
    """
    terms = keyword_tsquery(unicodedata.normalize("NFD", "Tiếng Việt")).split(" | ")
    assert [unicodedata.normalize("NFC", t) for t in terms] == ["'tiếng'", "'việt'"]


def test_relevance_probability_is_a_sigmoid_and_survives_extreme_logits():
    assert relevance_probability(0.0) == 0.5
    assert relevance_probability(6.0) > 0.99
    assert relevance_probability(-11.0) < 0.01
    # math.exp(-logit) overflows around -746; the UI still needs a number.
    assert relevance_probability(-1000.0) == 0.0
    assert 0.0 <= relevance_probability(1000.0) <= 1.0
    assert math.isclose(relevance_probability(2.0), 1 / (1 + math.exp(-2.0)))


def test_relevance_probability_preserves_order():
    logits = [-11.2, -10.7, -6.5, 0.4, 5.0]
    scores = [relevance_probability(x) for x in logits]
    assert scores == sorted(scores)


def test_citations_deep_link_to_the_lines_they_quote():
    url = anchored_url("https://github.com/a/b/blob/sha/docs/x.md", 18, 24)
    assert url == "https://github.com/a/b/blob/sha/docs/x.md#L18-L24"


def test_a_single_line_citation_still_gets_a_range():
    assert anchored_url("https://example.com/x.md", 7, 7).endswith("#L7-L7")
    assert anchored_url("https://example.com/x.md", 7, None).endswith("#L7-L7")


def test_documents_without_a_url_or_lines_are_left_alone():
    assert anchored_url(None, 1, 2) is None
    # A PDF has a page, not a line range — nothing to anchor to.
    assert anchored_url("https://example.com/x.pdf", None, None) == "https://example.com/x.pdf"


def test_source_carries_the_locator_the_retrieval_trace_and_the_quoted_text():
    source = to_source(make_candidate(), n=2)

    assert source.n == 2
    assert source.chunk_id == "41"
    assert source.locator.line_start == 18 and source.locator.page is None
    assert source.retrieval.vector_rank == 1 and source.retrieval.keyword_rank == 3
    assert source.retrieval.rerank_score == 0.94
    assert source.url.endswith("#L18-L24")
    assert source.snippet.startswith("Chunks are 400 tokens")


def test_a_chunk_only_one_leg_found_reports_the_other_as_missing():
    """This is the evidence that the search is hybrid rather than a claim."""
    source = to_source(make_candidate(keyword_rank=None), n=1)
    assert source.retrieval.vector_rank == 1
    assert source.retrieval.keyword_rank is None


def test_pdf_sources_report_a_page_and_no_anchor():
    source = to_source(
        make_candidate(
            source_type="pdf",
            path="handbook.pdf",
            doc_url=None,
            line_start=9,
            line_end=16,
            page=2,
        ),
        n=1,
    )
    assert source.locator.page == 2
    assert source.url is None


def test_every_ordering_carries_the_reindex_stable_tiebreak():
    """Cheap guard for the property `test_search_determinism.py` proves.

    That module needs Postgres and is skipped without it, so this pins the
    invariant in a test that always runs: none of the three scores is unique,
    so each ORDER BY must name `(document_id, ordinal)` after it. `chunk.id`
    would be wrong — it is BIGSERIAL and a re-ingest reassigns it.
    """
    clauses = []
    for line in HYBRID_SQL.splitlines():
        if "ORDER BY" not in line:
            continue
        clause = line.split("ORDER BY", 1)[1]
        # Window clauses sit inside `OVER (...) AS rank`; keep only the keys.
        clause = clause.split(")")[0] if ") AS rank" in line else clause
        clauses.append(clause.strip().replace("c.", ""))

    assert len(clauses) == 5, f"orderings changed shape: {clauses}"

    # The vector leg's inner ORDER BY is deliberately bare: sort keys there turn
    # the HNSW `Index Scan ... Order By` into a `Seq Scan + Sort`. It is ranked
    # by the window above it, which does carry the tiebreak, so the leg is still
    # deterministic over the rows it returns.
    bare = [c for c in clauses if not c.endswith("document_id, ordinal")]
    assert bare == ["distance"], f"ordering without a stable tiebreak: {bare}"
