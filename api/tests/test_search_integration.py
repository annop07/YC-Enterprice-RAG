"""Hybrid search against a real Postgres, over a corpus this test indexes.

The SQL is the interesting part of retrieval — a mocked database would test the
mock. These build a small corpus designed so each leg has something the other
one misses, then assert that fusion picks both up.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from app import db
from app.ingest.connectors import RawDocument
from app.ingest.pipeline import ingest
from app.retrieval.search import hybrid_search, keyword_tsquery
from tests.conftest import requires_db

pytestmark = [requires_db, pytest.mark.asyncio]

# "ef_search" appears verbatim only in the tuning document — a keyword query for
# it must find that chunk. "how do I start the stack" shares no vocabulary with
# "compose brings up the services", which is the vector leg's job.
CORPUS = [
    RawDocument(
        source_type="markdown",
        source_id="test/tuning.md",
        title="Tuning",
        path="test/tuning.md",
        text=(
            "# Tuning\n\n"
            "## Index parameters\n\n"
            "Set hnsw.ef_search to 100 at query time. The default trades away\n"
            "more recall than this workload can afford.\n"
        ),
    ),
    RawDocument(
        source_type="markdown",
        source_id="test/running.md",
        title="Running",
        path="test/running.md",
        text=(
            "# Running\n\n"
            "## Local environment\n\n"
            "One compose command brings up the database, the API and the web\n"
            "application together with seeded data.\n"
        ),
    ),
    RawDocument(
        source_type="markdown",
        source_id="test/rotation.md",
        title="Rotation",
        path="test/rotation.md",
        text=(
            "# Rotation\n\n"
            "## Handover\n\n"
            "The weekday support rotation is one primary and one secondary,\n"
            "with handover in the support channel at ten in the morning.\n"
        ),
    ),
]


@pytest_asyncio.fixture(scope="module", loop_scope="session", autouse=True)
async def corpus(pool):
    # Own database (see conftest), so the corpus these tests search is exactly
    # the three documents below and nothing else.
    async with db.pool().connection() as conn:
        await conn.execute("TRUNCATE document CASCADE")
    await ingest(CORPUS, force=True)
    yield


async def test_an_exact_token_is_found_even_though_it_is_a_bad_embedding_target():
    """`ef_search` is the kind of query embeddings are worst at."""
    result = await hybrid_search("hnsw.ef_search", top_k=3)

    top = result.sources[0]
    assert top.path == "test/tuning.md"
    assert top.retrieval.keyword_rank is not None, "the keyword leg should have found it"


async def test_a_paraphrase_with_no_shared_words_still_retrieves():
    result = await hybrid_search("how do I start everything on my laptop?", top_k=3)

    paths = [s.path for s in result.sources]
    assert "test/running.md" in paths

    running = next(s for s in result.sources if s.path == "test/running.md")
    assert running.retrieval.vector_rank is not None, "the vector leg should have found it"


async def test_sources_are_numbered_from_one_and_ordered_by_the_reranker():
    result = await hybrid_search("support rotation handover", top_k=3)

    assert [s.n for s in result.sources] == list(range(1, len(result.sources) + 1))
    scores = [s.retrieval.rerank_score for s in result.sources]
    assert scores == sorted(scores, reverse=True)


async def test_every_source_carries_a_locator_that_points_at_real_lines():
    result = await hybrid_search("compose seeded data", top_k=3)

    for source in result.sources:
        row = await db.fetch_one(
            "SELECT text FROM document WHERE id = %s", (source.document_id,)
        )
        lines = row[0].split("\n")
        quoted = "\n".join(
            lines[source.locator.line_start - 1 : source.locator.line_end]
        )
        assert source.snippet == quoted


async def test_fusion_reports_how_many_candidates_it_weighed():
    result = await hybrid_search("compose", top_k=1)

    assert result.candidates_considered >= len(result.sources)
    assert result.retrieval_ms >= 0


async def test_a_query_matching_nothing_returns_nothing_rather_than_noise():
    result = await hybrid_search("zzzqqq-not-a-word-in-this-corpus", top_k=5)

    # The vector leg always returns its nearest neighbours, so results are
    # expected — what matters is that they score as irrelevant.
    for source in result.sources:
        assert source.retrieval.keyword_rank is None
        assert source.retrieval.rerank_score < 0.5


async def test_an_empty_query_short_circuits():
    result = await hybrid_search("   ", top_k=5)
    assert result.sources == []
    assert result.candidates_considered == 0


# --- B-08: the index is not stemmed, so the query is a prefix --------------


async def test_a_word_form_the_document_does_not_use_still_reaches_it():
    """The corpus says "brings up"; someone asking about it types "bring".

    `simple` was chosen over `english` because the stemmer mangles every
    language that is not English, and the cost of that choice is that no word
    form matches any other. The stemmed `tsv_en` column pays it back without
    the stemmer touching the faithful copy.
    """
    indexed = "One compose command brings up the database"

    exact = await db.fetch_one(
        "SELECT to_tsvector('simple', %s) @@ to_tsquery('simple', %s)",
        (indexed, keyword_tsquery("bring")),
    )
    stemmed = await db.fetch_one(
        "SELECT to_tsvector('english', %s) @@ to_tsquery('english', %s)",
        (indexed, keyword_tsquery("bring")),
    )

    assert exact[0] is False, "the bug: 'bring' does not match 'brings'"
    assert stemmed[0] is True, "which the stemmed column is there to fix"

    result = await hybrid_search("which command brings the stack up?", top_k=3)
    running = next((s for s in result.sources if s.path == "test/running.md"), None)
    assert running is not None, "the document with the answer is not in the results"


async def test_a_stemmed_match_never_outranks_an_exact_one():
    """Which is what makes the second column free.

    `ts_rank_cd` has no IDF, so scoring the two columns as equals let a stemmed
    match displace an exact one — measured on the golden set as Recall@1 0.50
    -> 0.47. Ordering by `exact_hit` first means the stemmed column can only
    ever fill the tail of the candidate list.
    """
    # "handover" appears verbatim in the rotation document, so it is an exact
    # hit. "bring" appears nowhere: the running document says "brings", which
    # only the stemmed column can reach.
    result = await hybrid_search("handover bring", top_k=3)

    ranked = {s.path: s.retrieval.keyword_rank for s in result.sources}
    assert ranked.get("test/rotation.md") == 1, f"exact match lost rank 1: {ranked}"
    assert ranked.get("test/running.md") == 2, f"stemmed match not behind it: {ranked}"


@pytest.mark.parametrize(
    "question",
    [
        "how do I run the whole stack locally?",
        "set hnsw.ef_search to 100",
        "the X-Accel-Buffering header",
        "BAAI/bge-small-en-v1.5 dimensions",
        "ตั้งค่า chunk ไว้เท่าไร",
        "Tiếng Việt tổ hợp",
        "café",
        "'; DROP TABLE chunk; --",
        "drop & everything | now ! ( ) <-> :* ''",
        "what is it about?",
    ],
)
async def test_every_query_this_builds_is_a_tsquery_postgres_accepts(question: str):
    """`to_tsquery` raises on malformed input, and this string is built by hand
    from arbitrary user text — a term rule that emits one stray operator turns
    every search into a 500. Both configurations parse the same string now, so
    both are checked.
    """
    row = await db.fetch_one(
        """
        SELECT to_tsquery('simple', %(q)s)::text, to_tsquery('english', %(q)s)::text
        """,
        {"q": keyword_tsquery(question)},
    )
    assert row is not None
