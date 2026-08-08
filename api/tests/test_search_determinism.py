"""Retrieval order must not depend on physical row order.

`ts_rank_cd` has no IDF and quantises hard, so tied scores are the norm rather
than the exception, and RRF ties whenever two chunks hold the same rank in
opposite legs. A re-ingest deletes and re-inserts every chunk row (see
`pipeline.py`), which reassigns the BIGSERIAL primary key and changes the order
the heap yields tied rows in. Without a deterministic tiebreak the same corpus
then measures differently from one index to the next — a determinism defect
that reads as measurement noise.

These tests re-index the corpus *in reverse* between two identical searches, so
the physical order is as different as it can be while the content is unchanged.
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

# Every document mentions "beacon" exactly once in the same shape, so the
# keyword leg scores them identically and has nothing but the tiebreak to
# separate them. The trailing sentences differ only to keep the embeddings
# distinct, so the vector leg does not tie in the same places.
CORPUS = [
    RawDocument(
        source_type="markdown",
        source_id=f"test/beacon-{name}.md",
        title=f"Beacon {name.title()}",
        path=f"test/beacon-{name}.md",
        text=(
            f"# Beacon {name.title()}\n\n"
            f"The beacon is described here for the {name} case.\n\n"
            f"{filler}\n"
        ),
    )
    for name, filler in [
        ("alpha", "Mountains, rivers and the long coastal road."),
        ("bravo", "Compilers, linkers and the symbol table."),
        ("charlie", "Bread, yeast and a very slow overnight prove."),
        ("delta", "Telescopes, parallax and the distance ladder."),
        ("echo", "Kettles, pressure and the whistle of escaping steam."),
        ("foxtrot", "Harbours, tides and the timing of the ferry."),
    ]
]


@pytest_asyncio.fixture(scope="module", loop_scope="session", autouse=True)
async def corpus(pool):
    async with db.pool().connection() as conn:
        await conn.execute("TRUNCATE document CASCADE")
    await ingest(CORPUS, force=True)
    yield


async def _chunk_ids() -> list[int]:
    rows = await db.fetch_all("SELECT id FROM chunk ORDER BY id")
    return [r[0] for r in rows]


async def _reindex_in_reverse() -> None:
    """Same content, deliberately different physical order."""
    await ingest(list(reversed(CORPUS)), force=True)


async def _order(query: str, legs: str) -> list[tuple[str, int]]:
    """The result order, keyed by something a re-index cannot change.

    Not `chunk_id` — that is the BIGSERIAL this whole test is about. `path`
    plus the chunk's first line survives a re-index and names the same chunk.
    """
    result = await hybrid_search(query, top_k=10, legs=legs, rerank=False)
    return [(s.path, s.locator.line_start) for s in result.sources]


async def test_the_keyword_leg_really_does_tie_on_this_corpus():
    """Guard: without ties the determinism tests below would pass vacuously."""
    rows = await db.fetch_all(
        """
        SELECT ts_rank_cd(c.tsv, q) AS score, count(*)
        FROM chunk c, to_tsquery('simple', %s) AS q
        WHERE c.tsv @@ q
        GROUP BY 1 ORDER BY 1 DESC
        """,
        (keyword_tsquery("beacon"),),
    )
    assert rows, "the query should match something"
    assert any(count > 1 for _, count in rows), (
        f"expected tied ts_rank_cd scores, got groups {rows}"
    )


async def test_a_reindex_does_not_reorder_the_keyword_leg():
    before = await _order("beacon", "keyword")
    ids_before = await _chunk_ids()

    await _reindex_in_reverse()

    ids_after = await _chunk_ids()
    assert ids_before != ids_after, (
        "the re-index did not reassign chunk ids, so this test proves nothing"
    )
    assert await _order("beacon", "keyword") == before


async def test_a_reindex_does_not_reorder_fusion():
    before = await _order("beacon", "hybrid")
    ids_before = await _chunk_ids()

    await _reindex_in_reverse()

    assert await _chunk_ids() != ids_before
    assert await _order("beacon", "hybrid") == before


async def test_repeated_identical_searches_agree():
    first = await _order("beacon described here", "hybrid")
    assert await _order("beacon described here", "hybrid") == first
