"""Hybrid retrieval: two legs, fused, then re-ranked.

The vector leg finds paraphrases — a question phrased "how do I ship this"
reaches a section titled "Deployment". The keyword leg finds exact tokens —
error codes, flag names, a config key spelled one particular way — which is
where embeddings are weakest. Neither alone is enough, and the ranks of both
are carried through to the citation so the claim is checkable.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

import numpy as np

from app import db
from app.config import get_settings
from app.ingest.embedder import get_embedder
from app.retrieval.reranker import get_reranker
from app.schemas import Locator, RetrievalTrace, Source

#: Both legs are ordered and limited in a subquery so the indexes can serve the
#: ordering, then ranked in the outer query. Ranking before the limit would
#: force a full sort and throw away the HNSW and GIN indexes.
#:
#: Fusion is Reciprocal Rank Fusion: sum of 1 / (k + rank) over the legs a chunk
#: appears in. It reads positions only, so cosine distance and ts_rank never
#: have to be normalised onto a shared scale — which is exactly where a
#: hand-weighted hybrid score usually goes wrong.
#:
#: Every ordering carries `(document_id, ordinal)` as a tiebreak, because none
#: of the three scores is unique. `ts_rank_cd` is the worst offender — it has no
#: IDF and quantises hard, so a real query here returns 28 chunks sharing 7
#: distinct scores — and RRF ties whenever two chunks hold the same rank in
#: opposite legs. Without a tiebreak the order of tied rows is whatever the heap
#: yields, which a re-index reshuffles: the same corpus then measures
#: differently, and "reproducible from a stated corpus" stops being true.
#:
#: The key is `(document_id, ordinal)`, NOT `chunk.id`. `chunk.id` is BIGSERIAL
#: and a re-ingest deletes and re-inserts every row, so it is reassigned by
#: insertion time — deterministic within one index generation and reshuffled by
#: the next, which is precisely the bug. `document.id` is a hash of
#: `source_type:source_id` and `ordinal` is the chunk's position in its
#: document, so the pair survives a re-index and `UNIQUE (document_id, ordinal)`
#: makes it a total order.
#:
#: The vector leg tiebreaks in the window rather than in its inner ORDER BY:
#: adding sort keys there turns the HNSW `Index Scan ... Order By` into a
#: `Seq Scan + Sort` (checked with EXPLAIN), which is the cost this module's
#: subquery shape exists to avoid. The keyword leg has no such constraint — GIN
#: cannot order by `ts_rank_cd` at all, so that sort is already happening and
#: the extra keys are free; it tiebreaks in both places, which also pins which
#: rows survive its LIMIT.
HYBRID_SQL = """
WITH vec AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY distance, document_id, ordinal) AS rank
    FROM (
        SELECT id, document_id, ordinal, embedding <=> %(vector)s AS distance
        FROM chunk
        ORDER BY distance
        LIMIT %(vector_limit)s
    ) ranked
),
kw AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC, document_id, ordinal) AS rank
    FROM (
        SELECT c.id, c.document_id, c.ordinal, ts_rank_cd(c.tsv, q) AS score
        FROM chunk c, to_tsquery('simple', %(keywords)s) AS q
        WHERE c.tsv @@ q
        ORDER BY score DESC, c.document_id, c.ordinal
        LIMIT %(keyword_limit)s
    ) ranked
),
fused AS (
    SELECT
        COALESCE(v.id, k.id) AS chunk_id,
        v.rank AS vector_rank,
        k.rank AS keyword_rank,
        COALESCE(1.0 / (%(rrf_k)s + v.rank), 0)
      + COALESCE(1.0 / (%(rrf_k)s + k.rank), 0) AS rrf_score
    FROM vec v
    FULL OUTER JOIN kw k ON v.id = k.id
)
SELECT
    f.chunk_id, f.vector_rank, f.keyword_rank, f.rrf_score,
    c.content, c.heading_path, c.line_start, c.line_end, c.page,
    d.id, d.title, d.source_type, d.path, d.url
FROM fused f
JOIN chunk c ON c.id = f.chunk_id
JOIN document d ON d.id = c.document_id
ORDER BY f.rrf_score DESC, c.document_id, c.ordinal
LIMIT %(keep)s
"""


#: Function words that carry no retrieval signal. Postgres's `simple`
#: configuration has no stopword list at all — that is the price of using it,
#: and the reason it is used: the `english` stemmer would mangle every language
#: that is not English. Filtering here rather than in the index keeps the
#: indexed text faithful and the query cheap to change.
STOPWORDS = frozenset(
    """
    a an the this that these those and or but if then else when while for of to
    in on at by with from into about as is are was were be been being do does
    did done have has had can could should would will shall may might must
    it its i you he she they we me my your our their there here what which who
    whom how why where all any some no not so than too very just only own same
    """.split()
)

#: Words, identifiers and dotted names. Non-ASCII runs are kept whole, which is
#: enough for the vector leg to carry Thai but not enough for the keyword leg —
#: Thai is unsegmented, so a sentence becomes one useless token. That needs a
#: real segmenter before this leg is worth anything on Thai text.
TERM = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*|[^\W\d_]{2,}", re.UNICODE)


def keyword_tsquery(query: str) -> str:
    """Turn a question into an OR query over its content words.

    Two things are deliberate.

    **OR, not AND.** `websearch_to_tsquery` and `plainto_tsquery` both AND their
    terms, so "how do I run the whole stack locally?" has to appear *in full* in
    a chunk to match anything — it never does, and the keyword leg silently
    contributes nothing to every natural-language question. Measured on this
    corpus: 0 rows for the AND form, the right chunk ranked first for the OR
    form.

    **Stopwords removed.** `ts_rank_cd` has no IDF, so with stopwords left in,
    "the" and "do" score exactly as loudly as "compose" and the ranking becomes
    noise — measured, again: the correct chunk fell out of the top five.
    """
    terms: list[str] = []
    seen: set[str] = set()

    for match in TERM.finditer(query.lower()):
        term = match.group().strip(".-")
        if len(term) < 2 or term in STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)

    # Quoting each lexeme means a term containing punctuation cannot be read as
    # tsquery syntax. An empty string is a valid, matches-nothing tsquery.
    return " | ".join("'" + t.replace("'", "") + "'" for t in terms)


@dataclass
class Candidate:
    chunk_id: int
    document_id: str
    title: str
    source_type: str
    path: str
    doc_url: str | None
    heading_path: str | None
    line_start: int | None
    line_end: int | None
    page: int | None
    content: str
    vector_rank: int | None
    keyword_rank: int | None
    rrf_score: float
    rerank_score: float | None = None


@dataclass
class SearchResult:
    sources: list[Source]
    candidates_considered: int
    retrieval_ms: int


def anchored_url(doc_url: str | None, line_start: int | None, line_end: int | None) -> str | None:
    """Deep-link a citation to the lines it quotes.

    The document URL already pins a commit, so adding the line anchor produces
    a link that opens on the exact text that was indexed rather than on the
    top of a file that may since have moved underneath it.
    """
    if not doc_url or line_start is None:
        return doc_url
    end = line_end if line_end is not None else line_start
    return f"{doc_url}#L{line_start}-L{end}"


def to_source(candidate: Candidate, n: int) -> Source:
    return Source(
        n=n,
        chunk_id=str(candidate.chunk_id),
        document_id=candidate.document_id,
        title=candidate.title,
        source_type=candidate.source_type,  # type: ignore[arg-type]
        path=candidate.path,
        url=anchored_url(candidate.doc_url, candidate.line_start, candidate.line_end),
        heading_path=candidate.heading_path,
        locator=Locator(
            line_start=candidate.line_start,
            line_end=candidate.line_end,
            page=candidate.page,
        ),
        retrieval=RetrievalTrace(
            vector_rank=candidate.vector_rank,
            keyword_rank=candidate.keyword_rank,
            rrf_score=round(candidate.rrf_score, 6),
            rerank_score=(
                round(candidate.rerank_score, 4)
                if candidate.rerank_score is not None
                else None
            ),
        ),
        snippet=candidate.content,
    )


async def fuse_candidates(
    query: str,
    vector: list[float],
    per_leg: int,
    keep: int,
    *,
    vector_limit: int | None = None,
    keyword_limit: int | None = None,
) -> list[Candidate]:
    """Run both legs and fuse them.

    A leg is disabled by giving it a limit of zero rather than by branching the
    SQL — the evaluation harness compares vector-only, keyword-only and hybrid,
    and all three have to be the *same query* for the comparison to mean
    anything.
    """
    settings = get_settings()
    rows = await db.fetch_all(
        HYBRID_SQL,
        {
            "vector": np.asarray(vector, dtype=np.float32),
            "keywords": keyword_tsquery(query),
            "vector_limit": per_leg if vector_limit is None else vector_limit,
            "keyword_limit": per_leg if keyword_limit is None else keyword_limit,
            "rrf_k": settings.rrf_k,
            "keep": keep,
        },
    )
    return [
        Candidate(
            chunk_id=r[0],
            vector_rank=r[1],
            keyword_rank=r[2],
            rrf_score=float(r[3]),
            content=r[4],
            heading_path=r[5],
            line_start=r[6],
            line_end=r[7],
            page=r[8],
            document_id=r[9],
            title=r[10],
            source_type=r[11],
            path=r[12],
            doc_url=r[13],
        )
        for r in rows
    ]


async def hybrid_search(
    query: str,
    *,
    top_k: int | None = None,
    legs: str = "hybrid",
    rerank: bool = True,
) -> SearchResult:
    """`legs` is "hybrid", "vector" or "keyword" — the evaluation harness uses
    the narrower ones to measure what each contributes."""
    settings = get_settings()
    top_k = top_k or settings.top_k
    started = time.perf_counter()

    query = query.strip()
    if not query:
        return SearchResult([], 0, 0)

    embedder = get_embedder()
    vector = await asyncio.to_thread(embedder.embed_query, query)

    per_leg = settings.candidates_per_leg
    candidates = await fuse_candidates(
        query,
        vector,
        per_leg,
        settings.fusion_keep,
        vector_limit=0 if legs == "keyword" else per_leg,
        keyword_limit=0 if legs == "vector" else per_leg,
    )
    considered = len(candidates)

    if candidates and rerank:
        reranker = get_reranker()
        scores = await asyncio.to_thread(
            reranker.score, query, [c.content for c in candidates]
        )
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score
        # `list.sort` is stable, so equal re-rank scores keep the fused order —
        # which the SQL tiebreak now makes deterministic. That is the whole
        # chain: without it this sort would quietly inherit the heap's order.
        candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

    elapsed = int((time.perf_counter() - started) * 1000)
    return SearchResult(
        sources=[to_source(c, i + 1) for i, c in enumerate(candidates[:top_k])],
        candidates_considered=considered,
        retrieval_ms=elapsed,
    )
