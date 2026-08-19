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
#:
#: The keyword leg reads two columns, and `exact_hit` is what keeps that from
#: costing anything. `tsv` is the faithful `simple` index and `tsv_en` the
#: English-stemmed one, so a question about "the chunk size" reaches a chunk
#: that only ever says "chunks" — which `simple` alone cannot do, and which is
#: half of what people actually type. Ranking them as equals is what fails:
#: `ts_rank_cd` has no IDF, so a stemmed match scores as loudly as an exact one
#: and displaces it. Measured on the golden set, ranking them together cost
#: keyword-only Recall@1 0.50 -> 0.47 and MRR 0.640 -> 0.605. Sorting by
#: `exact_hit` first makes the stemmed column strictly additive: it can fill
#: the tail of the candidate list, never the head, and the golden set then
#: reproduces every cell of the previous table exactly.
#:
#: The `OR` costs nothing structural — checked with EXPLAIN, it plans as a
#: `BitmapOr` over the two GIN indexes. At this corpus size the planner reads
#: the whole table instead, as it did with one column and will for any table
#: this small.
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
    SELECT id, ROW_NUMBER() OVER (ORDER BY exact_hit DESC, score DESC, document_id, ordinal) AS rank
    FROM (
        SELECT c.id, c.document_id, c.ordinal,
               (c.tsv @@ q.exact) AS exact_hit,
               CASE WHEN c.tsv @@ q.exact
                    THEN ts_rank_cd(c.tsv, q.exact)
                    ELSE ts_rank_cd(c.tsv_en, q.stemmed)
               END AS score
        FROM chunk c,
             (SELECT to_tsquery('simple', %(keywords)s) AS exact,
                     to_tsquery('english', %(keywords)s) AS stemmed) q
        WHERE c.tsv @@ q.exact OR c.tsv_en @@ q.stemmed
        ORDER BY exact_hit DESC, score DESC, c.document_id, c.ordinal
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

#: Combining marks — Unicode category M. Python's `re` has no `\p{M}` and
#: deriving the property at import time means walking the whole code space, so
#: the ranges that reach a corpus like this one are listed instead.
#:
#: They have to be in `TERM` because a mark is not a letter: `str.isalnum()` is
#: False for every Thai vowel sign and tone mark, so `[^\W\d_]` rejects them and
#: a word came apart at each one — "ตั้งค่าไว้เท่าไร" tokenised to
#: `['งค', 'าไว', 'เท', 'าไร']`, none of which is a word. Postgres's `simple`
#: parser meanwhile keeps the whole run as one lexeme, so those fragments could
#: not match the index no matter what was indexed. Decomposed Vietnamese,
#: Devanagari and pointed Arabic and Hebrew break the same way.
MARKS = (
    "̀-ͯ"  # Latin/Greek/Cyrillic diacritics; decomposed Vietnamese
    "҃-҉"  # Cyrillic
    "֑-ׇֽֿׁׂׅׄ"  # Hebrew points
    "ؐ-ًؚ-ٰٟۖ-ۜ"  # Arabic
    "ऀ-ःऺ-ॏ॑-ॗॢॣ"  # Devanagari
    "ัิ-ฺ็-๎"  # Thai vowels above/below, tone marks
    "ັິ-ຼ່-ໍ"  # Lao
    "ါ-ှ"  # Myanmar
    "឴-៓"  # Khmer
    "᪰-᫿᷀-᷿⃐-⃰︠-︯"  # extended marks
)

#: Words, identifiers and dotted names, in one alternative rather than two.
#: Two alternatives split a word at the first non-ASCII letter, because the
#: regex engine takes the first branch that matches and not the longest one:
#: "Tiếng" came out as `['ti', 'ếng']` and "café" as `['caf']` — the accented
#: tail dropped for being a single character. One class of "letter, mark, digit
#: or joiner" keeps every run whole, which is what the `simple` parser indexes.
#:
#: Whole is as far as this goes for unsegmented scripts: a Thai sentence is
#: still one token and matches only a chunk containing that same run. Findable,
#: not searchable — a segmenter on both sides is what would make this leg worth
#: anything on Thai prose.
_LETTER = "(?:[^\\W\\d_]|[" + MARKS + "])"
TERM = re.compile(
    rf"(?:{_LETTER}|[0-9_])(?:{_LETTER}|[0-9_.\-])*",
    re.UNICODE,
)

def _lexeme(term: str) -> str:
    """One quoted tsquery lexeme.

    Quoting means a term containing punctuation cannot be read as tsquery
    syntax; the apostrophe is dropped rather than escaped because it cannot
    survive as a term character anyway.

    Deliberately *not* prefix-matched. Appending `:*` is the obvious answer to
    an unstemmed index — "chunk" would then find "chunks" — and it was measured
    on the golden set at four, five, six and seven characters minimum. Every
    one of them made retrieval worse: at four, keyword-only MRR fell from 0.640
    to 0.579 and hybrid RRF from 0.806 to 0.786, and no threshold recovered the
    baseline. The reason is the one this module's header already gives for
    needing a tiebreak: `ts_rank_cd` has no IDF, so a prefix that matches five
    extra words scores as loudly as the exact term did, and the ranking blurs.
    The word-form problem is solved in the SQL instead, by `tsv_en` — see
    `HYBRID_SQL`, where a stemmed match can only ever rank *below* an exact one.
    """
    return "'" + term.replace("'", "") + "'"


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

    The same string is given to both text-search configurations in
    `HYBRID_SQL`: `simple` reads it literally, `english` stems each lexeme.
    """
    terms: list[str] = []
    seen: set[str] = set()

    for match in TERM.finditer(query.lower()):
        term = match.group().strip(".-")
        if len(term) < 2 or term in STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)

    # An empty string is a valid tsquery that matches nothing — which is the
    # right answer for a question made entirely of function words.
    return " | ".join(_lexeme(t) for t in terms)


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
    #: The embedding model could not read the question (see `Embedder.reads`),
    #: so the vector leg was disabled and nothing was re-ranked. Surfaced rather
    #: than hidden: the caller has to be able to tell "we found nothing" apart
    #: from "we could not look".
    unreadable_query: bool = False
    #: The best relevance probability the cross-encoder gave anything, or None
    #: when it did not run (an un-reranked configuration, or an unreadable
    #: question). This is the number that says whether retrieval found an
    #: answer or merely found its twenty nearest rows — fusion returns its
    #: best twenty however bad they are, and rank has no opinion about that.
    top_rerank_score: float | None = None

    @property
    def low_confidence(self) -> bool:
        """Nothing retrieved was scored as clearly relevant.

        Reported, not acted on: see `MIN_RERANK_SCORE` for the measurement
        that says this signal is not sharp enough to drop rows with. It is
        sharp enough to stop the result being *presented* as an answer.
        """
        from app.config import get_settings  # local: avoids an import cycle

        return (
            self.top_rerank_score is not None
            and self.top_rerank_score < get_settings().min_rerank_score
        )


#: One line for the sources card, which has no room for the full explanation.
#: It lives beside `unreadable_query` rather than in the chat layer because
#: every caller that reads the flag owes the user the same sentence, and two
#: endpoints wording it separately is how they drift apart.
UNREADABLE_QUERY_NOTICE = (
    "โมเดล embedding อ่านคำถามนี้ไม่ออก จึงค้นแบบ semantic ไม่ได้ "
    "(the embedding model cannot read this question)"
)


#: Said in the sources card when nothing retrieved scored as clearly relevant.
#: These chunks are the closest twenty rows in the index, which is not the same
#: claim as "these answer the question" — and five cards with a zeroed
#: relevance bar make the stronger claim unless something says otherwise.
LOW_CONFIDENCE_NOTICE = (
    "ไม่มีเอกสารชิ้นไหนถูกจัดว่าเกี่ยวข้องชัดเจน — ผลด้านล่างคือชิ้นที่ใกล้เคียงที่สุด "
    "เท่านั้น โปรดตรวจสอบก่อนเชื่อ "
    "(nothing scored as clearly relevant; these are only the closest matches)"
)


def notice_for(result: SearchResult) -> str | None:
    """The `notice` field of a `SourcesEvent` built from `result`.

    `/search` documents itself as returning the same payload the chat stream
    sends, so it cannot construct that payload a second time by hand — it did,
    and it left the notice off.
    """
    if result.unreadable_query:
        return UNREADABLE_QUERY_NOTICE
    if result.low_confidence:
        return LOW_CONFIDENCE_NOTICE
    return None


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
    the narrower ones to measure what each contributes.

    Nothing is ever dropped for scoring badly — `SearchResult.low_confidence`
    reports it instead, and `MIN_RERANK_SCORE` carries the measurement that
    says why dropping would cost more than it buys."""
    settings = get_settings()
    top_k = top_k or settings.top_k
    started = time.perf_counter()

    query = query.strip()
    if not query:
        return SearchResult([], 0, 0)

    embedder = get_embedder()

    # Both models here are English-only by default, and both fail *silently* on
    # text they cannot tokenise: the embedder maps every unreadable question to
    # the same vector, and the cross-encoder then scores an arbitrary chunk at
    # 0.99 — measured on this corpus, where two unrelated Thai questions return
    # an identical top five headed by the same irrelevant chunk at 0.9905. A
    # confident wrong answer is the worst failure mode this system has, so when
    # the question is unreadable both blind stages are switched off and the
    # keyword leg carries the query alone.
    readable = embedder.reads(query)

    # Embedded even when unreadable: the vector is still a bound parameter of
    # the one shared query, and a second code path is exactly what the
    # `fuse_candidates` docstring exists to prevent.
    vector = await asyncio.to_thread(embedder.embed_query, query)

    per_leg = settings.candidates_per_leg
    candidates = await fuse_candidates(
        query,
        vector,
        per_leg,
        settings.fusion_keep,
        vector_limit=0 if legs == "keyword" or not readable else per_leg,
        keyword_limit=0 if legs == "vector" else per_leg,
    )
    considered = len(candidates)

    top_score: float | None = None
    if candidates and rerank and readable:
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

        # Kept, not cut. See `MIN_RERANK_SCORE`: the score is a reliable
        # "yes" and an unreliable "no", so it is carried out as a confidence
        # signal for the answer layer and the citation card to act on.
        top_score = candidates[0].rerank_score

    elapsed = int((time.perf_counter() - started) * 1000)
    return SearchResult(
        sources=[to_source(c, i + 1) for i, c in enumerate(candidates[:top_k])],
        candidates_considered=considered,
        retrieval_ms=elapsed,
        unreadable_query=not readable,
        top_rerank_score=top_score,
    )
