# Hybrid Search and Re-ranking

Retrieval is two independent legs fused into one ranking, then trimmed by a
cross-encoder before anything reaches the LLM. Implemented in
`api/app/retrieval/`.

## The two legs

The vector leg is a pgvector cosine search over the embedding column, backed by
an HNSW index. It finds paraphrases: a question phrased "how do I start
everything on my laptop" reaches a page titled "Running services locally"
without sharing a single content word with it.

The keyword leg is Postgres full-text search over a tsvector column with a GIN
index. It finds exact tokens — error codes, flag names, a header spelled one
particular way. Those are the queries embeddings are weakest at.

`hnsw.ef_search` is set to 100 per connection. The default trades away more
recall than this workload can afford.

## Why the keyword query is rewritten

`websearch_to_tsquery` and `plainto_tsquery` both AND their terms. A question
like "how do I run the whole stack locally?" therefore has to appear in a chunk
in full to match anything, which never happens — measured on this corpus, the
AND form returned zero rows for that question. The keyword leg was silently
contributing nothing to every natural-language query while still looking like
half of a hybrid search.

The query is rewritten into an OR over its content words instead. Stopwords are
removed first, because `ts_rank_cd` has no inverse document frequency: with "the"
and "do" left in, they score exactly as loudly as "compose", and the correct
chunk fell out of the top five entirely. With them removed it ranks first.

Every term is quoted before it reaches `to_tsquery`, so punctuation in a query
is data rather than operators.

## What Thai does to each leg

Thai defeats both legs, and it defeats them for two unrelated reasons that need
two unrelated fixes.

The keyword leg fails at the tokenizer, where the `simple` configuration
does not segment Thai, so an unbroken run of Thai characters becomes one token
covering the whole phrase. Postgres will match that token against a query token
spelled exactly the same way and against nothing else — no prefix of it, no word
inside it. Measured over six Thai questions, not one of the twenty chunks that
came back had a keyword rank at all.

The vector leg fails earlier and more completely. `BAAI/bge-small-en-v1.5` is an
English model and there is no Thai in its WordPiece vocabulary, so each unbroken
run of Thai encodes to a single `[UNK]`. Six Thai questions on six different
subjects therefore produced the same embedding — pairwise cosine 1.0000 — and
the leg returned the same twenty chunks for all of them. It is not ranking the
corpus badly. It is not reading the question.

Mixing Latin-script tokens into a Thai question changes the picture, and it is
the keyword leg that recovers. `simple` tokenizes `pgvector` or `hnsw.ef_search`
normally whatever surrounds them, while the vector leg still sees those tokens
against a background of `[UNK]`. Across six such questions the keyword leg
ranked the answering chunk above the vector leg four times and level with it
twice:

| question | vector rank | keyword rank |
| --- | --- | --- |
| `pgvector มีข้อเสียอะไรบ้าง` | 49 | 8 |
| `search รองรับภาษาไทยไหม` | 40 | 10 |
| `ทำไมต้องลบ stopwords ก่อนเรียก ts_rank_cd` | 9 | 1 |
| `hnsw.ef_search ตั้งค่าไว้เท่าไร` | 6 | 3 |

So the keyword leg is the one carrying mixed queries, which is the reverse of
what a hybrid search is usually assumed to do with a non-English question.

The two repairs are independent. The keyword leg needs a Thai segmenter feeding
`to_tsvector`. The vector leg needs a multilingual embedding model —
`api/app/config.py` points at `intfloat/multilingual-e5-large` — and swapping it
re-embeds the corpus, so it is a re-index rather than a configuration change.
Doing either one alone leaves the other leg exactly as blind as it is now.

## Fusion

The two rankings are combined with Reciprocal Rank Fusion: the sum of
1 / (60 + rank) over the legs a chunk appears in. RRF reads positions only, so
cosine distance and `ts_rank_cd` never have to be normalised onto a shared
scale — which is exactly where a hand-weighted hybrid score usually goes wrong.

Both legs are ordered and limited inside a subquery so the HNSW and GIN indexes
can serve the ordering, and ranked in the outer query. Ranking before the limit
would force a full sort and discard both indexes.

Each leg returns 50 candidates and fusion keeps 20, in a single SQL statement.

## Re-ranking

Those 20 go through a cross-encoder that reads the question and the chunk
together and scores how well one answers the other, rather than comparing two
embeddings that were computed independently. It is slower per pair but much
closer to relevance, and only the top 5 survive into the prompt.

The model emits unbounded logits — around -11 for an unrelated pair. What the
API returns is the sigmoid of that, which is the model's relevance probability
and the only form a progress bar can be drawn from.

## What the citation carries

Every source reports the rank it held in each leg. A chunk with a vector rank
and no keyword rank is one a keyword-only search would have missed, and the
reverse for the other leg. That is the evidence that the search is hybrid,
rather than the claim.
