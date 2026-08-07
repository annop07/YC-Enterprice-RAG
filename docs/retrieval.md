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

This is also the leg that Thai text does not reach yet. The `simple`
configuration does not segment Thai, so a Thai sentence becomes one unusable
token. The vector leg carries those queries alone until a segmenter is added.

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
