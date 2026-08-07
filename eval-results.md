# Retrieval evaluation

Reproduce with:

```bash
uv run --directory api python -m app.ingest ../docs
curl -X POST localhost:8100/ingest/github \
     -H 'Content-Type: application/json' -d '{"repo":"pgvector/pgvector"}'
uv run --directory api python -m app.eval --misses --out ../eval-results.md
```

This file deliberately lives **outside** `docs/`. Everything in `docs/` is
indexed, and a results table inside the corpus would change the corpus it is
measuring.

## Setup

30 questions, each pinned to the passage that answers it — matching is on the
document path *and* a distinctive phrase from the target chunk, so retrieving
some other part of the right file does not count as a hit.

pgvector's README and CHANGELOG are indexed alongside the project's own docs as
**distractors**: 71 of the corpus's 88 chunks answer none of the questions. Run
without them, every configuration scores at or near 1.00 on Recall@5 and the
comparison says nothing — with a `fusion_keep` of 20 against a 17-chunk corpus,
the re-ranker is sorting the entire index.

Corpus: 7 documents, 88 chunks · embeddings `BAAI/bge-small-en-v1.5` ·
re-ranker `Xenova/ms-marco-MiniLM-L-6-v2`

## Results

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| vector only | 0.67 | 0.87 | 0.90 | 0.757 |
| keyword only | 0.67 | 0.87 | 0.87 | 0.756 |
| hybrid (RRF) | 0.73 | 0.93 | **1.00** | 0.843 |
| hybrid + rerank | **0.77** | 0.93 | 0.97 | **0.853** |

## Reading it

**Fusion is where the gain is.** Either leg alone finds the right passage in the
top five about 88% of the time; fused, it is 100%. The two legs miss *different*
questions, which is the entire premise of running both — "Why not use larger
chunks?" is invisible to keyword search, and "Do e5 models need anything
special?" is invisible to the vector leg.

**Re-ranking trades recall for precision.** It moves the right passage to first
place more often (0.73 → 0.77) and lifts MRR, but it drops one question out of
the top five that fusion had found. On a corpus this size that is a single
question, so the honest reading is that the direction is right and the magnitude
is not yet measurable. Re-ranking is kept because it is the part that scales:
its job starts mattering when fusion is choosing 20 out of tens of thousands
rather than 20 out of 88.

**One question fails in every configuration.** "What is the trade-off of using
pgvector?" is answered in `architecture.md`, but the distractor corpus *is*
pgvector's own documentation, so every leg pulls that instead. That is the
distractors doing their job rather than a bug, and it is a fair example of what
this system gets wrong: a question whose vocabulary belongs to another document
in the index.

## Caveats worth stating

- 30 questions is small. Differences under ~0.05 are noise at this size.
- The questions were written by the same person who wrote the documents, which
  flatters retrieval. A set written by someone who had only read the questions
  would be harder and more honest.
- Everything here measures *retrieval*. Whether the model then answers correctly
  from what it was given is a separate question this harness does not ask.
