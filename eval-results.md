# Retrieval evaluation

Reproduce with:

```bash
uv run --directory api python -m app.ingest ../docs
curl -X POST localhost:8100/ingest/github \
     -H 'Content-Type: application/json' -d '{"repo":"pgvector/pgvector"}'
uv run --directory api python -m app.eval --misses
```

This file deliberately lives **outside** `docs/`. Everything in `docs/` is
indexed, and a results table inside the corpus would change the corpus it is
measuring.

**The harness scores the database, not this repository.** `app/eval/__main__.py`
reads `SELECT path, text FROM document` and searches whatever is indexed at the
time. Ingest another repository and every number below moves, because the
questions are now being answered against a larger haystack. That is a property
of the measurement, not a bug in it, but it means a table published without its
corpus is not reproducible. The corpus is therefore stated with the table.

## Setup

30 questions, each pinned to the passage that answers it — matching is on the
document path *and* a distinctive phrase from the target chunk, so retrieving
some other part of the right file does not count as a hit.

Documents that answer none of the questions are indexed alongside the project's
own docs as **distractors**. Without them every configuration scores at or near
1.00 on Recall@5 and the comparison says nothing: with a `fusion_keep` of 20
against a 17-chunk corpus, the re-ranker is sorting the entire index.

## Results

Corpus: 15 documents, 147 chunks · embeddings `BAAI/bge-small-en-v1.5` ·
re-ranker `Xenova/ms-marco-MiniLM-L-6-v2`

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| vector only | 0.57 | 0.77 | 0.87 | 0.673 |
| keyword only | 0.50 | 0.77 | 0.87 | 0.640 |
| hybrid (RRF) | 0.73 | 0.90 | 0.90 | 0.806 |
| hybrid + rerank | **0.80** | **0.97** | **0.97** | **0.878** |

Query readability: **10/10** classified correctly.

Recall says nothing about whether the question was read in the first place. Every
one of the 30 questions above is English, so the whole table can sit at 0.97
while a question in another script returns the same arbitrary chunks for any
input, at a re-rank score of 0.99 — which is what it did until the embedding
model's vocabulary coverage was checked before the vector leg ran. The second
number scores a ten-question set that pins that boundary from both sides: six
questions retrieval must decline to read, and four it must not, the last of them
English and unanswerable by this corpus so that *looked and found nothing* is
never allowed to collapse into *could not look*. It is in
[`api/app/eval/readability.json`](api/app/eval/readability.json), a misclassified
question fails the run, and the same set runs in CI as a test because CI does not
run this harness.

22 of the 147 chunks are the project's own documentation and answer the
questions; the other 125 are distractors, drawn from six repositories:
`pgvector/pgvector`, `annop07/doctora-spring-boot`, `annop07/TacticalFitAI`,
`annop07/annop07`, `Erik-Cupsa/GymTech` and `Erik-Cupsa/Erik-Cupsa`.

Reproducing these exact figures means indexing those six. The command above
ingests only `pgvector/pgvector`, which is the smaller corpus described in the
next section — run it and expect the numbers in that table, not this one.

These figures are stable across re-indexing, which earlier ones were not. The
sequence eval → `--force` re-ingest → eval → `--force` re-ingest → eval, which
deletes and re-inserts every chunk row twice over, returns identical values in
every cell. That became true once both retrieval legs got a deterministic
tiebreak: `ts_rank_cd` ties readily, and before the fix a tied group came back in
whatever physical order the last write happened to leave, so a re-index alone
could move a question in or out of the top five. Two narrower sources of
variation remain and are not fixable by a sort key — an exact cosine tie at the
vector leg's candidate limit can still change which rows are fetched, and HNSW is
approximate over a graph that depends on insertion order, so a re-index can in
principle shift the candidate set before any ordering applies.

### The earlier corpus, for comparison

The previously published table measured 7 documents and 88 chunks, 71 of them
distractors, with `pgvector/pgvector` as the only external repository:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | ΔMRR now |
| --- | --- | --- | --- | --- | --- |
| vector only | 0.67 | 0.87 | 0.90 | 0.757 | −0.084 |
| keyword only | 0.67 | 0.87 | 0.87 | 0.756 | −0.116 |
| hybrid (RRF) | 0.73 | 0.93 | 1.00 | 0.843 | −0.037 |
| hybrid + rerank | 0.77 | 0.93 | 0.97 | 0.853 | +0.025 |

Two things make this a loose comparison rather than a controlled one. The older
figures were measured before the ranking tiebreak, so some part of any
difference is the non-determinism described above rather than the corpus. And
`docs/` itself changed in between — `retrieval.md` grew from 4 chunks to 6 when
the section on Thai retrieval was corrected, `streaming.md` from 4 to 5 when the
citation guard was documented, `ingestion.md` from 4 to 5 when the re-indexing
gotcha was added. Read the direction, not the decimals.

The individual legs degraded, which is what a 76% larger distractor pool should
do. The keyword leg lost the most — 0.17 of Recall@1 and 0.116 of MRR — because
the documents added are technical README files whose vocabulary overlaps the
questions almost as well as the right answers do. Lexical matching has no way to
prefer the correct one. The vector leg gave up rather less, and `hybrid + rerank`
did not degrade at all.

**The perfect Recall@5 did not survive.** `hybrid (RRF)` scored 1.00 on the
88-chunk corpus and the README leaned on it. Here it is 0.90: fusion alone no
longer finds every question, and re-ranking is what recovers two of the three it
misses. A figure of 1.00 was always going to be a property of a small corpus
rather than of the method.

## Reading it

**Fusion is where the gain is.** Either leg alone tops out at 0.87 on Recall@5
and 0.57 on Recall@1; fused, that is 0.90 and 0.73, and 0.97 and 0.80 once
re-ranked. The two legs miss *different* questions, which is the entire premise
of running both — "Why not use larger chunks?" is invisible to keyword search,
and "Do e5 models need anything special?" is invisible to the vector leg.

**Re-ranking is the stage carrying the most weight.** It takes fusion's 0.73
Recall@1 to 0.80 and its 0.90 Recall@3 to 0.97, and it recovers two of the three
questions fusion alone does not find. That is the predicted behaviour: the
re-ranker's job starts mattering when fusion is choosing 20 out of a large index
rather than 20 out of a small one.

**One question fails in every configuration**, as it did before. "What is the
trade-off of using pgvector?" is answered in `architecture.md`, but the
distractor corpus *is* pgvector's own documentation, so every leg pulls that
instead. That is the distractors doing their job rather than a bug, and it is a
fair example of what this system gets wrong: a question whose vocabulary belongs
to another document in the index.

## Caveats worth stating

- 30 questions is small. Each question is worth 0.033, so every cell in these
  tables has a granularity of ±1 question and differences under ~0.05 should be
  read as noise. That is arithmetic rather than a measured error bar; the runs
  needed to establish one properly have not been done.
- The questions were written by the same person who wrote the documents, which
  flatters retrieval. A set written by someone who had only read the questions
  would be harder and more honest.
- The distractor corpus is whatever happened to be ingested rather than a set
  chosen to be adversarial. It is realistic, not controlled: five of the six
  repositories are unrelated to this project and one is pgvector's own
  documentation, which is far harder than the rest.
- Everything here measures *retrieval*. Whether the model then answers correctly
  from what it was given is a separate question this harness does not ask.
- Readability is a classification, not a ranking, so it is reported as a count
  rather than as a rate: ten questions is far too few to quote a percentage
  from. It says the boundary is where the set says it should be, not how the
  system behaves on the languages the set does not contain.
