# Mini Enterprise RAG & Doc Search Agent

> **Week 5 (YC) bootcamp deliverable** — ingest internal documents, ask questions,
> get answers that **cite the exact lines they came from**.

Modelled on [Onyx](https://github.com/onyx-dot-app/onyx): Markdown, PDF and GitHub
sources are chunked and embedded into Postgres, retrieved by **hybrid search**
(vector + full-text, fused and re-ranked), and answered by an LLM over SSE with
clickable citations.

**Status — end to end and measured.** Ask a question in the browser and it is
embedded, searched across both legs, fused, re-ranked, answered by an LLM over
the retrieved text, streamed back token by token, and stored with its citations.
The mock route handlers are still there and still speak the same contract:
comment out one environment variable and the UI runs with no backend at all.

## Does the retrieval actually work?

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| vector only | 0.57 | 0.77 | 0.87 | 0.673 |
| keyword only | 0.50 | 0.77 | 0.87 | 0.640 |
| hybrid (RRF) | 0.73 | 0.90 | 0.90 | 0.806 |
| hybrid + rerank | **0.80** | **0.97** | **0.97** | **0.878** |

30 questions over 147 chunks, of which 125 are distractors that answer none of
them. `uv run --directory api python -m app.eval` reproduces this, but only
against the same corpus — the harness scores whatever is in the database, so the
numbers move when the index does. [`eval-results.md`](eval-results.md) lists the
exact documents, the earlier and smaller distractor set these numbers replace,
and the questions nothing answers.

**Fusion is where the gain is** — the two legs miss *different* questions, which
is the whole premise of running both. Alone, neither leg gets past 0.87 at
Recall@5 or 0.57 at Recall@1; fused those are 0.90 and 0.73, and re-ranking
takes Recall@1 to 0.80. **The pipeline as a whole is what holds up:** against a
distractor set two thirds larger than the one first measured, the single legs
fell away — the keyword leg from 0.67 to 0.50 Recall@1 — while `hybrid + rerank`
did not degrade at all. The figures are stable across re-indexing, which earlier
ones were not. Individual cells are worth ±1 question at this sample size and
should be read that way: an earlier corpus put fusion at a perfect 1.00 on
Recall@5, and that did not survive a larger one.

## What it demonstrates

| Skill | Where it lives |
| --- | --- |
| SSE streaming client — POST + `ReadableStream`, frame-by-frame parse | [`frontend/src/lib/api.ts`](frontend/src/lib/api.ts) (`streamChat`) |
| Streaming state machine — retrieve → stream → done, with cancel | [`frontend/src/hooks/use-chat.ts`](frontend/src/hooks/use-chat.ts) |
| Citation cards showing **why** each chunk was retrieved | [`frontend/src/components/chat/citation-cards.tsx`](frontend/src/components/chat/citation-cards.tsx) |
| Click-through source viewer — real document, cited lines highlighted | [`frontend/src/components/chat/source-panel.tsx`](frontend/src/components/chat/source-panel.tsx) |
| Inline `[n]` citations parsed off a half-written stream | [`frontend/src/components/chat/answer-text.tsx`](frontend/src/components/chat/answer-text.tsx) |
| Session management — list, switch, delete, persist | [`frontend/src/lib/api.ts`](frontend/src/lib/api.ts), [`frontend/src/components/chat/session-sidebar.tsx`](frontend/src/components/chat/session-sidebar.tsx) |
| API contract the backend has to satisfy | [`frontend/src/lib/types.ts`](frontend/src/lib/types.ts) |
| Mock backend speaking real SSE at a realistic pace | [`frontend/src/app/api/mock/chat/route.ts`](frontend/src/app/api/mock/chat/route.ts) |
| Postgres schema — pgvector HNSW + `tsvector` GIN on the same row | [`api/app/schema.sql`](api/app/schema.sql) |
| Typed settings, model/dim coupling made explicit | [`api/app/config.py`](api/app/config.py) |
| Schema bootstrap + pooled async access | [`api/app/db.py`](api/app/db.py) |
| Connector protocol — a new source is one class, not a new pipeline | [`api/app/ingest/connectors.py`](api/app/ingest/connectors.py) |
| Structure-aware chunker with exact line ranges and real token counts | [`api/app/ingest/chunker.py`](api/app/ingest/chunker.py) |
| Embedder that counts with the same tokenizer it embeds with | [`api/app/ingest/embedder.py`](api/app/ingest/embedder.py) |
| Idempotent ingest — content hash, delete-and-insert in one transaction | [`api/app/ingest/pipeline.py`](api/app/ingest/pipeline.py) |
| PDF extraction with page locators and a scanned-document warning | [`api/app/ingest/pdf.py`](api/app/ingest/pdf.py) |
| GitHub via one recursive tree call, permalinks pinned to the commit | [`api/app/ingest/github.py`](api/app/ingest/github.py) |
| Upload and repository ingest endpoints | [`api/app/main.py`](api/app/main.py), `POST /ingest/files` · `POST /ingest/github` |
| Corpus panel — drop a file in, point at a repo, see and remove what is indexed | [`frontend/src/components/corpus/corpus-panel.tsx`](frontend/src/components/corpus/corpus-panel.tsx) |
| Hybrid search — both legs and RRF fusion in one SQL statement | [`api/app/retrieval/search.py`](api/app/retrieval/search.py), `POST /search` |
| Cross-encoder re-ranking, logits squashed to a relevance probability | [`api/app/retrieval/reranker.py`](api/app/retrieval/reranker.py) |
| Integration tests against a real Postgres, on their own database | [`api/tests/test_search_integration.py`](api/tests/test_search_integration.py) |
| Streaming chat over retrieval, with the SSE order the UI depends on | [`api/app/chat/service.py`](api/app/chat/service.py), `POST /chat` |
| Query rewriting so follow-up questions retrieve anything at all | [`rewrite_question`](api/app/chat/service.py) |
| Citation guard — invented citations removed and counted | [`api/app/chat/prompt.py`](api/app/chat/prompt.py) |
| Chat history that survives a re-index of the documents it cites | [`api/app/chat/store.py`](api/app/chat/store.py), `GET /sessions` |
| Evaluation harness — four configurations through one code path | [`api/app/eval/runner.py`](api/app/eval/runner.py), [`eval-results.md`](eval-results.md) |
| Containerised stack and CI that runs the SQL tests against real Postgres | [`docker-compose.yml`](docker-compose.yml), [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

## Quick start

**UI only** — no backend, no database, no API key:

```bash
npm install --prefix frontend && npm run dev --prefix frontend -- --port 3100
```

Open <http://localhost:3100>. The demo corpus answers four topics (hybrid search,
chunking, deployment, citations) and honestly refuses anything else.

**The whole stack in containers:**

```bash
cp .env.example api/.env   # then paste your key into OPENAI_API_KEY
docker compose --profile full up -d
uv run --directory api python -m app.ingest ../docs
```

Open <http://localhost:3100>.

**Or run the app processes directly** and keep only the database in Docker —
which is what the `full` profile exists to stay out of the way of:

```bash
docker compose up -d       # Postgres 16 + pgvector on host port 5433
uv run --directory api uvicorn app.main:app --port 8100 --reload
npm run dev --prefix frontend -- --port 3100
```

`GET /health` reports the database, the pgvector version and the corpus counts.
The schema is applied on startup — no migration step to remember.

**Index some documents:**

```bash
uv run --directory api python -m app.ingest ../docs
```

A first run marks every document `+` for created. The same command with
`--force` re-indexes a corpus that is already there, and prints `~` instead:

```
  ~ architecture.md                                  3 chunks
  ~ ingestion.md                                     5 chunks
  ~ retrieval.md                                     6 chunks
  ~ streaming.md                                     5 chunks
  ~ handbook.pdf                                     3 chunks
5 documents (5 written, 0 unchanged) · 22 chunks · budget 400 tokens
```

Re-running is free — an unchanged file is skipped without re-embedding. The
check is a hash of the file's text alone, so it cannot see a change in the
*extraction* code: improve how titles are parsed and a normal re-run reports
`unchanged` and keeps the old title, which is embedded into every chunk of that
document and not merely displayed. Pass `--force` after changing anything in the
extraction path. `GET /documents` lists what is indexed, and
[`docs/ingestion.md`](docs/ingestion.md) explains the trap.

Or from the browser: the corpus counts in the sidebar open a panel that takes
dropped Markdown and PDFs, indexes a GitHub repository, and lists what is
already in the index. Removing a document there leaves the answers that cited it
intact — `message_citation` keeps a snapshot of the source, so old citations
still render after the text behind them is gone.

The same thing over HTTP:

```bash
curl -X POST localhost:8100/ingest/files -F "files=@notes.md" -F "files=@handbook.pdf"
curl -X POST localhost:8100/ingest/github \
     -H 'Content-Type: application/json' \
     -d '{"repo":"pgvector/pgvector","path_prefix":"docs"}'
```

Set `GITHUB_TOKEN` for private repositories, and for the rate limit: GitHub
allows 60 requests an hour unauthenticated, which one medium repository spends.

**Search it:**

```bash
curl -X POST localhost:8100/search \
     -H 'Content-Type: application/json' \
     -d '{"query":"how do I run the whole stack locally?","top_k":3}'
```

```
[1] handbook.pdf p.2      vector #1  keyword #1   rerank 0.04
[2] architecture.md L1-24 vector #2  keyword #4   rerank 0.001
```

`POST /search` returns exactly the payload the chat stream sends as its
`sources` event, so retrieval can be inspected and measured without generating
an answer over it.

**Ask it:**

```bash
curl -N -X POST localhost:8100/chat \
     -H 'Content-Type: application/json' \
     -d '{"message":"What chunk size does ingestion use, and why?"}'
```

```
event: session   {"session_id":"s_…","title":"What chunk size does ingestion use…"}
event: sources   {"sources":[…],"candidates_considered":17,"retrieval_ms":1007}
event: token     {"text":"The ingestion pipeline uses "}
…
event: done      {"latency_ms":2543,"usage":{…},"dropped_citations":0,"model":"…"}
```

The turn is stored as it streams, so `GET /sessions` lists it and
`GET /sessions/{id}` replays it with its citations intact.

To send the UI at the API, set one variable in `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8100
```

> `next dev` logs one React warning about a script tag inside a component. That is
> the theme script in [`layout.tsx`](frontend/src/app/layout.tsx), which has to run
> synchronously in `<head>` to avoid a flash of the wrong background — the warning
> lives only in React's development bundle and is absent from `next build`.

## The SSE contract

Events arrive in this order, and the order is the point:

```
event: session   {"session_id":"s_…","title":"How does hybrid search work?"}
event: sources   {"sources":[…],"candidates_considered":20,"retrieval_ms":420}
event: token     {"text":"Retrieval "}
event: token     {"text":"runs "}          …
event: done      {"latency_ms":6210,"usage":{…},"dropped_citations":0,"model":"…"}
```

`sources` is flushed **before the first token** so the citation cards are on screen
and readable while the answer is still being written, instead of appearing after it.

`EventSource` is not used: it is GET-only and cannot carry a request body, so the
stream is read off a `fetch` POST and the SSE framing is parsed by hand
([`api.ts`](frontend/src/lib/api.ts)). The response also carries
`X-Accel-Buffering: no` — without it a proxy buffers the whole body and the answer
lands in one lump.

## Three decisions worth knowing about

**1. A citation names lines, not files.** Every chunk carries `line_start`/`line_end`
(or a `page`, for PDFs) written at split time and never recomputed. Clicking `[2]`
opens the real document with those lines highlighted, so a claim can be checked in
one click rather than taken on trust. GitHub links are built from the commit SHA,
never the branch, so they keep pointing at the text that was actually indexed.

**2. The citation card shows its own retrieval trace.** Each card prints
`vector #1 · keyword #2` and the re-rank score. A chunk with only one of the two
ranks is one that a single-leg retriever would have missed — which is the evidence
for calling the search hybrid, rather than the claim.

**3. Dropped citations are printed, not swallowed.** The backend checks every `[n]`
the model emits against the chunks it was actually shown and strips the ones that
match nothing. The count appears under every answer, including when it is zero —
a hallucination rate that only shows up when it is non-zero is one nobody looks at.
The UI greys out any marker with no matching source rather than rendering a dead link.

## Design

The UI reuses the design system from [`../AI-Engineer/frontend`](../AI-Engineer/frontend):
Next.js 16 + React 19, Tailwind v4, shadcn `base-nova` on Base UI, neutral OKLCH
tokens, Geist Sans/Mono, Tabler icons. The composer is the same frame as that
project's prompt box — bordered card, focus ring on the whole card, controls inside it.

## Two measurements that changed the design

**The default chat model could not be the workspace default.** On the KKU proxy,
`qwen3.7-max` turned out to be a reasoning model: it streams fine, but it spends
~3.5k characters of `reasoning` before the first *visible* token, so
time-to-first-token is **18 seconds** and a short answer costs ~940 completion
tokens. `gemini-3.6-flash` returns the whole answer in a single content delta —
it streams on paper and looks buffered on screen. The default here is
`qwen3-next-80b-a3b-instruct`: **0.9s** to first token, 30 deltas, no reasoning
preamble. `claude-haiku-4.5` (1.7s, 60 deltas) is the documented alternative.
This is exactly what the spike was for — finding it on day 5 would have meant
rebuilding the chat endpoint around a "thinking…" state.

**Chunks are 400 tokens, not the 500–1000 in the brief.** Every multilingual
embedding model fastembed offers — `intfloat/multilingual-e5-large`, the
paraphrase-multilingual pair — caps at **512 input tokens** and truncates past
that *silently*. A 1000-token chunk would have half its text embedded and no
error anywhere. 400 leaves room for the title and heading prefix that gets
prepended to the embedding input.

## A third measurement, found by ingesting a real repository

The first GitHub ingest turned pgvector's README into **901 chunks**. A 14k-token
document should produce about 40.

fastembed configures its tokenizer for inference: truncation on, and **padding
on**. Padding makes `encode_batch` pad every sequence to the longest one in the
batch, so counting a document line by line returned the same number for a
one-word line and a full paragraph — 165 tokens for all 1,362 lines, and zero
blank lines in a file that has 461. Nothing raised. The chunker simply packed by
a constant, and every `token_count` in the database was fiction.

The counting tokenizer now has both padding and truncation switched off, which
is what [`test_batch_counting_is_not_padded_to_the_longest_item`](api/tests/test_embedder.py)
pins. The same README now yields 58 chunks against an ideal of 35.

The lesson generalises: an inference-tuned tokenizer is not a measuring
instrument, and every wrong answer it gives is plausible.

## A fourth: the keyword leg was doing nothing

`websearch_to_tsquery` and `plainto_tsquery` both **AND** their terms. So a
question like "how do I run the whole stack locally?" only matches a chunk
containing every one of those words — which no chunk does. Measured: **zero
rows**. The keyword leg contributed nothing to any natural-language question
while still appearing, correctly wired and fully tested, as half of a hybrid
search.

Switching to OR over the query's terms matched everything, and ranked it badly:
`ts_rank_cd` has no inverse document frequency, so "the" and "do" score as
loudly as "compose". The right chunk sat outside the top five. Dropping
stopwords first put it at number one.

Both halves of that were invisible from the code. They only showed up by
running real questions against a real corpus and reading the ranks — which is
why every source in the API response carries the rank it held in each leg.

## A fifth: `now()` cannot order a conversation

The transcript came back with the answer above the question. Both rows are
written in one transaction and `created_at` defaults to `now()`, which in
Postgres is **transaction** time — identical for both. The tiebreaker was the
message id, which is random. A `seq` column fixed it, and
[`test_a_transcript_comes_back_in_the_order_it_was_written`](api/tests/test_chat_store_integration.py)
keeps it fixed.

The same tests turned up a second one: a citation's foreign key to its chunk
made a re-index landing mid-answer take the whole turn down. The link is now
resolved through a subquery that yields NULL when the chunk is gone — it was
only ever a convenience, since the stored source snapshot is what history
renders from.

## Known limits

- **Thai reaches neither leg, for two separate reasons.** `to_tsvector('simple',
  …)` does not segment Thai, so a Thai sentence becomes one unusable token and
  the keyword leg contributes nothing. The vector leg is in worse shape, not
  better: `BAAI/bge-small-en-v1.5` has no Thai in its vocabulary, so a run of
  Thai encodes to a single `[UNK]` and unrelated Thai questions come back with
  the same twenty chunks. Questions that mix Thai with Latin-script terms do
  retrieve, and there the keyword leg is usually the stronger of the two,
  because `simple` tokenizes those terms normally. Fixing this is a segmenter
  for one leg and a multilingual embedding model plus a re-index for the other.
  Measurements are in [`docs/retrieval.md`](docs/retrieval.md).
- **The evaluation set is small and self-authored.** 30 questions written by the
  person who wrote the documents flatters retrieval; differences under ~0.05 are
  noise at that size.
- **Answer quality is not measured.** The harness scores retrieval. Whether the
  model then answers correctly from what it was handed is a separate question.
- **HNSW is built at bootstrap.** Fine for a corpus of this size; a bulk load of
  millions of chunks wants the index created afterwards.

Leaving `NEXT_PUBLIC_API_BASE` unset is a separate mode, not a limitation: the UI
falls back to [`frontend/src/lib/mock-corpus.ts`](frontend/src/lib/mock-corpus.ts),
labels every screen **demo corpus**, and runs with no backend at all.
