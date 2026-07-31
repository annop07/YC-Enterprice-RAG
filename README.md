# Mini Enterprise RAG & Doc Search Agent

> **Week 5 (YC) bootcamp deliverable** — ingest internal documents, ask questions,
> get answers that **cite the exact lines they came from**.

Modelled on [Onyx](https://github.com/onyx-dot-app/onyx): Markdown, PDF and GitHub
sources are chunked and embedded into Postgres, retrieved by **hybrid search**
(vector + full-text, fused and re-ranked), and answered by an LLM over SSE with
clickable citations.

**Status.** Chat UI, SSE contract and citation model — built and running.
Infrastructure — Postgres 16 + pgvector, schema, indexes, FastAPI service — up.
Ingestion and retrieval are next; until they land the UI is served by mock route
handlers that speak the same contract, so the whole interaction is exercised end
to end rather than mocked at the component level.

## What it demonstrates

| Skill | Where it lives |
| --- | --- |
| SSE streaming client — POST + `ReadableStream`, frame-by-frame parse | [`frontend/src/lib/api.ts`](frontend/src/lib/api.ts) (`streamChat`) |
| Streaming state machine — retrieve → stream → done, with cancel | [`frontend/src/hooks/use-chat.ts`](frontend/src/hooks/use-chat.ts) |
| Citation cards showing **why** each chunk was retrieved | [`frontend/src/components/chat/citation-cards.tsx`](frontend/src/components/chat/citation-cards.tsx) |
| Click-through source viewer — real document, cited lines highlighted | [`frontend/src/components/chat/source-panel.tsx`](frontend/src/components/chat/source-panel.tsx) |
| Inline `[n]` citations parsed off a half-written stream | [`frontend/src/components/chat/answer-text.tsx`](frontend/src/components/chat/answer-text.tsx) |
| Session management — list, switch, delete, persist | [`frontend/src/lib/session-store.ts`](frontend/src/lib/session-store.ts) |
| API contract the backend has to satisfy | [`frontend/src/lib/types.ts`](frontend/src/lib/types.ts) |
| Mock backend speaking real SSE at a realistic pace | [`frontend/src/app/api/mock/chat/route.ts`](frontend/src/app/api/mock/chat/route.ts) |
| Postgres schema — pgvector HNSW + `tsvector` GIN on the same row | [`api/app/schema.sql`](api/app/schema.sql) |
| Typed settings, model/dim coupling made explicit | [`api/app/config.py`](api/app/config.py) |
| Schema bootstrap + pooled async access | [`api/app/db.py`](api/app/db.py) |

## Quick start

**UI only** — no backend, no database, no API key:

```bash
npm install --prefix frontend && npm run dev --prefix frontend -- --port 3100
```

Open <http://localhost:3100>. The demo corpus answers four topics (hybrid search,
chunking, deployment, citations) and honestly refuses anything else.

**With the API and database:**

```bash
cp .env.example api/.env   # then paste your key into OPENAI_API_KEY
docker compose up -d       # Postgres 16 + pgvector on host port 5433
uv run --directory api uvicorn app.main:app --port 8100
```

`GET /health` reports the database, the pgvector version and the corpus counts.
The schema is applied on startup — no migration step to remember. To send the UI
at it, set one variable in `frontend/.env.local`:

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

## What is not built yet

- Ingestion: Markdown / PDF / GitHub connectors, chunking, embedding
- RRF fusion query, cross-encoder re-ranking, the grounding guard
- `POST /chat` — the real SSE endpoint the mock currently stands in for
- The evaluation harness behind the Recall@5 numbers quoted in the demo corpus

Until those land, every number the demo answers with comes from
[`frontend/src/lib/mock-corpus.ts`](frontend/src/lib/mock-corpus.ts) and the header
says **demo corpus** on every screen.
