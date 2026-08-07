# Architecture

Three processes and one database. Everything except the LLM call runs locally,
so a full ingest-and-ask loop costs nothing but CPU.

## Components

- **API** — FastAPI. Owns ingestion, retrieval and the chat stream.
- **Postgres 16 + pgvector** — documents, chunks, embeddings, chat history.
- **Web** — Next.js chat UI. Talks to the API over HTTP only.

## Why one database instead of two

A dedicated vector store would mean keeping two systems in sync: a chunk deleted
in Postgres has to disappear from the vector index in the same transaction, or a
stale chunk gets cited and the citation link points at text that is no longer
there.

Putting the embedding column and the tsvector column on the same row removes the
problem entirely. Hybrid search becomes one SQL statement, metadata filters are a
WHERE clause, and deleting a document takes its vectors with it.

The cost is scale. pgvector with an HNSW index is comfortable into the low
millions of chunks; past that a dedicated store earns its complexity back.

## Schema notes

The full-text column uses the `simple` configuration rather than `english`. The
corpus is mixed-language and the English stemmer mangles everything that is not
English. Thai needs word segmentation before that column is useful at all, which
is not implemented yet.

Citations are stored twice on purpose. `message_citation` keeps a soft reference
to the chunk and a full snapshot of the source payload that was sent to the
client. Re-ingesting a document replaces its chunks, and an old answer must not
lose its citations because the text behind them was re-indexed.

The vector column width is substituted into the schema at bootstrap from
`EMBED_DIM`, because pgvector needs a fixed width to build an HNSW index. That
also means Postgres rejects a mismatched model rather than silently corrupting
the index.

## Request path

A question goes: embed, then vector leg and keyword leg in parallel, then RRF
fusion, then cross-encoder re-ranking, then the top few chunks into the prompt,
then tokens streamed back over SSE. Sources are flushed to the client before the
first token so the citation cards render while the answer is still being written.

Every step of that path is implemented. The chat endpoint writes the turn as it
streams, so a conversation is listed and replayable the moment it finishes.
