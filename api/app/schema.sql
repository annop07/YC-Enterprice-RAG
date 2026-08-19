-- Schema for the enterprise RAG store.
--
-- Applied on startup by `db.bootstrap()`, which substitutes __EMBED_DIM__ from
-- settings — pgvector needs a fixed width to build an HNSW index, so the
-- column cannot be declared until the embedding model is known. That is also
-- why changing EMBED_MODEL to one with a different width is a migration and
-- not a config edit: Postgres rejects the mismatch instead of silently
-- corrupting the index.
--
-- Because of that token this file is not runnable through psql as-is.

CREATE EXTENSION IF NOT EXISTS vector;

-- --------------------------------------------------------------------------
-- Corpus
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document (
    id           TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL CHECK (source_type IN ('markdown', 'pdf', 'github')),
    -- Stable identity within a source: a path, or "owner/repo@path". Re-running
    -- a connector must land on the same row rather than insert a duplicate.
    source_id    TEXT NOT NULL,
    title        TEXT NOT NULL,
    path         TEXT NOT NULL,
    url          TEXT,
    -- Hash of the extracted text. Equal hash on re-ingest means skip entirely.
    content_hash TEXT NOT NULL,
    -- The full extracted text is kept so the source viewer can show the cited
    -- lines in context. Chunks alone cannot reconstruct the gaps between them.
    text         TEXT NOT NULL,
    meta         JSONB NOT NULL DEFAULT '{}'::jsonb,
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_type, source_id)
);

CREATE TABLE IF NOT EXISTS chunk (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    -- What the answer quotes and the citation card shows.
    content      TEXT NOT NULL,
    -- What was actually embedded: content prefixed with title + heading path.
    -- Stored so a re-embed is reproducible without re-deriving the prefix.
    embed_input  TEXT NOT NULL,
    heading_path TEXT,
    -- Markdown and GitHub chunks carry a line range; PDF chunks carry a page.
    line_start   INT,
    line_end     INT,
    page         INT,
    token_count  INT NOT NULL,
    embedding    VECTOR(__EMBED_DIM__) NOT NULL,
    -- 'simple' rather than 'english': the corpus is mixed-language and the
    -- English stemmer mangles everything that is not English. Thai needs
    -- word segmentation before this column is useful at all.
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    -- And 'english' as well as, never instead of. `simple` does not stem, so
    -- "chunk" does not match "chunks" and "deploy" does not match "deployment"
    -- — measured: the tsvector for "The chunks are 400 tokens" does not match
    -- the query 'chunk'. Searching both columns and taking the better score
    -- recovers English word forms without the stemmer being allowed anywhere
    -- near the faithful copy the other languages depend on.
    tsv_en       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, ordinal)
);

-- For databases created before `tsv_en` existed. A generated column is filled
-- in for every existing row by the ALTER itself, so this is not a re-index.
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS tsv_en TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunk_document_idx ON chunk (document_id);
CREATE INDEX IF NOT EXISTS chunk_tsv_idx ON chunk USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunk_tsv_en_idx ON chunk USING gin (tsv_en);
-- Cosine, to match the normalised embeddings fastembed returns.
CREATE INDEX IF NOT EXISTS chunk_embedding_idx
    ON chunk USING hnsw (embedding vector_cosine_ops);

-- --------------------------------------------------------------------------
-- Chat
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chat_session (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_message (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_session(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    -- latency, usage, model, dropped_citations — whatever the `done` event sent.
    meta       JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Insertion order. `created_at` cannot order a conversation: the question
    -- and its answer are written in one transaction, and now() is transaction
    -- time, so both rows carry the same timestamp and the transcript comes back
    -- with the answer above the question.
    seq        BIGSERIAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- For databases created before `seq` existed.
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS seq BIGSERIAL;

CREATE INDEX IF NOT EXISTS chat_message_session_idx
    ON chat_message (session_id, seq);

CREATE TABLE IF NOT EXISTS message_citation (
    message_id TEXT NOT NULL REFERENCES chat_message(id) ON DELETE CASCADE,
    n          INT  NOT NULL,
    -- Soft reference: re-ingesting a document deletes and re-creates its
    -- chunks, and old answers must not lose their citations because of it.
    chunk_id   BIGINT REFERENCES chunk(id) ON DELETE SET NULL,
    -- Full Source payload as it was sent to the client, so history renders
    -- identically even after the chunk it pointed at is gone.
    source     JSONB NOT NULL,
    PRIMARY KEY (message_id, n)
);

-- --------------------------------------------------------------------------
-- Ingestion jobs
-- --------------------------------------------------------------------------

-- Indexing is slow — a repository is one HTTP round trip per file, and every
-- chunk is embedded on the CPU — and it used to run inside the request that
-- asked for it. That request had no way to say how far it had got, so the UI
-- showed one spinner for the whole thing; a proxy or a browser could time it
-- out and leave the work half done with nobody holding the result; and closing
-- the tab killed it. The work is now a row here, and the request that starts
-- it only has to hand back an id.
CREATE TABLE IF NOT EXISTS ingest_job (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL CHECK (source IN ('files', 'github')),
    -- What is being indexed, for a list that has to be readable: "4 files" or
    -- "pgvector/pgvector".
    label       TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('running', 'done', 'failed')),
    -- The phase a running job is in. A GitHub job spends most of its time
    -- fetching, before there is anything to count.
    phase       TEXT,
    -- Null while unknown: a connector discovers its documents as it goes.
    total       INT,
    done        INT NOT NULL DEFAULT 0,
    -- The document being worked on right now.
    current     TEXT,
    -- The finished IngestResponse, so a completed job renders the same report
    -- the synchronous endpoint used to return.
    report      JSONB,
    -- Set only when the job as a whole failed. A single document that could
    -- not be read is not this: it is one "failed" row inside `report`, and the
    -- other documents in the same job still get indexed.
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ingest_job_created_idx ON ingest_job (created_at DESC);
