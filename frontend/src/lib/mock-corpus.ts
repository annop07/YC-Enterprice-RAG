/**
 * Demo corpus for the mock backend.
 *
 * Server-only: imported by the route handlers under `app/api/mock/`, never by
 * a component. It exists so the chat UI streams real SSE frames — with real
 * line numbers pointing at real text — before the FastAPI service is written.
 * Delete this file and point NEXT_PUBLIC_API_BASE at the API to go live.
 */
import type { Source, SourceType } from "./types";

interface MockDoc {
  id: string;
  title: string;
  path: string;
  source_type: SourceType;
  url: string | null;
  indexed_at: string;
  text: string;
}

interface MockChunk {
  id: string;
  doc_id: string;
  heading_path: string;
  /** Verbatim slice of the document — line numbers are derived from it */
  snippet: string;
  /** PDFs are located by page; Markdown and GitHub docs by line range */
  page?: number;
}

interface MockAnswer {
  /** Any one of these in the question routes to this answer */
  keywords: string[];
  answer: string;
  cites: {
    chunk_id: string;
    vector_rank: number | null;
    keyword_rank: number | null;
    rrf_score: number;
    rerank_score: number;
  }[];
}

// --- Documents ------------------------------------------------------------

export const DOCS: MockDoc[] = [
  {
    id: "d_arch",
    title: "Architecture Overview",
    path: "docs/architecture.md",
    source_type: "markdown",
    url: null,
    indexed_at: "2026-09-08T09:12:00Z",
    text: `# Architecture Overview

The service is three processes and one database. Everything that is not the
LLM call runs locally, so a full ingest-and-ask loop costs nothing but CPU.

## Components

- **API** — FastAPI. Owns ingestion, retrieval and the chat stream.
- **Postgres 16 + pgvector** — documents, chunks, embeddings, chat history.
- **Web** — Next.js chat UI. Talks to the API over HTTP only.

## Why one database instead of two

A dedicated vector store would mean keeping two systems in sync: a chunk
deleted in Postgres has to disappear from the vector index in the same
transaction, or a stale chunk gets cited and the citation link 404s.

Putting the embedding column and the tsvector column on the same row removes
the problem entirely. Hybrid search becomes one SQL statement, metadata
filters are a WHERE clause, and a deleted document takes its vectors with it.

The cost is scale: pgvector with HNSW is comfortable into the low millions of
chunks. Past that a dedicated store earns its complexity back.

## Request path

A question goes: embed → vector leg + keyword leg in parallel → RRF fusion →
cross-encoder re-rank → top 5 into the prompt → stream tokens back over SSE.
Sources are flushed to the client before the first token so the citation cards
render while the answer is still being written.
`,
  },
  {
    id: "d_ingest",
    title: "Ingestion Pipeline",
    path: "docs/ingestion.md",
    source_type: "markdown",
    url: null,
    indexed_at: "2026-09-08T09:12:00Z",
    text: `# Ingestion Pipeline

Every source is a Connector: an object that yields RawDocument. Adding
Confluence or Google Drive is one new file, not a new pipeline.

## Connectors

- **FileConnector** — uploaded .md and .pdf
- **GitHubConnector** — one call to the trees API, then the .md files it lists
- **DirConnector** — a local folder, used by the test fixtures

GitHub permalinks are built from the commit SHA, never from the branch name.
A link built against main rots the moment someone edits the file; a link
against the SHA points at the exact text that was indexed.

## Chunking

Chunks are 400 tokens with 80 tokens of overlap. The splitter is
structure-aware: it breaks on Markdown headings first and only falls back to
a hard token split when a single section is too long to fit.

400 rather than the 500-1000 originally planned: every multilingual model
fastembed offers caps at 512 input tokens and truncates past that silently,
and the title and heading prefix has to fit inside that budget too.

Token counts come from the embedding model's own tokenizer. Counting
characters and dividing by four is off by 30% on Thai text, which silently
truncates the tail of every chunk that contains any.

Each chunk carries its heading breadcrumb, its line range in the source file,
and its token count. The line range is what makes a citation clickable, so it
is written at split time and never recomputed.

## Embedding

The text that gets embedded is not the text that gets displayed. Embedding
input is prefixed with the document title and heading path — a cheap version
of contextual retrieval that costs no extra LLM call and measurably lifts
recall on short chunks that would otherwise lose their context.

## Re-indexing

Each document carries a content hash. Re-ingesting an unchanged file is a
no-op. A changed file deletes all of its chunks and re-inserts them in one
transaction, so a partially re-indexed document is never visible to search.
`,
  },
  {
    id: "d_retrieval",
    title: "Hybrid Search & Re-ranking",
    path: "docs/retrieval.md",
    source_type: "markdown",
    url: null,
    indexed_at: "2026-09-08T09:12:00Z",
    text: `# Hybrid Search & Re-ranking

Retrieval is two independent legs fused into one ranking, then trimmed by a
cross-encoder before anything reaches the LLM.

## The two legs

The vector leg is a pgvector cosine search over the embedding column, backed
by an HNSW index. It finds paraphrases: a question about "how do I ship this"
still matches a section titled "Deployment".

The keyword leg is Postgres full-text search over a tsvector column with a GIN
index. It finds exact tokens: error codes, flag names, a config key spelled
one specific way. Those are the queries where embeddings are weakest.

## Fusion

The two rankings are combined with Reciprocal Rank Fusion, score = sum of
1 / (60 + rank) across the legs a chunk appears in. RRF only reads positions,
so cosine distance and ts_rank never have to be normalised onto a shared
scale — an ad-hoc weighted sum of the two is exactly where hybrid search
usually goes wrong.

Both legs return 50 candidates and fusion keeps the top 20.

## Re-ranking

Those 20 go through a cross-encoder that scores each chunk against the
question directly instead of comparing two independent embeddings. It is
slower per pair but reads far more like relevance, and only the top 5 survive
into the prompt.

## Measured

On a 30-question evaluation set over 88 chunks, Recall@5 is 0.90 for the vector
leg alone, 0.87 for keyword alone, and 1.00 once fused. Re-ranking lifts
Recall@1 from 0.73 to 0.77 and MRR from 0.843 to 0.853, at the cost of one
question dropping out of the top five.
`,
  },
  {
    id: "d_deploy",
    title: "Deployment",
    path: "docs/deployment.md",
    source_type: "markdown",
    url: null,
    indexed_at: "2026-09-08T09:12:00Z",
    text: `# Deployment

## Local

docker compose up brings up Postgres with the pgvector extension, the API on
port 8000, and the web app on port 3000. The compose file seeds the corpus on
first boot, so an empty database still answers questions.

## Environment

- OPENAI_BASE_URL and OPENAI_API_KEY point at the LLM gateway
- DATABASE_URL points at Postgres
- EMBED_MODEL and EMBED_DIM must agree with the vector column width

Changing EMBED_MODEL after documents exist requires a migration: the column
is declared vector(1024), and a model with a different width will be rejected
by Postgres rather than silently corrupting the index.

## Index build order

Bulk load first, create the HNSW index afterwards. Building it up front makes
the initial ingest several times slower for no benefit.

At query time, set hnsw.ef_search to 100. The default trades away more recall
than this workload can afford.

## Streaming behind a proxy

The chat endpoint sets X-Accel-Buffering: no. Without it nginx buffers the
whole response and the answer arrives in one lump — the stream still works,
it just stops looking like one.
`,
  },
  {
    id: "d_readme",
    title: "enterprise-rag — README",
    path: "README.md",
    source_type: "github",
    url: "https://github.com/example-org/enterprise-rag/blob/7f3a91c/README.md",
    indexed_at: "2026-09-08T09:14:00Z",
    text: `# enterprise-rag

Mini enterprise RAG and document search agent: ingest internal docs, ask
questions, get answers that cite the exact lines they came from.

## Status

Frontend first. The chat UI, the SSE contract and the citation model are
implemented; the FastAPI service behind them is next.

## Quick start

    docker compose up
    open http://localhost:3000

## Grounding

The model is instructed to answer only from retrieved context and to mark
every claim with a bracketed citation. After generation, each citation is
checked against the chunks that were actually in the prompt. Ones that point
at nothing are stripped from the answer and counted, and the count is shown
in the UI rather than swallowed.
`,
  },
  {
    id: "d_onboarding",
    title: "Engineering Onboarding Handbook",
    path: "handbook/onboarding.pdf",
    source_type: "pdf",
    url: null,
    indexed_at: "2026-09-08T09:20:00Z",
    text: `Engineering Onboarding Handbook

Page 6 — Access and environments

New engineers get read access to staging on day one. Production access is
granted after the first on-call shadow rotation and requires a second
approver from the platform team.

Page 7 — Running services locally

Every service in the monorepo ships a compose file. The convention is that
docker compose up must be sufficient to get a working local environment with
seeded data; anything that needs a manual step after that is considered a
bug in the service, not in the onboarding.

Page 8 — Support rotation

The weekday rotation is one primary and one secondary. Handover happens at
10:00 in the support channel, with open incidents listed explicitly.
`,
  },
];

// --- Chunks ---------------------------------------------------------------

const CHUNKS: MockChunk[] = [
  {
    id: "c_hybrid_legs",
    doc_id: "d_retrieval",
    heading_path: "Hybrid Search & Re-ranking > The two legs",
    snippet: `The vector leg is a pgvector cosine search over the embedding column, backed
by an HNSW index. It finds paraphrases: a question about "how do I ship this"
still matches a section titled "Deployment".

The keyword leg is Postgres full-text search over a tsvector column with a GIN
index. It finds exact tokens: error codes, flag names, a config key spelled
one specific way. Those are the queries where embeddings are weakest.`,
  },
  {
    id: "c_fusion",
    doc_id: "d_retrieval",
    heading_path: "Hybrid Search & Re-ranking > Fusion",
    snippet: `The two rankings are combined with Reciprocal Rank Fusion, score = sum of
1 / (60 + rank) across the legs a chunk appears in. RRF only reads positions,
so cosine distance and ts_rank never have to be normalised onto a shared
scale — an ad-hoc weighted sum of the two is exactly where hybrid search
usually goes wrong.`,
  },
  {
    id: "c_rerank",
    doc_id: "d_retrieval",
    heading_path: "Hybrid Search & Re-ranking > Re-ranking",
    snippet: `Those 20 go through a cross-encoder that scores each chunk against the
question directly instead of comparing two independent embeddings. It is
slower per pair but reads far more like relevance, and only the top 5 survive
into the prompt.`,
  },
  {
    id: "c_eval",
    doc_id: "d_retrieval",
    heading_path: "Hybrid Search & Re-ranking > Measured",
    snippet: `On a 30-question evaluation set over 88 chunks, Recall@5 is 0.90 for the vector
leg alone, 0.87 for keyword alone, and 1.00 once fused. Re-ranking lifts
Recall@1 from 0.73 to 0.77 and MRR from 0.843 to 0.853, at the cost of one
question dropping out of the top five.`,
  },
  {
    id: "c_chunking",
    doc_id: "d_ingest",
    heading_path: "Ingestion Pipeline > Chunking",
    snippet: `Chunks are 400 tokens with 80 tokens of overlap. The splitter is
structure-aware: it breaks on Markdown headings first and only falls back to
a hard token split when a single section is too long to fit.

400 rather than the 500-1000 originally planned: every multilingual model
fastembed offers caps at 512 input tokens and truncates past that silently,
and the title and heading prefix has to fit inside that budget too.`,
  },
  {
    id: "c_tokenizer",
    doc_id: "d_ingest",
    heading_path: "Ingestion Pipeline > Chunking",
    snippet: `Token counts come from the embedding model's own tokenizer. Counting
characters and dividing by four is off by 30% on Thai text, which silently
truncates the tail of every chunk that contains any.`,
  },
  {
    id: "c_embed_input",
    doc_id: "d_ingest",
    heading_path: "Ingestion Pipeline > Embedding",
    snippet: `The text that gets embedded is not the text that gets displayed. Embedding
input is prefixed with the document title and heading path — a cheap version
of contextual retrieval that costs no extra LLM call and measurably lifts
recall on short chunks that would otherwise lose their context.`,
  },
  {
    id: "c_permalink",
    doc_id: "d_ingest",
    heading_path: "Ingestion Pipeline > Connectors",
    snippet: `GitHub permalinks are built from the commit SHA, never from the branch name.
A link built against main rots the moment someone edits the file; a link
against the SHA points at the exact text that was indexed.`,
  },
  {
    id: "c_onedb",
    doc_id: "d_arch",
    heading_path: "Architecture Overview > Why one database instead of two",
    snippet: `A dedicated vector store would mean keeping two systems in sync: a chunk
deleted in Postgres has to disappear from the vector index in the same
transaction, or a stale chunk gets cited and the citation link 404s.

Putting the embedding column and the tsvector column on the same row removes
the problem entirely. Hybrid search becomes one SQL statement, metadata
filters are a WHERE clause, and a deleted document takes its vectors with it.`,
  },
  {
    id: "c_path",
    doc_id: "d_arch",
    heading_path: "Architecture Overview > Request path",
    snippet: `A question goes: embed → vector leg + keyword leg in parallel → RRF fusion →
cross-encoder re-rank → top 5 into the prompt → stream tokens back over SSE.
Sources are flushed to the client before the first token so the citation cards
render while the answer is still being written.`,
  },
  {
    id: "c_compose",
    doc_id: "d_deploy",
    heading_path: "Deployment > Local",
    snippet: `docker compose up brings up Postgres with the pgvector extension, the API on
port 8000, and the web app on port 3000. The compose file seeds the corpus on
first boot, so an empty database still answers questions.`,
  },
  {
    id: "c_index_order",
    doc_id: "d_deploy",
    heading_path: "Deployment > Index build order",
    snippet: `Bulk load first, create the HNSW index afterwards. Building it up front makes
the initial ingest several times slower for no benefit.

At query time, set hnsw.ef_search to 100. The default trades away more recall
than this workload can afford.`,
  },
  {
    id: "c_proxy",
    doc_id: "d_deploy",
    heading_path: "Deployment > Streaming behind a proxy",
    snippet: `The chat endpoint sets X-Accel-Buffering: no. Without it nginx buffers the
whole response and the answer arrives in one lump — the stream still works,
it just stops looking like one.`,
  },
  {
    id: "c_grounding",
    doc_id: "d_readme",
    heading_path: "enterprise-rag > Grounding",
    snippet: `The model is instructed to answer only from retrieved context and to mark
every claim with a bracketed citation. After generation, each citation is
checked against the chunks that were actually in the prompt. Ones that point
at nothing are stripped from the answer and counted, and the count is shown
in the UI rather than swallowed.`,
  },
  {
    id: "c_local_env",
    doc_id: "d_onboarding",
    heading_path: "Onboarding > Running services locally",
    page: 7,
    snippet: `Every service in the monorepo ships a compose file. The convention is that
docker compose up must be sufficient to get a working local environment with
seeded data; anything that needs a manual step after that is considered a
bug in the service, not in the onboarding.`,
  },
];

// --- Canned answers -------------------------------------------------------

export const ANSWERS: MockAnswer[] = [
  {
    keywords: ["hybrid", "search", "retriev", "rrf", "fusion", "rerank", "re-rank", "ค้นหา"],
    answer: `Retrieval runs two independent legs and fuses them.

The **vector leg** is a pgvector cosine search over the embedding column with an HNSW index — it catches paraphrases, so a question phrased "how do I ship this" still reaches a section titled "Deployment". The **keyword leg** is Postgres full-text search over a tsvector column with a GIN index, which is what actually finds error codes, flag names and exact config keys — the queries embeddings are worst at [1].

The two rankings are merged with Reciprocal Rank Fusion, \`1 / (60 + rank)\` summed across the legs a chunk appeared in. Because RRF reads only positions, cosine distance and \`ts_rank\` never have to be normalised onto a common scale — which is the usual place a hand-weighted hybrid score goes wrong [2].

Each leg returns 50 candidates, fusion keeps 20, and a cross-encoder scores those 20 against the question directly before the top 5 enter the prompt [3].

Measured over 30 questions and 88 chunks, fusion is where the gain is: Recall@5 goes from 0.90 vector-only and 0.87 keyword-only to **1.00** once fused. Re-ranking then trades a little recall for precision — Recall@1 0.73 → 0.77, MRR 0.843 → 0.853, with one question falling out of the top five [4].`,
    cites: [
      { chunk_id: "c_hybrid_legs", vector_rank: 1, keyword_rank: 2, rrf_score: 0.0325, rerank_score: 0.94 },
      { chunk_id: "c_fusion", vector_rank: 3, keyword_rank: 1, rrf_score: 0.0323, rerank_score: 0.91 },
      { chunk_id: "c_rerank", vector_rank: 2, keyword_rank: null, rrf_score: 0.0161, rerank_score: 0.87 },
      { chunk_id: "c_eval", vector_rank: null, keyword_rank: 4, rrf_score: 0.0156, rerank_score: 0.72 },
      { chunk_id: "c_path", vector_rank: 7, keyword_rank: 9, rrf_score: 0.0294, rerank_score: 0.55 },
    ],
  },
  {
    keywords: ["chunk", "token", "overlap", "split", "embed", "ingest", "แบ่ง"],
    answer: `Chunks are **400 tokens with 80 tokens of overlap**, and the splitter is structure-aware: it breaks on Markdown headings first and only hard-splits when one section is too long to fit [1].

400 rather than the 500–1000 originally planned, because every multilingual model fastembed offers caps at 512 input tokens and truncates past that silently — and the title/heading prefix has to fit in the same budget [1].

Token counts come from the embedding model's own tokenizer, not from a characters-÷-4 estimate — that estimate is off by about 30% on Thai text, which silently truncates the tail of every chunk containing any [2].

One detail worth knowing: the text that is embedded is not the text that is displayed. The embedding input is prefixed with the document title and heading breadcrumb, a cheap form of contextual retrieval that costs no extra LLM call and lifts recall on short chunks that would otherwise lose their context [3].`,
    cites: [
      { chunk_id: "c_chunking", vector_rank: 1, keyword_rank: 1, rrf_score: 0.0328, rerank_score: 0.96 },
      { chunk_id: "c_tokenizer", vector_rank: 2, keyword_rank: 4, rrf_score: 0.0317, rerank_score: 0.83 },
      { chunk_id: "c_embed_input", vector_rank: 4, keyword_rank: null, rrf_score: 0.0156, rerank_score: 0.76 },
      { chunk_id: "c_permalink", vector_rank: 9, keyword_rank: 12, rrf_score: 0.0284, rerank_score: 0.41 },
    ],
  },
  {
    keywords: ["deploy", "docker", "compose", "run", "local", "proxy", "nginx", "รัน", "ติดตั้ง"],
    answer: `\`docker compose up\` brings up Postgres with pgvector, the API on port 8000 and the web app on port 3000, and seeds the corpus on first boot so an empty database still answers [1].

That matches the house rule in the onboarding handbook: a single compose command has to be enough for a working local environment with seeded data, and anything needing a manual step afterwards counts as a bug in the service [2].

Two deployment details that bite:

- Bulk load **before** creating the HNSW index, and set \`hnsw.ef_search = 100\` at query time — the default gives up more recall than this workload can afford [3].
- The chat endpoint sets \`X-Accel-Buffering: no\`. Without it nginx buffers the whole response and the answer lands in one lump, so the stream stops looking like a stream [4].`,
    cites: [
      { chunk_id: "c_compose", vector_rank: 1, keyword_rank: 1, rrf_score: 0.0328, rerank_score: 0.95 },
      { chunk_id: "c_local_env", vector_rank: 3, keyword_rank: 2, rrf_score: 0.0322, rerank_score: 0.88 },
      { chunk_id: "c_index_order", vector_rank: 5, keyword_rank: null, rrf_score: 0.0154, rerank_score: 0.69 },
      { chunk_id: "c_proxy", vector_rank: null, keyword_rank: 3, rrf_score: 0.0159, rerank_score: 0.64 },
    ],
  },
  {
    keywords: ["架构", "architecture", "postgres", "pgvector", "database", "why", "สถาปัตย"],
    answer: `Everything lives in one Postgres 16 database with pgvector, rather than Postgres plus a dedicated vector store.

The reason is consistency. Two systems means a chunk deleted in Postgres has to vanish from the vector index in the same transaction — miss that and a stale chunk gets cited, and the citation link 404s. Keeping the embedding column and the tsvector column on the same row removes the failure mode: hybrid search is one SQL statement, metadata filters are a \`WHERE\` clause, and deleting a document takes its vectors with it [1].

The trade-off is scale — pgvector with HNSW is comfortable into the low millions of chunks, and past that a dedicated store earns its complexity back.

The request path itself is: embed → vector + keyword legs in parallel → RRF fusion → cross-encoder re-rank → top 5 into the prompt → tokens streamed back over SSE, with sources flushed before the first token so citation cards render while the answer is still being written [2].`,
    cites: [
      { chunk_id: "c_onedb", vector_rank: 1, keyword_rank: 3, rrf_score: 0.0322, rerank_score: 0.93 },
      { chunk_id: "c_path", vector_rank: 2, keyword_rank: 1, rrf_score: 0.0326, rerank_score: 0.81 },
      { chunk_id: "c_compose", vector_rank: 8, keyword_rank: null, rrf_score: 0.0147, rerank_score: 0.34 },
    ],
  },
  {
    keywords: ["citation", "hallucin", "ground", "อ้างอิง", "มั่ว"],
    answer: `The model is told to answer only from the retrieved context and to mark every claim with a bracketed citation. After generation, each citation is checked against the chunks that were actually in the prompt — any that point at nothing are stripped from the answer and counted, and that count is shown in the UI instead of being swallowed [1].

Citations stay clickable because the line range is written at split time and never recomputed [2], and GitHub links are built from the commit SHA rather than the branch, so a link keeps pointing at the exact text that was indexed even after someone edits the file [3].`,
    cites: [
      { chunk_id: "c_grounding", vector_rank: 1, keyword_rank: 1, rrf_score: 0.0328, rerank_score: 0.97 },
      { chunk_id: "c_chunking", vector_rank: 4, keyword_rank: null, rrf_score: 0.0156, rerank_score: 0.58 },
      { chunk_id: "c_permalink", vector_rank: 2, keyword_rank: 5, rrf_score: 0.0308, rerank_score: 0.86 },
    ],
  },
];

/**
 * No canned answer matched. Rather than inventing something, the mock does
 * what the real prompt is instructed to do when retrieval comes back weak —
 * which is the behaviour actually worth demoing.
 */
export const FALLBACK: MockAnswer = {
  keywords: [],
  answer: `I could not find anything in the indexed corpus that answers this.

The six indexed documents cover the architecture, the ingestion pipeline, hybrid search and re-ranking, deployment, the project README and the engineering onboarding handbook. Nothing retrieved above the relevance floor for this question, so there is no grounded answer to give — rather than guess, the honest response is that the corpus does not contain it.

Try asking about chunking, hybrid search, or how to run the stack locally.`,
  cites: [],
};

export function pickAnswer(question: string): MockAnswer {
  const q = question.toLowerCase();
  let best: MockAnswer | null = null;
  let bestHits = 0;

  for (const a of ANSWERS) {
    const hits = a.keywords.filter((k) => q.includes(k)).length;
    if (hits > bestHits) {
      best = a;
      bestHits = hits;
    }
  }
  return best ?? FALLBACK;
}

// --- Source assembly ------------------------------------------------------

const docById = new Map(DOCS.map((d) => [d.id, d]));
const chunkById = new Map(CHUNKS.map((c) => [c.id, c]));

/**
 * Line numbers are derived from where the snippet actually sits in the
 * document text, so the range on a citation card and the range the source
 * viewer highlights can never drift apart.
 */
function lineRange(doc: MockDoc, snippet: string): [number, number] | null {
  const at = doc.text.indexOf(snippet);
  if (at === -1) return null;
  const start = doc.text.slice(0, at).split("\n").length;
  return [start, start + snippet.split("\n").length - 1];
}

export function buildSources(answer: MockAnswer): Source[] {
  return answer.cites.map((cite, i) => {
    const chunk = chunkById.get(cite.chunk_id);
    if (!chunk) throw new Error(`unknown mock chunk: ${cite.chunk_id}`);
    const doc = docById.get(chunk.doc_id);
    if (!doc) throw new Error(`unknown mock doc: ${chunk.doc_id}`);

    const range = chunk.page ? null : lineRange(doc, chunk.snippet);

    return {
      n: i + 1,
      chunk_id: chunk.id,
      document_id: doc.id,
      title: doc.title,
      source_type: doc.source_type,
      path: doc.path,
      url:
        doc.url && range
          ? `${doc.url}#L${range[0]}-L${range[1]}`
          : doc.url,
      heading_path: chunk.heading_path,
      locator: {
        line_start: range ? range[0] : null,
        line_end: range ? range[1] : null,
        page: chunk.page ?? null,
      },
      retrieval: {
        vector_rank: cite.vector_rank,
        keyword_rank: cite.keyword_rank,
        rrf_score: cite.rrf_score,
        rerank_score: cite.rerank_score,
      },
      snippet: chunk.snippet,
    };
  });
}

export const CORPUS_STATS = {
  documents: DOCS.length,
  chunks: CHUNKS.length,
  // Matches EMBED_MODEL in the API's .env — the demo should not advertise a
  // model the real pipeline is not using.
  embed_model: "BAAI/bge-small-en-v1.5",
  demo: true,
};

export function documentSummaries() {
  return DOCS.map((d) => ({
    id: d.id,
    title: d.title,
    path: d.path,
    source_type: d.source_type,
    url: d.url,
    chunk_count: CHUNKS.filter((c) => c.doc_id === d.id).length,
    indexed_at: d.indexed_at,
  }));
}

export function documentText(id: string) {
  const doc = docById.get(id);
  return doc ? { id: doc.id, path: doc.path, text: doc.text } : null;
}
