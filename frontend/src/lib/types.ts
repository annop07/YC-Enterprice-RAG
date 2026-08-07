/**
 * Types mirroring the FastAPI backend schemas (api/app/schemas.py).
 *
 * The frontend was built before the backend, so this file *is* the contract:
 * whatever ships in `api/` has to serialise to exactly these shapes.
 */

export type SourceType = "markdown" | "pdf" | "github";

/**
 * Where in the corpus a retrieved chunk came from.
 *
 * A chunk is located either by line range (Markdown, GitHub) or by page
 * (PDF) — never both — so the viewer has to handle each separately rather
 * than assuming line numbers exist.
 */
export interface Locator {
  line_start: number | null;
  line_end: number | null;
  page: number | null;
}

/**
 * Why a chunk made the cut. Kept in the payload — and shown on the citation
 * card — because "hybrid search" is a claim, and these two ranks are the
 * evidence: a chunk the vector leg missed but keyword found (or the reverse)
 * is the whole reason the two legs are fused instead of one being used alone.
 */
export interface RetrievalTrace {
  /** Rank in the vector (pgvector cosine) leg, null if it never surfaced */
  vector_rank: number | null;
  /** Rank in the Postgres full-text leg, null if it never surfaced */
  keyword_rank: number | null;
  /** Reciprocal Rank Fusion score of the two legs above */
  rrf_score: number;
  /** Cross-encoder score after re-ranking — what the final order is by */
  rerank_score: number | null;
}

export interface Source {
  /** Citation number the answer refers to as `[n]` — 1-based */
  n: number;
  chunk_id: string;
  document_id: string;
  /** Human title of the document, e.g. "Deployment Guide" */
  title: string;
  source_type: SourceType;
  /** Path inside the corpus, e.g. "docs/deploy.md" */
  path: string;
  /** Deep link to the real thing — GitHub permalink, uploaded file route */
  url: string | null;
  /** Heading breadcrumb inside the document, e.g. "Setup > Docker" */
  heading_path: string | null;
  locator: Locator;
  retrieval: RetrievalTrace;
  /** The chunk text itself — what the LLM actually read */
  snippet: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// --- SSE events -----------------------------------------------------------
// Order on the wire is: session → sources → token* → done. `sources` arrives
// before the first token on purpose, so citation cards can render while the
// answer is still being written instead of popping in at the end.

export interface SessionEvent {
  session_id: string;
  title: string;
}

export interface SourcesEvent {
  sources: Source[];
  /** Candidates that survived fusion before the re-ranker trimmed them */
  candidates_considered: number;
  retrieval_ms: number;
}

export interface TokenEvent {
  text: string;
}

export interface DoneEvent {
  message_id: string;
  latency_ms: number;
  usage: Usage;
  /**
   * Citations the model emitted that pointed at nothing it was shown. They
   * are stripped from the answer; the count is surfaced instead of hidden,
   * so the hallucination rate is visible in the UI.
   */
  dropped_citations: number;
  model: string;
}

export interface ErrorEvent {
  detail: string;
}

export type ChatEvent =
  | { event: "session"; data: SessionEvent }
  | { event: "sources"; data: SourcesEvent }
  | { event: "token"; data: TokenEvent }
  | { event: "done"; data: DoneEvent }
  | { event: "error"; data: ErrorEvent };

// --- Chat state -----------------------------------------------------------

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  /** Search-side stats, available before the first token */
  retrieval: { candidates_considered: number; retrieval_ms: number } | null;
  /** Present once the answer finished streaming */
  meta: DoneEvent | null;
  error: string | null;
}

export interface SessionSummary {
  id: string;
  title: string;
  updated_at: string;
}

/**
 * A message as the server stored it. `meta` is deliberately loose: it is
 * whatever the `done` event carried at the time, so an answer from an older
 * model still renders with the model and cost it actually ran at.
 */
export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  meta: Record<string, unknown> | null;
}

export interface SessionDetail {
  id: string;
  title: string;
  updated_at: string;
  messages: StoredMessage[];
}

// --- Corpus ---------------------------------------------------------------

export interface DocumentSummary {
  id: string;
  title: string;
  path: string;
  source_type: SourceType;
  url: string | null;
  chunk_count: number;
  indexed_at: string;
}

export interface CorpusStats {
  documents: number;
  chunks: number;
  /** Embedding model backing the vector leg, e.g. "BAAI/bge-m3" */
  embed_model: string;
  /** True while no real backend is configured and the mock route answers */
  demo: boolean;
}
