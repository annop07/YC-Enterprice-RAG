// Thin typed client for the RAG backend.
import type {
  ChatEvent,
  CorpusStats,
  DocumentSummary,
  IngestJob,
  SessionDetail,
  SessionSummary,
  Source,
} from "./types";

/**
 * Unset means "no FastAPI yet" — requests fall through to the mock route
 * handlers in `app/api/mock/`, which speak the same SSE contract. Point this
 * at the real service and nothing else in the UI changes.
 */
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "/api/mock";
export const IS_DEMO = !process.env.NEXT_PUBLIC_API_BASE;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${detail.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export async function getCorpusStats(): Promise<CorpusStats> {
  return json(await fetch(`${API_BASE}/corpus`, { cache: "no-store" }));
}

export async function getDocuments(): Promise<DocumentSummary[]> {
  return json(await fetch(`${API_BASE}/documents`, { cache: "no-store" }));
}

/** Starts indexing and returns immediately. Poll it with `getJob`. */
export async function ingestFiles(files: File[]): Promise<IngestJob> {
  const body = new FormData();
  for (const file of files) body.append("files", file);
  // No Content-Type header on purpose: the browser has to set the multipart
  // boundary itself, and setting it by hand produces a body the server cannot parse.
  return json(await fetch(`${API_BASE}/ingest/files`, { method: "POST", body }));
}

export async function ingestGitHub(request: {
  repo: string;
  path_prefix?: string;
  ref?: string;
}): Promise<IngestJob> {
  return json(
    await fetch(`${API_BASE}/ingest/github`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}

/** One ingest job, as it stands right now. */
export async function getJob(id: string): Promise<IngestJob> {
  return json(
    await fetch(`${API_BASE}/jobs/${encodeURIComponent(id)}`, { cache: "no-store" }),
  );
}

export async function getSessions(): Promise<SessionSummary[]> {
  return json(await fetch(`${API_BASE}/sessions`, { cache: "no-store" }));
}

export async function getSession(id: string): Promise<SessionDetail> {
  return json(
    await fetch(`${API_BASE}/sessions/${encodeURIComponent(id)}`, {
      cache: "no-store",
    }),
  );
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/sessions/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    throw new Error(`API ${res.status}`);
  }
}

export async function deleteDocument(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) throw new Error(`API ${res.status}`);
}

/** Full text of one document, for the source viewer. */
export async function getDocumentText(
  documentId: string,
): Promise<{ id: string; path: string; text: string }> {
  return json(
    await fetch(`${API_BASE}/documents/${encodeURIComponent(documentId)}`, {
      cache: "no-store",
    }),
  );
}

/**
 * Stream one answer.
 *
 * `EventSource` is not an option here: it is GET-only and cannot carry a
 * message body, so the stream is read off a plain `fetch` POST instead and
 * the SSE framing is parsed by hand below.
 */
export async function* streamChat(
  body: { message: string; session_id?: string; top_k?: number },
  signal: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${detail.slice(0, 200)}`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;

      // SSE frames are separated by a blank line. A chunk can split mid-frame,
      // so only whole frames are consumed and the remainder stays buffered.
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

function parseFrame(frame: string): ChatEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // keep-alive comment
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }

  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) } as ChatEvent;
  } catch {
    return null;
  }
}

// --- Display helpers ------------------------------------------------------

/** "docs/deploy.md:12-48" or "handbook.pdf p.7" — the string under a citation. */
export function locatorLabel(source: Source): string {
  const { line_start, line_end, page } = source.locator;
  if (page !== null) return `${source.path} · p.${page}`;
  if (line_start !== null) {
    const range =
      line_end !== null && line_end !== line_start
        ? `${line_start}-${line_end}`
        : `${line_start}`;
    return `${source.path}:${range}`;
  }
  return source.path;
}

/** "vector #2 · keyword #5" — empty leg omitted rather than shown as null. */
export function retrievalLabel(source: Source): string {
  const { vector_rank, keyword_rank } = source.retrieval;
  const parts: string[] = [];
  if (vector_rank !== null) parts.push(`vector #${vector_rank}`);
  if (keyword_rank !== null) parts.push(`keyword #${keyword_rank}`);
  return parts.join(" · ");
}
