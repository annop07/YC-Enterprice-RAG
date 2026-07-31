/**
 * Chat sessions, persisted in localStorage.
 *
 * Exposed as an external store rather than effect-synced state: the server has
 * no localStorage, so `getServerSnapshot` returns an empty list and the client
 * swaps in the stored sessions on hydration without a cascading render.
 *
 * This is deliberately the *same* shape the backend will serve once
 * `chat_session` / `chat_message` tables exist — swapping the reads for HTTP
 * calls should not touch a single component.
 */
import type { ChatMessage, Source } from "./types";

const KEY = "enterprise-rag:sessions";

export interface StoredSession {
  id: string;
  title: string;
  updated_at: string;
  messages: ChatMessage[];
}

let listeners: (() => void)[] = [];

/**
 * `useSyncExternalStore` compares snapshots by identity and re-renders on any
 * change, so re-parsing JSON on every call would loop forever. The parsed list
 * is cached and only rebuilt when a write bumps the version.
 */
let cache: StoredSession[] | null = null;
const EMPTY: StoredSession[] = [];

function read(): StoredSession[] {
  if (cache) return cache;
  try {
    const raw = localStorage.getItem(KEY);
    cache = raw ? (JSON.parse(raw) as StoredSession[]) : EMPTY;
  } catch {
    cache = EMPTY;
  }
  return cache;
}

function write(next: StoredSession[]): void {
  cache = next;
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Quota or private mode — the in-memory cache still holds this session.
  }
  for (const l of listeners) l();
}

export function subscribeSessions(callback: () => void): () => void {
  listeners = [...listeners, callback];
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) {
      cache = null; // another tab wrote — drop the cache and re-read
      callback();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners = listeners.filter((l) => l !== callback);
    window.removeEventListener("storage", onStorage);
  };
}

export function getSessionsSnapshot(): StoredSession[] {
  return read();
}

export function getSessionsServerSnapshot(): StoredSession[] {
  return EMPTY;
}

export function newSessionId(): string {
  return `s_${Math.random().toString(36).slice(2, 10)}`;
}

export function upsertSession(session: StoredSession): void {
  const rest = read().filter((s) => s.id !== session.id);
  write([session, ...rest]);
}

export function deleteSession(id: string): void {
  write(read().filter((s) => s.id !== id));
}

export function getSession(id: string | null): StoredSession | null {
  if (!id) return null;
  return read().find((s) => s.id === id) ?? null;
}

/** First line of the opening question, trimmed — matches what the backend will title with. */
export function deriveTitle(firstMessage: string): string {
  const line = firstMessage.trim().split("\n")[0];
  return line.length > 56 ? `${line.slice(0, 56)}…` : line || "New chat";
}

export function emptyAssistantMessage(id: string): ChatMessage {
  return {
    id,
    role: "assistant",
    content: "",
    sources: [],
    retrieval: null,
    meta: null,
    error: null,
  };
}

export function userMessage(id: string, content: string): ChatMessage {
  return {
    id,
    role: "user",
    content,
    sources: [] as Source[],
    retrieval: null,
    meta: null,
    error: null,
  };
}
