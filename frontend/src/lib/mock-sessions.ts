/**
 * Session store for the mock backend.
 *
 * Server-only, like `mock-corpus.ts`: imported by the route handlers under
 * `app/api/mock/`, never by a component.
 *
 * It exists because the UI calls `GET /sessions` on mount and after every
 * turn. With no handler behind it those calls 404, and the client swallows
 * the failure — so the demo silently loses the whole session half of the
 * contract, and the sidebar reads "No conversations yet" no matter how much
 * you ask it. The mock is supposed to speak the same contract as the API;
 * this is the part of it that was missing.
 */
import type { SessionDetail, SessionSummary, Source, StoredMessage } from "./types";

interface MockSession {
  id: string;
  title: string;
  updated_at: string;
  messages: StoredMessage[];
}

/**
 * Parked on `globalThis` rather than held in a module variable: a route
 * handler module is re-evaluated on every hot reload, and a plain `const`
 * would drop the conversation you are in the middle of every time a file is
 * saved. It is still process memory — restarting the server starts the demo
 * over, which is the honest shape for a backend that does not exist.
 */
const globalForSessions = globalThis as unknown as {
  __ragMockSessions?: Map<string, MockSession>;
};
const sessions = (globalForSessions.__ragMockSessions ??= new Map<string, MockSession>());

//: Same cut as `store.derive_title` in the API, so the sidebar row and the
//: header read identically whichever backend answered.
const TITLE_CHARS = 56;

export function deriveTitle(firstMessage: string): string {
  const line = firstMessage.trim().split("\n")[0];
  return line.length > TITLE_CHARS ? `${line.slice(0, TITLE_CHARS)}…` : line || "New chat";
}

/**
 * The title the `session` event should carry for this turn.
 *
 * Derived from the *first* user message of the conversation, never the latest
 * one — a follow-up question must not rename the session it belongs to. That
 * is what the API does, and a session whose title changed under the user
 * would be the mock disagreeing with it.
 */
export function titleFor(sessionId: string, question: string): string {
  return sessions.get(sessionId)?.title ?? deriveTitle(question);
}

let counter = 0;
const nextId = () => `m_${Date.now().toString(36)}_${counter++}`;

/** Persist one question/answer pair. Returns the assistant message id. */
export function saveTurn(turn: {
  session_id: string;
  question: string;
  answer: string;
  sources: Source[];
  meta: Record<string, unknown>;
}): string {
  const session: MockSession = sessions.get(turn.session_id) ?? {
    id: turn.session_id,
    title: deriveTitle(turn.question),
    updated_at: "",
    messages: [],
  };

  const userId = nextId();
  const assistantId = nextId();
  session.messages.push(
    { id: userId, role: "user", content: turn.question, sources: [], meta: null },
    {
      id: assistantId,
      role: "assistant",
      content: turn.answer,
      sources: turn.sources,
      meta: turn.meta,
    },
  );
  session.updated_at = new Date().toISOString();
  sessions.set(session.id, session);

  return assistantId;
}

export function listSessions(): SessionSummary[] {
  // ISO-8601 sorts lexicographically in chronological order, so this is the
  // same "most recently updated first" the API gets from an ORDER BY.
  return [...sessions.values()]
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    .map(({ id, title, updated_at }) => ({ id, title, updated_at }));
}

export function getSession(id: string): SessionDetail | null {
  const session = sessions.get(id);
  if (!session) return null;
  // Copied on the way out: the caller renders this, and handing it the array
  // the next turn is about to push onto is a mutation waiting to happen.
  return { ...session, messages: [...session.messages] };
}

export function deleteSession(id: string): boolean {
  return sessions.delete(id);
}
