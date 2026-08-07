import type { ChatMessage, Source, StoredMessage } from "./types";

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

export function newSessionId(): string {
  return `s_${Math.random().toString(36).slice(2, 14)}`;
}

/**
 * A persisted message rendered as if it had just streamed.
 *
 * The server stores the `done` payload as a loose `meta` object, so history and
 * live answers reach the same components through the same shape — the meta line
 * under an old answer shows the model and cost it actually ran at.
 */
export function fromStored(message: StoredMessage): ChatMessage {
  const meta = message.meta ?? {};
  const hasMeta = message.role === "assistant" && typeof meta.model === "string";

  return {
    id: message.id,
    role: message.role,
    content: message.content,
    sources: message.sources ?? [],
    retrieval:
      typeof meta.retrieval_ms === "number"
        ? {
            candidates_considered:
              typeof meta.candidates_considered === "number"
                ? meta.candidates_considered
                : (message.sources?.length ?? 0),
            retrieval_ms: meta.retrieval_ms,
          }
        : null,
    meta: hasMeta
      ? {
          message_id: message.id,
          latency_ms: typeof meta.latency_ms === "number" ? meta.latency_ms : 0,
          usage: (meta.usage as ChatMessage["meta"] extends null
            ? never
            : { prompt_tokens: number; completion_tokens: number; total_tokens: number }) ?? {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
          },
          dropped_citations:
            typeof meta.dropped_citations === "number" ? meta.dropped_citations : 0,
          model: meta.model as string,
        }
      : null,
    error: null,
  };
}
