"use client";

import { useCallback, useRef, useState } from "react";
import { streamChat } from "@/lib/api";
import { emptyAssistantMessage, userMessage } from "@/lib/messages";
import type { ChatMessage } from "@/lib/types";

/**
 * `retrieving` is a distinct state from `streaming` on purpose: the gap
 * between them is search time, and showing "searching 6 documents" there is
 * more honest than a spinner that implies the model is already writing.
 */
export type ChatStatus = "idle" | "retrieving" | "streaming";

export interface UseChat {
  messages: ChatMessage[];
  status: ChatStatus;
  send: (text: string) => Promise<void>;
  stop: () => void;
  load: (messages: ChatMessage[]) => void;
}

let counter = 0;
const nextId = () => `m_${Date.now().toString(36)}_${counter++}`;

export function useChat(sessionId: string): UseChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const abortRef = useRef<AbortController | null>(null);
  // `status` updates asynchronously, so a fast double-submit can slip past a
  // check on it and open two streams. A ref flips synchronously.
  const inFlight = useRef(false);

  const patch = useCallback((id: string, fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));
  }, []);

  const send = useCallback(
    async (text: string) => {
      const prompt = text.trim();
      if (!prompt || inFlight.current) return;
      inFlight.current = true;

      const assistantId = nextId();
      setMessages((prev) => [
        ...prev,
        userMessage(nextId(), prompt),
        emptyAssistantMessage(assistantId),
      ]);
      setStatus("retrieving");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        for await (const ev of streamChat(
          { message: prompt, session_id: sessionId },
          controller.signal,
        )) {
          switch (ev.event) {
            case "sources":
              patch(assistantId, (m) => ({
                ...m,
                sources: ev.data.sources,
                retrieval: {
                  candidates_considered: ev.data.candidates_considered,
                  retrieval_ms: ev.data.retrieval_ms,
                },
              }));
              setStatus("streaming");
              break;
            case "token":
              patch(assistantId, (m) => ({
                ...m,
                content: m.content + ev.data.text,
              }));
              break;
            case "done":
              patch(assistantId, (m) => ({ ...m, meta: ev.data }));
              break;
            case "error":
              patch(assistantId, (m) => ({ ...m, error: ev.data.detail }));
              break;
            default:
              break; // `session` is handled by the shell that owns the id
          }
        }
      } catch (e) {
        // A user-initiated stop is not a failure — the partial answer stays
        // on screen and only a real error gets an error line.
        if (!(e instanceof DOMException && e.name === "AbortError")) {
          patch(assistantId, (m) => ({
            ...m,
            error: e instanceof Error ? e.message : "request failed",
          }));
        }
      } finally {
        abortRef.current = null;
        inFlight.current = false;
        setStatus("idle");
      }
    },
    [patch, sessionId],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const load = useCallback((next: ChatMessage[]) => {
    abortRef.current?.abort();
    setMessages(next);
    setStatus("idle");
  }, []);

  return { messages, status, send, stop, load };
}
