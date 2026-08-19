"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "@/lib/api";
import { emptyAssistantMessage, userMessage } from "@/lib/messages";
import type { ChatMessage, SessionEvent } from "@/lib/types";

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

export interface UseChatOptions {
  /**
   * The `session` frame, which lands before retrieval even starts.
   *
   * It carries the two things the server, not the client, is the authority
   * on: the id the turn was actually filed under — the client proposes one,
   * the server is free to hand back another — and the session title, which is
   * derived from the *first* question of the conversation and is what the
   * sidebar row shows. Whoever owns the session id gets told; nobody has to
   * derive either value a second way and hope the two agree.
   */
  onSession?: (event: SessionEvent) => void;
}

let counter = 0;
const nextId = () => `m_${Date.now().toString(36)}_${counter++}`;

export function useChat(sessionId: string, options: UseChatOptions = {}): UseChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>("idle");
  const abortRef = useRef<AbortController | null>(null);
  // Read through a ref so the caller can pass an inline closure without
  // handing `send` a new identity on every render.
  const onSessionRef = useRef(options.onSession);
  useEffect(() => {
    onSessionRef.current = options.onSession;
  }, [options.onSession]);
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
            case "session":
              // Only while this turn is still the one on screen. `session` is
              // the first frame and the callback moves the shell, so a user
              // who starts a chat and immediately opens another one can be
              // dragged back into the first by a frame that was already in
              // flight. Every other frame here patches a message by id and is
              // a harmless no-op once that message is gone; this one is not.
              if (!controller.signal.aborted) onSessionRef.current?.(ev.data);
              break;
            case "sources":
              patch(assistantId, (m) => ({
                ...m,
                sources: ev.data.sources,
                retrieval: {
                  candidates_considered: ev.data.candidates_considered,
                  retrieval_ms: ev.data.retrieval_ms,
                  notice: ev.data.notice ?? null,
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
              // An event this build does not know about is ignored rather
              // than treated as a failure, so the contract can grow a frame
              // without every older client breaking on it.
              break;
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
