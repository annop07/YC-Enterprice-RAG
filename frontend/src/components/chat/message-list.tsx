"use client";

import { IconAlertTriangle, IconSearch, IconSparkles } from "@tabler/icons-react";
import { AnswerText } from "@/components/chat/answer-text";
import { CitationCards } from "@/components/chat/citation-cards";
import type { ChatStatus } from "@/hooks/use-chat";
import type { ChatMessage, Source } from "@/lib/types";

const fmt = (n: number) => n.toLocaleString();

export function MessageList({
  messages,
  status,
  onOpenSource,
  activeChunkId,
}: {
  messages: ChatMessage[];
  status: ChatStatus;
  onOpenSource: (source: Source) => void;
  activeChunkId: string | null;
}) {
  return (
    <div className="space-y-8">
      {messages.map((message, i) =>
        message.role === "user" ? (
          <UserBubble content={message.content} key={message.id} />
        ) : (
          <Answer
            activeChunkId={activeChunkId}
            key={message.id}
            live={i === messages.length - 1 ? status : "idle"}
            message={message}
            onOpenSource={onOpenSource}
          />
        ),
      )}
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] whitespace-pre-wrap rounded-xl bg-muted px-3.5 py-2.5 text-[0.9375rem] text-foreground leading-6">
        {content}
      </p>
    </div>
  );
}

function Answer({
  message,
  live,
  onOpenSource,
  activeChunkId,
}: {
  message: ChatMessage;
  live: ChatStatus;
  onOpenSource: (source: Source) => void;
  activeChunkId: string | null;
}) {
  const retrieving = live === "retrieving";
  const streaming = live === "streaming";

  return (
    <article className="space-y-4">
      <div className="flex items-center gap-2 text-muted-foreground text-xs">
        <IconSparkles aria-hidden size={14} stroke={1.6} />
        <span className="font-medium">Answer</span>
        {retrieving && (
          <span className="inline-flex items-center gap-1.5">
            <IconSearch
              className="animate-pulse motion-reduce:animate-none"
              size={12}
            />
            Searching the corpus…
          </span>
        )}
      </div>

      {/* Sources land before the first token, so they sit above the answer and
          are already readable while it is being written. */}
      <CitationCards
        activeChunkId={activeChunkId}
        onOpen={onOpenSource}
        retrieval={message.retrieval}
        sources={message.sources}
      />

      {message.content ? (
        <AnswerText
          onCite={(n) => {
            const source = message.sources.find((s) => s.n === n);
            if (source) onOpenSource(source);
          }}
          sources={message.sources}
          streaming={streaming}
          text={message.content}
        />
      ) : (
        retrieving && <SkeletonLines />
      )}

      {message.error && (
        <p className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-destructive text-sm">
          <IconAlertTriangle className="mt-0.5 shrink-0" size={14} />
          {message.error}
        </p>
      )}

      {message.meta && <MetaLine meta={message.meta} />}
    </article>
  );
}

function MetaLine({ meta }: { meta: NonNullable<ChatMessage["meta"]> }) {
  return (
    <p className="flex flex-wrap items-center gap-x-2 gap-y-1 border-border border-t pt-3 text-muted-foreground text-xs">
      <span className="font-mono text-foreground">{meta.model}</span>
      <span aria-hidden>·</span>
      <span className="font-mono tabular-nums">
        {(meta.latency_ms / 1000).toFixed(1)}s
      </span>
      <span aria-hidden>·</span>
      <span className="font-mono tabular-nums">
        {fmt(meta.usage.prompt_tokens)} → {fmt(meta.usage.completion_tokens)}
      </span>
      <span>tokens</span>
      <span aria-hidden>·</span>
      {/* Printed even when zero: a hallucination rate you can only see when
          it is non-zero is one nobody looks at. */}
      <span className={meta.dropped_citations > 0 ? "text-destructive" : undefined}>
        <span className="font-mono tabular-nums">{meta.dropped_citations}</span>
        {" citations dropped"}
      </span>
    </p>
  );
}

function SkeletonLines() {
  return (
    <div
      aria-hidden
      className="space-y-2.5 motion-safe:animate-pulse"
    >
      <div className="h-3.5 w-[92%] rounded bg-muted" />
      <div className="h-3.5 w-[78%] rounded bg-muted" />
      <div className="h-3.5 w-[85%] rounded bg-muted" />
    </div>
  );
}
