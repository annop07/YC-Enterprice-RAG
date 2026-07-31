"use client";

import {
  IconBrandGithub,
  IconFileTypePdf,
  IconMarkdown,
} from "@tabler/icons-react";
import { locatorLabel, retrievalLabel } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Source, SourceType } from "@/lib/types";

const ICONS: Record<SourceType, typeof IconMarkdown> = {
  markdown: IconMarkdown,
  pdf: IconFileTypePdf,
  github: IconBrandGithub,
};

export function CitationCards({
  sources,
  retrieval,
  onOpen,
  activeChunkId,
}: {
  sources: Source[];
  retrieval: { candidates_considered: number; retrieval_ms: number } | null;
  onOpen: (source: Source) => void;
  activeChunkId: string | null;
}) {
  if (sources.length === 0) return null;

  return (
    <section className="space-y-2.5">
      <div className="flex items-baseline gap-2">
        <h3 className="font-medium text-foreground text-xs">Sources</h3>
        {retrieval && (
          <p className="text-muted-foreground text-xs">
            <span className="font-mono tabular-nums">
              {retrieval.candidates_considered}
            </span>
            {" candidates fused, top "}
            <span className="font-mono tabular-nums">{sources.length}</span>
            {" kept · "}
            <span className="font-mono tabular-nums">
              {retrieval.retrieval_ms}ms
            </span>
          </p>
        )}
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {sources.map((source) => (
          <CitationCard
            active={source.chunk_id === activeChunkId}
            key={source.chunk_id}
            onOpen={onOpen}
            source={source}
          />
        ))}
      </div>
    </section>
  );
}

function CitationCard({
  source,
  onOpen,
  active,
}: {
  source: Source;
  onOpen: (source: Source) => void;
  active: boolean;
}) {
  const Icon = ICONS[source.source_type];
  const rank = retrievalLabel(source);

  return (
    <button
      className={cn(
        "group flex w-full flex-col gap-2 overflow-hidden rounded-xl bg-card p-3 text-left",
        "ring-1 ring-foreground/10 transition-colors",
        "hover:bg-accent/60 focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-2",
        active && "bg-accent/60 ring-foreground/25",
      )}
      onClick={() => onOpen(source)}
      type="button"
    >
      <div className="flex w-full items-start gap-2">
        <span className="mt-px flex size-4.5 shrink-0 items-center justify-center rounded-[5px] border border-border bg-muted font-mono text-[10px] text-muted-foreground tabular-nums">
          {source.n}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate font-medium text-foreground text-xs">
            {source.title}
          </span>
          <span className="mt-0.5 block truncate font-mono text-[11px] text-muted-foreground">
            {locatorLabel(source)}
          </span>
        </span>
        <Icon
          aria-hidden
          className="mt-px shrink-0 text-muted-foreground"
          size={14}
          stroke={1.6}
        />
      </div>

      <p className="line-clamp-2 text-[11px] text-muted-foreground leading-5">
        {source.snippet.replace(/\s+/g, " ")}
      </p>

      <div className="flex w-full items-center gap-2">
        {/* The two ranks are the receipt for "hybrid": a chunk with only one
            of them is one the other leg never found. */}
        <span className="truncate font-mono text-[10px] text-muted-foreground">
          {rank || "—"}
        </span>
        {source.retrieval.rerank_score !== null && (
          <span className="ml-auto flex shrink-0 items-center gap-1.5">
            <span
              aria-hidden
              className="h-1 w-8 overflow-hidden rounded-full bg-muted"
            >
              <span
                className="block h-full rounded-full bg-foreground/40"
                style={{
                  width: `${Math.round(source.retrieval.rerank_score * 100)}%`,
                }}
              />
            </span>
            <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
              {source.retrieval.rerank_score.toFixed(2)}
            </span>
          </span>
        )}
      </div>
    </button>
  );
}
