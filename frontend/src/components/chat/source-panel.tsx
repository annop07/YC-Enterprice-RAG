"use client";

import { useEffect, useRef, useState } from "react";
import { IconExternalLink, IconLoader2, IconX } from "@tabler/icons-react";
import { getDocumentText, locatorLabel } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { Source } from "@/lib/types";

/**
 * The click-through half of a citation: the cited lines, in the real
 * document, highlighted in place. A citation that only names a file is a
 * claim; this is the part that makes it checkable.
 */
export function SourcePanel({
  source,
  onClose,
}: {
  source: Source | null;
  onClose: () => void;
}) {
  // Both results are tagged with the document they belong to, and read back
  // through a match on the current id. Clearing them at the top of the effect
  // instead would be a setState in an effect body — and would briefly show
  // the previous document's text under the new document's heading.
  const [loaded, setLoaded] = useState<{ id: string; text: string } | null>(null);
  const [failed, setFailed] = useState<{ id: string; message: string } | null>(
    null,
  );
  const highlightRef = useRef<HTMLDivElement>(null);

  const documentId = source?.document_id ?? null;
  const text = loaded && loaded.id === documentId ? loaded.text : null;
  const error = failed && failed.id === documentId ? failed.message : null;

  useEffect(() => {
    if (!documentId) return;
    let cancelled = false;

    getDocumentText(documentId)
      .then((doc) => {
        if (!cancelled) setLoaded({ id: documentId, text: doc.text });
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setFailed({
            id: documentId,
            message:
              e instanceof Error ? e.message : "could not load document",
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [documentId]);

  useEffect(() => {
    if (!source) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [source, onClose]);

  // Scroll the cited range into view once the text is on screen. Runs after
  // paint, so the ref is attached by the time it fires.
  useEffect(() => {
    if (text && highlightRef.current) {
      highlightRef.current.scrollIntoView({ block: "center" });
    }
  }, [text]);

  if (!source) return null;

  const { line_start, line_end } = source.locator;
  const lines = text?.split("\n") ?? [];

  return (
    <>
      <button
        aria-label="Close source"
        className="fixed inset-0 z-40 bg-foreground/10 backdrop-blur-[1px]"
        onClick={onClose}
        tabIndex={-1}
        type="button"
      />

      <aside
        aria-label={`Source: ${source.title}`}
        className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col border-border border-l bg-background shadow-xl"
      >
        <header className="flex items-start gap-3 border-border border-b px-5 py-4">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="flex size-4.5 shrink-0 items-center justify-center rounded-[5px] border border-border bg-muted font-mono text-[10px] text-muted-foreground tabular-nums">
                {source.n}
              </span>
              <h2 className="truncate font-heading font-semibold text-foreground text-sm tracking-tight">
                {source.title}
              </h2>
            </div>
            <p className="mt-1 truncate font-mono text-muted-foreground text-xs">
              {locatorLabel(source)}
            </p>
            {source.heading_path && (
              <p className="mt-1 truncate text-muted-foreground text-xs">
                {source.heading_path}
              </p>
            )}
          </div>

          <div className="flex shrink-0 items-center gap-1">
            {source.url && (
              <Button
                aria-label="Open original"
                className="rounded-md"
                render={
                  <a href={source.url} rel="noreferrer noopener" target="_blank" />
                }
                size="icon-sm"
                variant="ghost"
              >
                <IconExternalLink size={16} />
              </Button>
            )}
            <Button
              aria-label="Close"
              className="rounded-md"
              onClick={onClose}
              size="icon-sm"
              variant="ghost"
            >
              <IconX size={16} />
            </Button>
          </div>
        </header>

        <div className="flex flex-wrap items-center gap-1.5 border-border border-b px-5 py-2.5">
          <Badge className="font-normal" variant="outline">
            {source.source_type}
          </Badge>
          {source.retrieval.vector_rank !== null && (
            <Badge className="font-normal" variant="outline">
              <span className="text-muted-foreground">vector</span>
              <span className="ml-1 font-mono">#{source.retrieval.vector_rank}</span>
            </Badge>
          )}
          {source.retrieval.keyword_rank !== null && (
            <Badge className="font-normal" variant="outline">
              <span className="text-muted-foreground">keyword</span>
              <span className="ml-1 font-mono">#{source.retrieval.keyword_rank}</span>
            </Badge>
          )}
          <Badge className="font-normal" variant="outline">
            <span className="text-muted-foreground">rrf</span>
            <span className="ml-1 font-mono">
              {source.retrieval.rrf_score.toFixed(4)}
            </span>
          </Badge>
          {source.retrieval.rerank_score !== null && (
            <Badge className="font-normal" variant="outline">
              <span className="text-muted-foreground">rerank</span>
              <span className="ml-1 font-mono">
                {source.retrieval.rerank_score.toFixed(2)}
              </span>
            </Badge>
          )}
        </div>

        <div className="scroll-thin flex-1 overflow-y-auto px-2 py-4">
          {error && <p className="px-3 text-destructive text-sm">{error}</p>}

          {!text && !error && (
            <p className="flex items-center gap-2 px-3 text-muted-foreground text-sm">
              <IconLoader2 className="animate-spin motion-reduce:animate-none" size={14} />
              Loading document…
            </p>
          )}

          {text && line_start === null && (
            // PDFs have no line numbers to point at, so the chunk itself is
            // shown highlighted instead of a range inside the page text.
            <div className="space-y-4 px-3">
              <div
                className="rounded-lg bg-accent/70 px-3 py-2 text-sm leading-6 ring-1 ring-foreground/10"
                ref={highlightRef}
              >
                <pre className="whitespace-pre-wrap font-sans">{source.snippet}</pre>
              </div>
              <pre className="whitespace-pre-wrap font-mono text-muted-foreground text-xs leading-6">
                {text}
              </pre>
            </div>
          )}

          {text && line_start !== null && (
            <div className="font-mono text-xs leading-6">
              {lines.map((line, i) => {
                const no = i + 1;
                const inRange = no >= line_start && no <= (line_end ?? line_start);
                return (
                  <div
                    className={cn(
                      "flex gap-3 px-1",
                      inRange && "bg-accent/70",
                      inRange && no === line_start && "rounded-t-md",
                      inRange && no === (line_end ?? line_start) && "rounded-b-md",
                    )}
                    key={no}
                    ref={inRange && no === line_start ? highlightRef : undefined}
                  >
                    <span
                      aria-hidden
                      className={cn(
                        "w-8 shrink-0 select-none text-right tabular-nums",
                        inRange ? "text-foreground/60" : "text-muted-foreground/50",
                      )}
                    >
                      {no}
                    </span>
                    <span
                      className={cn(
                        "min-w-0 whitespace-pre-wrap break-words",
                        inRange ? "text-foreground" : "text-muted-foreground",
                      )}
                    >
                      {line || " "}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
