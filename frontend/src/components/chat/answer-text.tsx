"use client";

import { Fragment, type ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/types";

/**
 * The answer arrives as plain text with `**bold**`, `` `code` `` and `[n]`
 * citation markers — the shape an LLM produces without being asked for HTML.
 *
 * It is rendered by hand rather than with a markdown library for one reason:
 * this runs on a half-finished string on every token, and a real parser
 * either throws or re-lays-out the whole answer when it meets `**bo`. Here an
 * unclosed marker just stays literal text until its closing marker arrives.
 */
// Built per call rather than hoisted: a /g regex carries `lastIndex`, and one
// shared instance would hand its cursor to the next message being rendered.
const inlinePattern = () => /\*\*(.+?)\*\*|`([^`]+)`|\[(\d+)\]/g;

export function AnswerText({
  text,
  sources,
  onCite,
  streaming = false,
}: {
  text: string;
  sources: Source[];
  onCite: (n: number) => void;
  streaming?: boolean;
}) {
  const blocks = text.split(/\n{2,}/);

  return (
    <div className="space-y-3 text-pretty text-[0.9375rem] text-foreground leading-7">
      {blocks.map((block, bi) => {
        const last = bi === blocks.length - 1;
        const lines = block.split("\n");
        const isList = lines.every((l) => /^\s*[-*]\s+/.test(l)) && block.trim() !== "";

        if (isList) {
          return (
            <ul className="ml-1 space-y-1.5" key={bi}>
              {lines.map((line, li) => (
                <li className="flex gap-2.5" key={li}>
                  <span aria-hidden className="mt-2.5 size-1 shrink-0 rounded-full bg-muted-foreground" />
                  <span>
                    <Inline
                      onCite={onCite}
                      sources={sources}
                      text={line.replace(/^\s*[-*]\s+/, "")}
                    />
                    {last && li === lines.length - 1 && streaming && <Caret />}
                  </span>
                </li>
              ))}
            </ul>
          );
        }

        return (
          <p key={bi}>
            <Inline onCite={onCite} sources={sources} text={block} />
            {last && streaming && <Caret />}
          </p>
        );
      })}
    </div>
  );
}

function Caret() {
  return (
    <span
      aria-hidden
      className="caret-blink ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-foreground"
    />
  );
}

function Inline({
  text,
  sources,
  onCite,
}: {
  text: string;
  sources: Source[];
  onCite: (n: number) => void;
}) {
  const out: ReactNode[] = [];
  let cursor = 0;
  let key = 0;

  const pattern = inlinePattern();
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) out.push(text.slice(cursor, match.index));

    const [, bold, code, citation] = match;
    if (bold !== undefined) {
      out.push(
        <strong className="font-semibold" key={key++}>
          {bold}
        </strong>,
      );
    } else if (code !== undefined) {
      out.push(
        <code
          className="rounded-[5px] bg-muted px-1 py-0.5 font-mono text-[0.85em] text-foreground"
          key={key++}
        >
          {code}
        </code>,
      );
    } else if (citation !== undefined) {
      const n = Number(citation);
      const source = sources.find((s) => s.n === n);
      out.push(
        source ? (
          <CitationChip key={key++} n={n} onCite={onCite} source={source} />
        ) : (
          // A marker with no matching source is a citation the guard would
          // have stripped. Shown greyed rather than as a live link.
          <span
            className="text-muted-foreground text-xs"
            key={key++}
            title="citation did not match any retrieved chunk"
          >
            [{n}]
          </span>
        ),
      );
    }
    cursor = match.index + match[0].length;
  }

  if (cursor < text.length) out.push(text.slice(cursor));

  return (
    <>
      {out.map((node, i) => (
        <Fragment key={i}>{node}</Fragment>
      ))}
    </>
  );
}

function CitationChip({
  n,
  source,
  onCite,
}: {
  n: number;
  source: Source;
  onCite: (n: number) => void;
}) {
  return (
    <button
      className={cn(
        // No trailing margin: a citation is nearly always followed by a
        // period, and a gap there reads as a typo.
        "ml-[0.15em] inline-flex h-[1.15em] min-w-[1.15em] translate-y-[-0.15em] items-center justify-center rounded-[5px]",
        "border border-border bg-muted px-[0.3em] align-middle font-mono text-[0.7em] text-muted-foreground tabular-nums",
        "transition-colors hover:border-foreground/30 hover:bg-accent hover:text-foreground",
        "focus-visible:outline-2 focus-visible:outline-ring focus-visible:outline-offset-2",
      )}
      onClick={() => onCite(n)}
      title={`${source.title} — ${source.path}`}
      type="button"
    >
      {n}
    </button>
  );
}
