"use client";

import { Fragment, type ReactNode } from "react";
import { scanBlocks, type Block } from "@/lib/markdown-blocks";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/types";

/**
 * The answer as the model writes it: plain Markdown with `[n]` citation
 * markers, rendered live on a string whose last line is usually half a word.
 *
 * It is rendered by hand rather than with a Markdown library for one reason:
 * this runs on a half-finished answer on every token, and a real parser
 * either throws or re-lays-out the whole answer when it meets `**bo`. Here an
 * unclosed marker stays literal text until its closing marker arrives, and an
 * unclosed fence renders as the code block it is about to be.
 *
 * Structure comes from [`scanBlocks`](@/lib/markdown-blocks), which is where
 * the rules and the reasoning about partial input live. This file is the
 * inline pass — bold, code spans and citations — and the markup.
 */

// Built per call rather than hoisted: a /g regex carries `lastIndex`, and one
// shared instance would hand its cursor to the next message being rendered.
//
// Italics are deliberately absent. `*` and `_` both appear inside identifiers
// and paths that answers over technical documents are full of — `__init__`,
// `snake_case`, `SELECT *` — and mangling those is a worse failure than not
// emphasising a word.
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
  const blocks = scanBlocks(text);

  return (
    <div className="space-y-3 text-pretty text-[0.9375rem] text-foreground leading-7">
      {blocks.map((block, i) => (
        <BlockView
          block={block}
          caret={streaming && i === blocks.length - 1}
          key={i}
          onCite={onCite}
          sources={sources}
        />
      ))}
      {/* A table or a rule has nowhere sensible to put the cursor, so when one
          is what is currently arriving it gets its own line. */}
      {streaming && needsOwnCaretLine(blocks) && (
        <p>
          <Caret />
        </p>
      )}
    </div>
  );
}

function needsOwnCaretLine(blocks: Block[]): boolean {
  const last = blocks[blocks.length - 1];
  return last === undefined || last.kind === "table" || last.kind === "rule";
}

function BlockView({
  block,
  sources,
  onCite,
  caret,
}: {
  block: Block;
  sources: Source[];
  onCite: (n: number) => void;
  caret: boolean;
}) {
  const inline = (text: string) => (
    <Inline onCite={onCite} sources={sources} text={text} />
  );

  switch (block.kind) {
    case "code":
      return (
        <div className="relative">
          {block.lang && (
            <span className="absolute top-1.5 right-2.5 font-mono text-[10px] text-muted-foreground">
              {block.lang}
            </span>
          )}
          <pre className="scroll-thin overflow-x-auto rounded-lg border border-border bg-muted/60 px-3 py-2.5">
            <code className="font-mono text-[0.8125rem] leading-6">
              {block.code}
              {caret && <Caret />}
            </code>
          </pre>
        </div>
      );

    case "heading": {
      // Nested inside an article that already owns h1..h3, so the answer's own
      // headings start below them rather than competing with the page outline.
      const Tag = `h${Math.min(6, block.level + 3)}` as "h4" | "h5" | "h6";
      return (
        <Tag
          className={cn(
            "font-heading font-semibold text-foreground tracking-tight",
            block.level <= 2 ? "pt-1 text-base" : "text-[0.9375rem]",
          )}
        >
          {inline(block.text)}
          {caret && <Caret />}
        </Tag>
      );
    }

    case "list": {
      const List = block.ordered ? "ol" : "ul";
      return (
        <List className="ml-1 space-y-1.5" start={block.ordered ? block.start : undefined}>
          {block.items.map((item, i) => (
            <li
              className="flex gap-2.5"
              key={i}
              // Two spaces of source indent is one step in, capped so a badly
              // indented answer cannot push its text off the panel.
              style={
                item.indent >= 2
                  ? { marginLeft: `${Math.min(Math.floor(item.indent / 2), 4) * 0.875}rem` }
                  : undefined
              }
            >
              {block.ordered ? (
                <span className="w-4 shrink-0 text-right font-mono text-muted-foreground text-xs tabular-nums leading-7">
                  {block.start + i}.
                </span>
              ) : (
                <span
                  aria-hidden
                  className="mt-2.5 size-1 shrink-0 rounded-full bg-muted-foreground"
                />
              )}
              <span className="min-w-0">
                {inline(item.text)}
                {caret && i === block.items.length - 1 && <Caret />}
              </span>
            </li>
          ))}
        </List>
      );
    }

    case "table":
      return (
        <div className="scroll-thin overflow-x-auto rounded-lg ring-1 ring-foreground/10">
          <table className="w-full border-collapse text-left text-[0.8125rem]">
            <thead>
              <tr className="bg-muted/60">
                {block.header.map((cell, i) => (
                  <th
                    className="border-border border-b px-3 py-2 font-medium text-muted-foreground"
                    key={i}
                  >
                    {inline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, r) => (
                <tr className="border-border border-b last:border-b-0" key={r}>
                  {row.map((cell, c) => (
                    <td className="px-3 py-2 align-top leading-6 tabular-nums" key={c}>
                      {inline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

    case "quote":
      return (
        <blockquote className="border-border border-l-2 pl-3 text-muted-foreground">
          {inline(block.text)}
          {caret && <Caret />}
        </blockquote>
      );

    case "rule":
      return <hr className="border-border" />;

    default:
      return (
        <p>
          {inline(block.text)}
          {caret && <Caret />}
        </p>
      );
  }
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
