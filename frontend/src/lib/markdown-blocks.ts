/**
 * Block-level scan of a Markdown answer that is still being written.
 *
 * The renderer this feeds ([`answer-text.tsx`](../components/chat/answer-text.tsx))
 * runs on every token, over a string whose last line is usually half a word.
 * A conventional Markdown parser is the wrong tool for that: it either throws
 * on the incomplete input or re-lays-out the whole answer each time the shape
 * of a partial construct changes.
 *
 * So the scan is deliberately one pass, line by line, and every rule is
 * written to have a sensible reading of a truncated document rather than an
 * error. An unterminated fence runs to the end of the text and renders as a
 * code block — which is exactly right while one is arriving. A `|` row with no
 * delimiter under it yet is still a paragraph, and becomes a table on the line
 * that proves it is one.
 *
 * Splitting text into blocks used to be `text.split(/\n{2,}/)`, which is where
 * this file comes from: a fenced code block with a blank line in it was cut
 * into pieces, and the inline pass then matched the third backtick of ``` to
 * the first backtick of the closing fence and rendered the whole listing as
 * one inline `<code>` with the language name inside it. Headings, numbered
 * lists and tables had no representation at all and reached the screen as
 * their own source.
 *
 * Kept apart from the component so it can be exercised without React.
 */

export interface ListItem {
  /** Leading spaces, so a nested bullet renders indented without a tree. */
  indent: number;
  text: string;
}

export type Block =
  | { kind: "code"; lang: string; code: string; closed: boolean }
  | { kind: "heading"; level: number; text: string }
  | { kind: "list"; ordered: boolean; start: number; items: ListItem[] }
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "quote"; text: string }
  | { kind: "rule" }
  | { kind: "paragraph"; text: string };

// Up to three leading spaces still opens a block in CommonMark; four makes it
// an indented code block, which nothing here needs to support because models
// fence their code.
const FENCE = /^\s{0,3}(`{3,}|~{3,})(.*)$/;
const HEADING = /^\s{0,3}(#{1,6})(?:\s+(.*?))?\s*$/;
const RULE = /^\s{0,3}([-*_])(?:\s*\1){2,}\s*$/;
const BULLET = /^(\s*)[-*+]\s+(.*)$/;
const ORDERED = /^(\s*)(\d{1,9})[.)]\s+(.*)$/;
const QUOTE = /^\s{0,3}>\s?(.*)$/;
/** A closed ATX heading — `## Setup ##` — carries no text in its trailing run. */
const TRAILING_HASHES = /\s+#+$/;

/**
 * A fence is closed only by a run of its own character at least as long as
 * the one that opened it, so ``` inside a ~~~ block is content — the same
 * rule `in_fenced_code` applies on the ingestion side, and for the same
 * reason: guessing a close would let one stray fence re-read everything
 * below it.
 */
function closesFence(line: string, marker: string): boolean {
  const trimmed = line.trim();
  if (trimmed.length < marker.length) return false;
  for (const char of trimmed) if (char !== marker[0]) return false;
  return true;
}

/** `| --- | :--: |` — pipes, dashes, colons and space, and at least one dash. */
function isTableDelimiter(line: string): boolean {
  const trimmed = line.trim();
  return (
    trimmed.includes("|") && trimmed.includes("-") && /^[|\s:-]+$/.test(trimmed)
  );
}

/** Cells of one table row. Escaped pipes are not supported — nor emitted. */
function splitRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

/** A header row is only a table once the row under it says so. */
function opensTable(lines: string[], i: number): boolean {
  return (
    lines[i].includes("|") &&
    i + 1 < lines.length &&
    isTableDelimiter(lines[i + 1])
  );
}

/** Whether this line starts something that is not a paragraph. */
function opensBlock(lines: string[], i: number): boolean {
  const line = lines[i];
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    RULE.test(line) ||
    BULLET.test(line) ||
    ORDERED.test(line) ||
    QUOTE.test(line) ||
    opensTable(lines, i)
  );
}

export function scanBlocks(source: string): Block[] {
  const lines = source.split("\n");
  const blocks: Block[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i++;
      continue;
    }

    // --- fenced code ------------------------------------------------------
    const fence = FENCE.exec(line);
    if (fence) {
      const marker = fence[1];
      const body: string[] = [];
      i++;
      while (i < lines.length && !closesFence(lines[i], marker)) {
        body.push(lines[i]);
        i++;
      }
      const closed = i < lines.length;
      if (closed) i++;
      blocks.push({
        kind: "code",
        // The info string may carry more than the language; the first word of
        // it is the part anyone displays.
        lang: fence[2].trim().split(/\s+/)[0] ?? "",
        code: body.join("\n"),
        closed,
      });
      continue;
    }

    // --- heading ----------------------------------------------------------
    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({
        kind: "heading",
        level: heading[1].length,
        text: (heading[2] ?? "").replace(TRAILING_HASHES, "").trim(),
      });
      i++;
      continue;
    }

    // Before the list rules: `---` and `***` are rules, not bullets with no text.
    if (RULE.test(line)) {
      blocks.push({ kind: "rule" });
      i++;
      continue;
    }

    // --- table ------------------------------------------------------------
    if (opensTable(lines, i)) {
      const header = splitRow(line);
      i += 2; // the header and the delimiter under it
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      blocks.push({ kind: "table", header, rows });
      continue;
    }

    // --- list -------------------------------------------------------------
    const bullet = BULLET.exec(line);
    const ordered = ORDERED.exec(line);
    if (bullet || ordered) {
      const isOrdered = ordered !== null;
      const items: ListItem[] = [];
      const start = ordered ? Number(ordered[2]) : 1;

      while (i < lines.length && lines[i].trim()) {
        const nextBullet = BULLET.exec(lines[i]);
        const nextOrdered = ORDERED.exec(lines[i]);

        if (isOrdered && nextOrdered) {
          items.push({ indent: nextOrdered[1].length, text: nextOrdered[3] });
          i++;
        } else if (!isOrdered && nextBullet && !nextOrdered) {
          items.push({ indent: nextBullet[1].length, text: nextBullet[2] });
          i++;
        } else if (nextBullet || nextOrdered || opensBlock(lines, i)) {
          break; // the list ended and something else began
        } else {
          // A wrapped item: the model hard-wrapped one bullet over two lines,
          // and the second line belongs to the first rather than to nothing.
          items[items.length - 1].text += ` ${lines[i].trim()}`;
          i++;
        }
      }

      blocks.push({ kind: "list", ordered: isOrdered, start, items });
      continue;
    }

    // --- quote ------------------------------------------------------------
    if (QUOTE.test(line)) {
      const quoted: string[] = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        quoted.push(QUOTE.exec(lines[i])![1]);
        i++;
      }
      blocks.push({ kind: "quote", text: quoted.join("\n") });
      continue;
    }

    // --- paragraph --------------------------------------------------------
    // The first line is taken unconditionally — the checks above have already
    // ruled out every other kind of block, and consuming it is what
    // guarantees the loop advances.
    const paragraph = [line];
    i++;
    while (i < lines.length && lines[i].trim() && !opensBlock(lines, i)) {
      paragraph.push(lines[i]);
      i++;
    }
    blocks.push({ kind: "paragraph", text: paragraph.join("\n") });
  }

  return blocks;
}
