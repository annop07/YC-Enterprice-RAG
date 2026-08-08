"""Structure-aware, token-aware chunking with exact line ranges.

Two properties matter here and everything else follows from them:

1. **Chunk boundaries fall on line boundaries.** That is what makes
   `line_start`/`line_end` exact, and an exact line range is what makes a
   citation clickable rather than decorative. The one exception is a single
   line too long to fit, which is split on token offsets — those pieces all
   report the same line number, which is still true.

2. **Token counts come from the embedding model's own tokenizer.** A
   characters-over-four estimate is off by ~30% on Thai, and the failure mode
   is silent: the model truncates at 512 and the tail of the chunk is simply
   never embedded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Sequence

ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")

#: Counts tokens for a batch of strings. Injected so the chunker is testable
#: without loading a 67 MB ONNX model.
TokenCounter = Callable[[Sequence[str]], list[int]]
#: Character offsets of each token in a string, for splitting over-long lines.
OffsetFn = Callable[[str], list[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    content: str
    heading_path: str | None
    line_start: int
    line_end: int
    token_count: int
    page: int | None = None


@dataclass(frozen=True)
class _Unit:
    """A line, or a slice of a line that was too long to keep whole."""

    text: str
    line_no: int
    tokens: int
    heading_path: str | None


def in_fenced_code(lines: Sequence[str]) -> list[bool]:
    """Whether each line sits inside a fenced code block, markers included.

    Both breadcrumbs and title extraction need this and must not disagree about
    it, so the fence state machine lives here once rather than in each caller.

    A block is closed only by its own marker, so ``` inside a ~~~ block is
    content. An unterminated fence runs to the end of the document: that is
    what CommonMark says, and guessing a close would make one stray ``` in a
    README silently re-interpret everything below it as prose.
    """
    flags: list[bool] = []
    fence: str | None = None

    for line in lines:
        fence_match = FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            flags.append(True)  # the marker line is never itself a heading
            continue
        flags.append(fence is not None)

    return flags


def heading_paths(lines: Sequence[str]) -> list[str | None]:
    """Breadcrumb for every line, e.g. "Ingestion Pipeline > Chunking".

    A heading line gets the path *including itself*, so a chunk that starts at
    "## Chunking" is labelled with that section rather than the one above it.
    Fenced code is skipped — a shell comment starting with `#` is not a heading.
    """
    stack: list[tuple[int, str]] = []
    out: list[str | None] = []

    for line, fenced in zip(lines, in_fenced_code(lines)):
        if not fenced:
            heading = ATX_HEADING.match(line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                if title:
                    stack.append((level, title))

        out.append(" > ".join(t for _, t in stack) or None)

    return out


def _split_long_line(
    line: str, line_no: int, path: str | None, max_tokens: int, offsets_of: OffsetFn
) -> list[_Unit]:
    """Cut one over-long line on token boundaries, not character guesses."""
    offsets = offsets_of(line)
    if not offsets:
        return []

    units: list[_Unit] = []
    for start in range(0, len(offsets), max_tokens):
        window = offsets[start : start + max_tokens]
        text = line[window[0][0] : window[-1][1]]
        units.append(_Unit(text, line_no, len(window), path))
    return units


def _to_units(
    lines: Sequence[str],
    paths: Sequence[str | None],
    counts: Sequence[int],
    max_tokens: int,
    offsets_of: OffsetFn,
) -> list[_Unit]:
    units: list[_Unit] = []
    for i, line in enumerate(lines):
        line_no = i + 1
        if counts[i] > max_tokens:
            units.extend(_split_long_line(line, line_no, paths[i], max_tokens, offsets_of))
        else:
            units.append(_Unit(line, line_no, counts[i], paths[i]))
    return units


def _trim_blank_edges(window: list[_Unit]) -> list[_Unit]:
    first = next((i for i, u in enumerate(window) if u.text.strip()), None)
    if first is None:
        return []
    last = len(window) - 1
    while not window[last].text.strip():
        last -= 1
    return window[first : last + 1]


def chunk_markdown(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
    token_offsets: OffsetFn,
) -> list[Chunk]:
    lines = text.split("\n")
    paths = heading_paths(lines)
    units = _to_units(lines, paths, count_tokens(lines), max_tokens, token_offsets)
    if not units:
        return []

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0

    while start < len(units):
        end = start  # exclusive
        total = 0

        while end < len(units):
            nxt = units[end].tokens
            if total + nxt > max_tokens and end > start:
                break
            # A heading starts a new section; break before it once the current
            # chunk is substantial, so sections stay whole where they can.
            if (
                end > start
                and total >= max_tokens // 2
                and ATX_HEADING.match(units[end].text)
            ):
                break
            total += nxt
            end += 1

        # Blank lines at either edge are dropped from the content, so they must
        # be dropped from the range too — otherwise a chunk claims a line it
        # does not actually contain, and the citation highlight starts one line
        # early. The reported range has to be exactly what `content` covers.
        window = _trim_blank_edges(units[start:end])

        if window:
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    content="\n".join(u.text for u in window),
                    heading_path=window[0].heading_path,
                    line_start=window[0].line_no,
                    line_end=window[-1].line_no,
                    token_count=sum(u.tokens for u in window),
                )
            )
            ordinal += 1

        if end >= len(units):
            break

        # Step back over whole units until roughly `overlap_tokens` of the
        # previous chunk is carried into the next one.
        #
        # The floor bounds how far back that can go. Blank lines cost zero
        # tokens, so a purely token-based budget lets the walk-back cross any
        # number of them for free; in a document of short lines it can retreat
        # most of the way to the start, advance a line or two per chunk, and
        # emit piles of near-duplicate chunks. Giving back at most half the
        # window makes progress a property of the loop rather than a hope.
        floor = start + max(1, (end - start) // 2)
        back = 0
        cursor = end
        while cursor > floor and back + units[cursor - 1].tokens <= overlap_tokens:
            cursor -= 1
            back += units[cursor].tokens

        # A single line can be wider than the whole overlap budget, and blank
        # lines cost nothing to step over — so the loop above can "overlap" by
        # a couple of empty lines and hand the next chunk no shared context at
        # all. If nothing with text was carried, carry the last real line,
        # unless it is big enough to eat half the next chunk.
        if overlap_tokens > 0 and not any(u.text.strip() for u in units[cursor:end]):
            last_content = next(
                (i for i in range(end - 1, floor - 1, -1) if units[i].text.strip()), None
            )
            if last_content is not None and units[last_content].tokens <= max_tokens // 2:
                cursor = last_content

        start = cursor

    return chunks


#: Pages are joined with a blank line between them when a PDF is flattened into
#: `document.text`. Both the assembler and the chunker read this constant, so
#: the line offsets they compute cannot drift apart.
PAGE_SEPARATOR = "\n\n"


def assemble_pages(pages: Sequence[str]) -> tuple[str, list[int]]:
    """Flatten pages into one document, and say where each page starts.

    The source viewer needs a single text to render; the citation needs to know
    which page a chunk came from. Keeping the per-page start lines lets a chunk
    carry both a page number and a line range into the flattened text.
    """
    text = PAGE_SEPARATOR.join(pages)
    starts: list[int] = []
    line = 1
    # A page occupies `count("\n") + 1` lines, and the separator's first
    # newline is the one that ends the page's last line — so only the newlines
    # after it add lines of their own.
    separator_lines = PAGE_SEPARATOR.count("\n")
    for page in pages:
        starts.append(line)
        line += page.count("\n") + separator_lines
    return text, starts


def chunk_pages(
    pages: Sequence[str],
    *,
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter,
    token_offsets: OffsetFn,
) -> list[Chunk]:
    """Chunk a paginated document.

    A chunk never spans a page boundary, so the page it reports is unambiguous.
    Line numbers are shifted into the flattened document, which is what the
    source viewer renders.
    """
    _, starts = assemble_pages(pages)
    chunks: list[Chunk] = []
    ordinal = 0

    for page_no, (page_text, start_line) in enumerate(zip(pages, starts), start=1):
        for chunk in chunk_markdown(
            page_text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            count_tokens=count_tokens,
            token_offsets=token_offsets,
        ):
            chunks.append(
                replace(
                    chunk,
                    ordinal=ordinal,
                    page=page_no,
                    line_start=chunk.line_start + start_line - 1,
                    line_end=chunk.line_end + start_line - 1,
                )
            )
            ordinal += 1

    return chunks


def build_embed_input(title: str, heading_path: str | None, content: str) -> str:
    """What actually gets embedded.

    Prefixing the title and heading breadcrumb is a cheap form of contextual
    retrieval: it costs no extra LLM call and it stops short chunks — a bare
    list, a code block — from losing every clue about what they belong to.
    The prefix is not shown in the citation; only `content` is.
    """
    if not heading_path:
        breadcrumb = title
    elif heading_path == title or heading_path.startswith(f"{title} > "):
        # The document title usually *is* the H1, so the breadcrumb already
        # starts with it. Prepending again wastes tokens out of a 512 budget.
        breadcrumb = heading_path
    else:
        breadcrumb = f"{title} > {heading_path}"
    return f"{breadcrumb}\n\n{content}"
