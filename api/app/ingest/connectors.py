"""Sources of documents.

A connector is the only thing that knows where text comes from. Everything
downstream — chunking, embedding, storage — sees `RawDocument` and nothing
else, so adding Confluence or Google Drive is one new class here rather than
a second pipeline.
"""
from __future__ import annotations

import bisect
import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Protocol

from app.ingest.chunker import in_fenced_code

SourceType = Literal["markdown", "pdf", "github"]

MARKDOWN_SUFFIXES = {".md", ".mdx", ".markdown"}
#: Vendored trees are full of Markdown that is not this corpus. Without this,
#: pointing the connector at a project root indexes every dependency's README.
SKIP_DIRS = {"node_modules", "__pycache__", "dist", "build", "target", "site-packages"}
_MD_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
#: Profile and project READMEs routinely open with `<h1 align="center">`, which
#: no Markdown heading regex will ever see. Attributes may be quoted, unquoted
#: or absent, and both the tag and its contents may wrap across lines — `[^>]*`
#: and DOTALL cover that without pulling in an HTML parser.
_HTML_H1 = re.compile(r"<h1\b[^>]*>(.*?)</h1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
#: Tags that stand for a break between words. Everything else is inline markup
#: whose removal must *not* insert one, or `<strong>Hi</strong>, there` would
#: come out as "Hi , there".
_BREAK_TAG = re.compile(r"<\s*/?\s*(?:br|p|div|hr|li|tr|td|th)\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]*>")


@dataclass(frozen=True)
class RawDocument:
    """One document as pulled from a source, before chunking."""

    source_type: SourceType
    #: Stable identity *within* a source — a relative path, or "owner/repo@path".
    #: Re-running a connector must produce the same value or the document will
    #: be inserted a second time instead of updated.
    source_id: str
    title: str
    #: What the citation shows, e.g. "docs/deploy.md".
    path: str
    text: str
    url: str | None = None
    meta: dict = field(default_factory=dict)
    #: Set only by paginated sources. Its presence is what tells the pipeline
    #: to chunk per page and stamp a page number on every chunk.
    pages: tuple[str, ...] | None = None

    @property
    def id(self) -> str:
        digest = hashlib.sha1(f"{self.source_type}:{self.source_id}".encode())
        return digest.hexdigest()[:16]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


class Connector(Protocol):
    """Anything that can yield documents."""

    name: str

    def load(self) -> Iterator[RawDocument]: ...


def _strip_inline_html(fragment: str) -> str:
    """Plain text out of an HTML heading's contents.

    Badges, logo images and linked names are markup around the title, not the
    title, so every tag comes out and what is left is collapsed to one line.
    """
    fragment = _COMMENT.sub("", fragment)
    fragment = _BREAK_TAG.sub(" ", fragment)
    fragment = _ANY_TAG.sub("", fragment)
    # Last, so that an escaped `&lt;b&gt;` survives as literal text instead of
    # being unescaped into a tag and then stripped.
    fragment = html.unescape(fragment)
    return " ".join(fragment.split())


def _is_titleish(text: str) -> bool:
    """Whether a stripped heading says anything a citation card could show.

    A badge-only header strips to nothing and a rule-like one to `---`; both
    are worse than the filename. Letters, digits and emoji all count, so
    non-ASCII titles are never rejected.
    """
    return any(unicodedata.category(ch)[0] not in "PZC" for ch in text)


def title_from_markdown(text: str, fallback: str) -> str:
    """First usable heading — Markdown `#` or HTML `<h1>` — else the filename.

    Whichever kind comes first in the document wins: a README that opens with
    a centered `<h1>` and only later has `## Install`-style Markdown means the
    centered one as its title, and the reverse holds just as much. Headings
    that strip to nothing usable are skipped rather than returned empty.

    Fenced code is not a heading source: a `# ` in an opening ```bash usage
    block is a shell comment, and an `<h1>` there is markup being demonstrated.
    An HTML heading is judged by the line its opening tag is on.
    """
    lines = text.split("\n")
    fenced = in_fenced_code(lines)
    line_starts: list[int] = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1  # the newline `split` consumed

    def is_prose(start: int) -> bool:
        return not fenced[bisect.bisect_right(line_starts, start) - 1]

    candidates = [
        (m.start(), m.group(1).strip())
        for m in _MD_H1.finditer(text)
        if is_prose(m.start())
    ]
    candidates += [
        (m.start(), _strip_inline_html(m.group(1)))
        for m in _HTML_H1.finditer(text)
        if is_prose(m.start())
    ]

    for _, title in sorted(candidates, key=lambda c: c[0]):
        if _is_titleish(title):
            return title
    return fallback.replace("-", " ").replace("_", " ").strip() or fallback


def iter_files(root: Path, suffixes: set[str]) -> tuple[list[Path], Path]:
    """Files of the given kinds under `root`, and the base their paths are relative to.

    Dot-directories and vendored trees are skipped here rather than in each
    connector, so `DirConnector` and the PDF walk cannot disagree about what
    counts as part of the corpus.
    """
    root = root.resolve()
    if root.is_file():
        return ([root] if root.suffix.lower() in suffixes else []), root.parent

    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        parts = path.relative_to(root).parts
        if any(p.startswith(".") for p in parts):
            continue  # .git, .venv, dot-directories generally
        if any(p in SKIP_DIRS for p in parts):
            continue
        found.append(path)
    return found, root


class DirConnector:
    """Every Markdown file under a directory.

    Paths are stored relative to the root so the same corpus ingested from a
    different absolute location updates its rows instead of duplicating them.
    """

    name = "dir"

    def __init__(self, root: Path, suffixes: set[str] | None = None) -> None:
        self.root = root.resolve()
        self.suffixes = suffixes or MARKDOWN_SUFFIXES

    def load(self) -> Iterator[RawDocument]:
        candidates, base = iter_files(self.root, self.suffixes)

        for path in candidates:
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            rel = path.relative_to(base).as_posix()
            yield RawDocument(
                source_type="markdown",
                source_id=rel,
                title=title_from_markdown(text, path.stem),
                path=rel,
                text=text,
                meta={"bytes": path.stat().st_size},
            )


class FileConnector:
    """Explicit files — what an upload endpoint will hand over."""

    name = "file"

    def __init__(self, paths: list[Path]) -> None:
        self.paths = [p.resolve() for p in paths]

    def load(self) -> Iterator[RawDocument]:
        for path in self.paths:
            if path.suffix.lower() not in MARKDOWN_SUFFIXES:
                raise ValueError(
                    f"{path.name}: FileConnector handles Markdown; PDF arrives with PDFConnector"
                )
            text = path.read_text(encoding="utf-8", errors="replace")
            yield RawDocument(
                source_type="markdown",
                source_id=path.name,
                title=title_from_markdown(text, path.stem),
                path=path.name,
                text=text,
            )
