"""PDF extraction via PyMuPDF.

PDFs are the source where "we indexed it" and "we can answer from it" come
apart most often: a scanned document has pages, has a page count, imports
without error, and contains no extractable text at all. The connector counts
what it got and says so rather than quietly indexing nothing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pymupdf

from app.ingest.chunker import assemble_pages
from app.ingest.connectors import RawDocument

log = logging.getLogger(__name__)

#: Below this many characters per page, the file is almost certainly scanned
#: images with no text layer, and needs OCR rather than an indexer.
MIN_CHARS_PER_PAGE = 40


def extract_pages(data: bytes) -> list[str]:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return [page.get_text("text").strip() for page in doc]


def pdf_document(data: bytes, *, name: str, source_id: str | None = None) -> RawDocument:
    pages = extract_pages(data)
    text, _ = assemble_pages(pages)

    non_empty = [p for p in pages if p.strip()]
    density = len(text) / len(pages) if pages else 0
    if pages and density < MIN_CHARS_PER_PAGE:
        log.warning(
            "%s: %d of %d pages have text and only %.0f chars per page — this "
            "looks like a scan without a text layer. It will index as almost "
            "nothing; run OCR over it first.",
            name,
            len(non_empty),
            len(pages),
            density,
        )

    stem = Path(name).stem
    title = _title_from_pdf(data) or stem.replace("-", " ").replace("_", " ")
    return RawDocument(
        source_type="pdf",
        source_id=source_id or name,
        title=title.strip() or stem,
        path=name,
        text=text,
        pages=tuple(pages),
        meta={"pages": len(pages), "pages_with_text": len(non_empty)},
    )


def _title_from_pdf(data: bytes) -> str | None:
    """Document metadata title, when the producer bothered to set one."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        title = (doc.metadata or {}).get("title")
    return title.strip() if title and title.strip() else None


class PDFConnector:
    """PDF files from disk."""

    name = "pdf"

    def __init__(self, paths: list[Path], root: Path | None = None) -> None:
        self.paths = [p.resolve() for p in paths]
        self.root = root.resolve() if root else None

    def load(self) -> Iterator[RawDocument]:
        for path in self.paths:
            rel = (
                path.relative_to(self.root).as_posix()
                if self.root and path.is_relative_to(self.root)
                else path.name
            )
            yield pdf_document(path.read_bytes(), name=rel)
