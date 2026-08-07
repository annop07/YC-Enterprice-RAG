"""In-memory documents, for the upload endpoint.

A connector reads from somewhere; an upload arrives already read. This is the
one place that turns bytes plus a filename into the same `RawDocument` the
connectors produce, so nothing downstream has to know which door it came in by.
"""
from __future__ import annotations

from pathlib import Path

from app.ingest.connectors import MARKDOWN_SUFFIXES, RawDocument, title_from_markdown
from app.ingest.pdf import pdf_document

SUPPORTED = MARKDOWN_SUFFIXES | {".pdf"}


def document_from_upload(filename: str, data: bytes) -> RawDocument:
    name = Path(filename).name
    suffix = Path(name).suffix.lower()

    if suffix == ".pdf":
        return pdf_document(data, name=name)

    if suffix in MARKDOWN_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
        return RawDocument(
            source_type="markdown",
            source_id=name,
            title=title_from_markdown(text, Path(name).stem),
            path=name,
            text=text,
        )

    raise ValueError(
        f"{name}: unsupported file type {suffix or '(none)'} — "
        f"expected one of {', '.join(sorted(SUPPORTED))}"
    )
