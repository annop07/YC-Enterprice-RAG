"""In-memory documents, for the upload endpoint.

A connector reads from somewhere; an upload arrives already read. This is the
one place that turns bytes plus a filename into the same `RawDocument` the
connectors produce, so nothing downstream has to know which door it came in by.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Sequence

from app.ingest.connectors import MARKDOWN_SUFFIXES, RawDocument, title_from_markdown
from app.ingest.pdf import pdf_document

SUPPORTED = MARKDOWN_SUFFIXES | {".pdf"}


def upload_name(filename: str) -> str:
    """The name an uploaded file is filed under.

    Only the last segment survives, and it survives as data: this string is
    chosen by whoever made the request, and it ends up as the document's
    identity, as the `path` shown on every citation card, and as the label in
    the corpus panel. Backslashes are folded first so a Windows client's
    `C:\docs\notes.md` loses its directories the same way a POSIX path does,
    and a name that reduces to nothing — `..`, `/`, spaces — becomes
    `upload` rather than the empty string, which would file every such file
    under the same blank identity.
    """
    name = PurePosixPath(filename.replace("\\", "/")).name.strip()
    # `PurePosixPath("..").name` is `".."`, not the empty string — the relative
    # markers are ordinary components to it. Left through, a file called `..`
    # would be filed under that name, which is neither a name nor a path.
    return "upload" if name in {"", ".", ".."} else name


def duplicate_names(filenames: Sequence[str]) -> list[str]:
    """Names that more than one file in the same request would be filed under.

    A document's identity is `source_type:source_id`, and for an upload the
    `source_id` is its name — so two files called `README.md` are one document
    to this system, and ingesting both in one request means the second silently
    replaces the first. The report still said two documents written, because
    two were, and the corpus was left holding one of them with no indication
    that the other had ever arrived.

    Two files with the same name cannot be selected in one file picker, so
    this is always a mistake worth refusing rather than a preference worth
    guessing at. Across *separate* requests the same name is deliberate — that
    is how re-uploading an edited file updates it in place — and is reported as
    `updated`, not swallowed.
    """
    counts = Counter(upload_name(f) for f in filenames)
    return sorted(name for name, n in counts.items() if n > 1)


def document_from_upload(filename: str, data: bytes) -> RawDocument:
    name = upload_name(filename)
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
