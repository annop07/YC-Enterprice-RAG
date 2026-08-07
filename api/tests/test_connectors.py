from __future__ import annotations

from pathlib import Path

from app.ingest.connectors import DirConnector, RawDocument, title_from_markdown


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_title_comes_from_the_first_h1_then_falls_back_to_the_filename():
    assert title_from_markdown("# Deployment Guide\n\nbody", "deploy") == "Deployment Guide"
    assert title_from_markdown("no heading here", "deploy-guide") == "deploy guide"


def test_dir_connector_walks_markdown_and_skips_vendored_trees(tmp_path: Path):
    write(tmp_path, "docs/setup.md", "# Setup\n\nbody\n")
    write(tmp_path, "docs/nested/deep.md", "# Deep\n\nbody\n")
    write(tmp_path, "notes.txt", "not markdown")
    write(tmp_path, "node_modules/pkg/README.md", "# Dependency\n\nbody\n")
    write(tmp_path, ".git/hooks/README.md", "# Git\n\nbody\n")

    docs = list(DirConnector(tmp_path).load())
    paths = sorted(d.path for d in docs)

    assert paths == ["docs/nested/deep.md", "docs/setup.md"]


def test_source_id_is_relative_so_the_same_corpus_updates_in_place(tmp_path: Path):
    write(tmp_path, "docs/setup.md", "# Setup\n\nbody\n")
    doc = next(iter(DirConnector(tmp_path).load()))

    assert doc.source_id == "docs/setup.md"
    assert doc.title == "Setup"
    assert doc.source_type == "markdown"


def test_blank_files_are_skipped(tmp_path: Path):
    write(tmp_path, "empty.md", "\n\n   \n")
    assert list(DirConnector(tmp_path).load()) == []


def test_document_id_is_stable_and_hash_tracks_content():
    a = RawDocument("markdown", "docs/x.md", "X", "docs/x.md", "one")
    b = RawDocument("markdown", "docs/x.md", "X renamed", "docs/x.md", "one")
    c = RawDocument("markdown", "docs/x.md", "X", "docs/x.md", "two")

    # Same source, so the same row — even though the title changed.
    assert a.id == b.id == c.id
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash
