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


def test_markdown_h1_extraction_is_unchanged_by_the_html_support():
    # The first `#` still wins over later ones, `##` is still not an H1, and
    # leading/trailing space around the text is still dropped.
    assert title_from_markdown("# First\n\n# Second\n", "readme") == "First"
    assert title_from_markdown("## Subheading\n\nbody", "deploy-guide") == "deploy guide"
    assert title_from_markdown("#   Padded   \n", "readme") == "Padded"
    assert title_from_markdown("intro\n\n# Later Heading\n", "readme") == "Later Heading"
    # No heading and nothing readable in the name: the raw fallback comes back.
    assert title_from_markdown("body", "-") == "-"


def test_title_comes_from_an_html_h1_with_attributes():
    text = '<h1 align="center">Hi 👋 I\'m Erik Cupsa</h1>\n\nbody'
    assert title_from_markdown(text, "README.md") == "Hi 👋 I'm Erik Cupsa"


def test_html_h1_attributes_may_be_unquoted_single_quoted_or_absent():
    assert title_from_markdown("<h1 align=center>Centered</h1>", "readme") == "Centered"
    assert title_from_markdown("<h1 id='top'>Anchored</h1>", "readme") == "Anchored"
    assert title_from_markdown("<H1>Shouted</H1>", "readme") == "Shouted"


def test_html_h1_may_span_lines_in_the_tag_and_in_its_contents():
    text = '<h1\n  align="center"\n  id="top"\n>\n  Enterprise\n  RAG\n</h1>\n'
    assert title_from_markdown(text, "readme") == "Enterprise RAG"


def test_inline_markup_inside_an_html_h1_is_stripped_to_plain_text():
    text = (
        '<h1 align="center">'
        '<img src="wave.gif" width="30"> Hi, I\'m <a href="https://x.dev">Erik</a>'
        "<br><strong>Software</strong> <span><em>Engineer</em></span>"
        "</h1>"
    )
    assert title_from_markdown(text, "readme") == "Hi, I'm Erik Software Engineer"


def test_stripping_inline_markup_does_not_split_words_from_punctuation():
    text = "<h1><strong>Savy</strong>, an AI tool</h1>"
    assert title_from_markdown(text, "readme") == "Savy, an AI tool"


def test_html_entities_in_a_heading_are_unescaped():
    text = "<h1>Search &amp; Retrieval &#39;25 &lt;beta&gt;</h1>"
    assert title_from_markdown(text, "readme") == "Search & Retrieval '25 <beta>"


def test_non_ascii_headings_survive_intact():
    assert title_from_markdown("<h1>คู่มือการติดตั้ง 🇹🇭</h1>", "readme") == "คู่มือการติดตั้ง 🇹🇭"
    assert title_from_markdown("# คู่มือการติดตั้ง", "readme") == "คู่มือการติดตั้ง"


def test_a_heading_with_nothing_but_badges_falls_back_to_the_filename():
    text = '<h1 align="center"><img src="banner.png"></h1>\n\nbody'
    assert title_from_markdown(text, "README.md") == "README.md"


def test_a_heading_that_strips_to_punctuation_or_space_falls_back():
    assert title_from_markdown("<h1> </h1>", "deploy-guide") == "deploy guide"
    assert title_from_markdown("<h1>---</h1>", "deploy-guide") == "deploy guide"
    assert title_from_markdown("<h1><!-- todo --></h1>", "deploy-guide") == "deploy guide"


def test_an_unusable_first_heading_defers_to_the_next_one():
    # The logo-then-name opening, which is why skipping beats falling straight
    # back to the filename.
    text = '<h1 align="center"><img src="logo.png"></h1>\n<h1 align="center">Savy</h1>'
    assert title_from_markdown(text, "README.md") == "Savy"


def test_the_first_heading_of_either_kind_in_document_order_wins():
    html_first = '<h1 align="center">Erik Cupsa</h1>\n\n# Installation\n'
    assert title_from_markdown(html_first, "README.md") == "Erik Cupsa"

    markdown_first = "# Installation\n\n<h1 align=\"center\">Erik Cupsa</h1>\n"
    assert title_from_markdown(markdown_first, "README.md") == "Installation"


def test_an_unclosed_html_h1_is_not_treated_as_a_heading():
    assert title_from_markdown('<h1 align="center">Erik', "deploy-guide") == "deploy guide"


def test_a_heading_inside_a_fenced_block_is_not_the_title():
    text = "Usage:\n\n```bash\n# Install the thing\n```\n\n# Real Heading\n"
    assert title_from_markdown(text, "README.md") == "Real Heading"


def test_a_document_whose_only_heading_is_fenced_falls_back_to_the_filename():
    text = "Usage:\n\n```sh\n# Install the thing\n```\n\nbody\n"
    assert title_from_markdown(text, "deploy-guide") == "deploy guide"


def test_tilde_fences_hide_headings_too():
    text = "~~~\n# Not A Heading\n~~~\n\n# Real Heading\n"
    assert title_from_markdown(text, "readme") == "Real Heading"


def test_a_fence_is_closed_only_by_its_own_marker():
    # The ``` inside a ~~~ block is content, so the `#` after it stays hidden.
    text = "~~~\n```\n# Still Fenced\n~~~\n\n# Real Heading\n"
    assert title_from_markdown(text, "readme") == "Real Heading"


def test_indented_fence_markers_are_recognised():
    text = "  ```python\n  # not a heading\n  ```\n\n# Real Heading\n"
    assert title_from_markdown(text, "readme") == "Real Heading"


def test_an_html_heading_inside_a_fenced_block_is_not_the_title():
    text = 'Example:\n\n```html\n<h1 align="center">Demo</h1>\n```\n\n# Real Heading\n'
    assert title_from_markdown(text, "README.md") == "Real Heading"


def test_an_unterminated_fence_hides_every_heading_after_it():
    text = "```\n# Never Closed\n\n# Also Swallowed\n"
    assert title_from_markdown(text, "deploy-guide") == "deploy guide"
    # A real heading *before* the stray fence is still found.
    assert title_from_markdown(f"# Top\n\n{text}", "readme") == "Top"


def test_a_fence_like_run_inside_a_heading_does_not_open_a_fence():
    # ``` mid-line is not a fence marker, so the heading is used as-is and the
    # `#` below it stays visible.
    assert title_from_markdown("<h1>Use ``` for code</h1>", "readme") == "Use ``` for code"
    assert title_from_markdown("# Use ``` for code\n", "readme") == "Use ``` for code"


def test_dir_connector_titles_a_readme_by_its_html_heading(tmp_path: Path):
    write(tmp_path, "README.md", '<h1 align="center">Hi 👋 I\'m Erik Cupsa</h1>\n\nbody\n')
    doc = next(iter(DirConnector(tmp_path).load()))

    assert doc.title == "Hi 👋 I'm Erik Cupsa"


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
