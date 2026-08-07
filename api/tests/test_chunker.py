"""Chunker tests.

Token counting is faked with a word counter so these run without loading the
ONNX model — the chunker takes the counter as a parameter precisely so this is
possible. What is being tested is the *packing*, and above all that the line
ranges it reports are true, because a wrong line range is a citation that
points at the wrong text while looking perfectly correct.
"""
from __future__ import annotations

import re
from typing import Sequence

from app.ingest.chunker import build_embed_input, chunk_markdown, heading_paths

WORD = re.compile(r"\S+")


def count_words(texts: Sequence[str]) -> list[int]:
    return [len(WORD.findall(t)) for t in texts]


def word_offsets(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in WORD.finditer(text)]


def chunk(text: str, *, max_tokens: int = 20, overlap: int = 5):
    return chunk_markdown(
        text,
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        count_tokens=count_words,
        token_offsets=word_offsets,
    )


DOC = """# Handbook

Intro line one with several words in it.
Intro line two with several words in it.

## Setup

Setup line one with several words in it.
Setup line two with several words in it.
Setup line three with several words in it.

### Docker

Docker line one with several words in it.
Docker line two with several words in it.

## Support

Support line one with several words in it.
"""


def test_heading_paths_track_nesting():
    paths = heading_paths(DOC.split("\n"))
    lines = DOC.split("\n")

    assert paths[lines.index("# Handbook")] == "Handbook"
    assert paths[lines.index("## Setup")] == "Handbook > Setup"
    assert paths[lines.index("### Docker")] == "Handbook > Setup > Docker"
    # "## Support" pops both Setup and Docker off the stack.
    assert paths[lines.index("## Support")] == "Handbook > Support"


def test_heading_paths_ignore_headings_inside_code_fences():
    text = "# Real\n\n```bash\n# not a heading\n```\n\nbody\n"
    paths = heading_paths(text.split("\n"))
    assert paths[-2] == "Real"


def test_line_ranges_match_the_source_exactly():
    """The property the whole citation feature rests on."""
    lines = DOC.split("\n")
    for c in chunk(DOC):
        assert c.content == "\n".join(lines[c.line_start - 1 : c.line_end])


def test_ranges_do_not_claim_blank_lines_around_the_content():
    """A range one line wider than the text highlights the wrong first line."""
    for c in chunk(DOC):
        assert c.content.split("\n")[0].strip()
        assert c.content.split("\n")[-1].strip()


def test_chunks_respect_the_token_budget():
    for c in chunk(DOC, max_tokens=20):
        assert c.token_count <= 20


def test_consecutive_chunks_overlap():
    chunks = chunk(DOC, max_tokens=20, overlap=10)
    assert len(chunks) > 1
    for previous, nxt in zip(chunks, chunks[1:]):
        assert nxt.line_start <= previous.line_end, "no overlap between chunks"


def test_one_line_is_carried_over_even_when_it_exceeds_the_overlap_budget():
    """Lines here are ~8 tokens, so a 5-token budget fits none of them whole."""
    chunks = chunk(DOC, max_tokens=20, overlap=5)
    for previous, nxt in zip(chunks, chunks[1:]):
        assert nxt.line_start <= previous.line_end


def test_ordinals_are_sequential():
    chunks = chunk(DOC)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_every_chunk_carries_its_heading():
    for c in chunk(DOC):
        assert c.heading_path is not None
        assert c.heading_path.startswith("Handbook")


def test_a_line_longer_than_the_budget_is_split_on_token_boundaries():
    long_line = " ".join(f"w{i}" for i in range(50))
    chunks = chunk(f"# T\n\n{long_line}\n", max_tokens=10, overlap=0)

    pieces = [c for c in chunks if c.content.startswith("w")]
    assert len(pieces) >= 5
    for c in pieces:
        assert c.token_count <= 10
        # Split inside one line, so both ends still name that same line.
        assert c.line_start == c.line_end == 3
    # Nothing is lost in the split.
    assert " ".join(p.content for p in pieces) == long_line


def test_short_lines_and_blank_lines_do_not_explode_the_chunk_count():
    """Each chunk must advance, even where the overlap budget could eat the window.

    Blank lines cost zero tokens, so an overlap budget measured only in tokens
    lets the walk-back cross any number of them for free — in a document of
    short lines it can retreat most of the way to the start and advance a line
    or two per chunk. The floor in the walk-back is what bounds it.
    """
    # 300 lines, ~2 tokens each, blank line after every one — 600 tokens total.
    text = "# Reference\n\n" + "\n\n".join(f"- item {i}" for i in range(300))
    chunks = chunk(text, max_tokens=100, overlap=20)

    total_tokens = sum(count_words([text]))
    ideal = total_tokens / 100
    assert len(chunks) <= ideal * 2, (
        f"{len(chunks)} chunks for ~{ideal:.0f} chunks' worth of text"
    )


def test_every_document_terminates_with_progress_on_each_chunk():
    """Each chunk must start strictly after the previous one started."""
    text = "\n\n".join(["x"] * 200)
    chunks = chunk(text, max_tokens=10, overlap=9)
    starts = [c.line_start for c in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_empty_and_blank_documents_produce_nothing():
    assert chunk("") == []
    assert chunk("\n\n   \n") == []


def test_embed_input_carries_the_breadcrumb_but_content_does_not():
    text = build_embed_input("Handbook", "Handbook > Setup", "body text")
    assert text.endswith("body text")
    assert build_embed_input("Handbook", None, "body").startswith("Handbook\n\n")


def test_embed_input_does_not_repeat_a_title_the_breadcrumb_already_starts_with():
    """The H1 usually is the title; doubling it burns tokens for nothing."""
    assert build_embed_input("Handbook", "Handbook > Setup", "b").startswith(
        "Handbook > Setup\n\n"
    )
    assert build_embed_input("Handbook", "Handbook", "b").startswith("Handbook\n\n")
    # A different heading tree still gets the title in front of it.
    assert build_embed_input("Handbook", "Appendix > A", "b").startswith(
        "Handbook > Appendix > A\n\n"
    )
