"""Prompt construction and the citation guard."""
from __future__ import annotations

import re

from app.schemas import Source

SYSTEM = """You answer questions about an organisation's internal documents.

Rules:
- Use only the numbered context below. Never use outside knowledge, and never \
fill a gap in the context with something that sounds plausible.
- Mark every factual claim with the number of the context block it came from, \
like [2]. A claim with no number is treated as unsupported.
- If the context does not answer the question, say so plainly and stop. Do not \
guess, and do not pad the answer with what the context does happen to contain.
- Be concise, and prefer the documents' own wording over paraphrase."""

REWRITE_SYSTEM = """Rewrite the user's latest question so it stands on its own \
without the conversation.

Resolve pronouns and references to earlier turns ("that", "it", "the second \
one") into the thing they refer to. Keep the user's wording and language \
otherwise. If the question already stands alone, repeat it unchanged.

Reply with the rewritten question and nothing else."""

CITATION = re.compile(r"\[(\d+)\]")
#: A dropped citation leaves a gap before the punctuation that followed it.
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?)])")
_REPEATED_SPACE = re.compile(r"[ \t]{2,}")


def locator_label(source: Source) -> str:
    if source.locator.page is not None:
        return f"{source.path} p.{source.locator.page}"
    if source.locator.line_start is not None:
        return f"{source.path}:{source.locator.line_start}-{source.locator.line_end}"
    return source.path


def build_context(sources: list[Source]) -> str:
    """Numbered blocks, in the order the model is asked to cite them by."""
    blocks = []
    for source in sources:
        heading = f" · {source.heading_path}" if source.heading_path else ""
        blocks.append(
            f"[{source.n}] {source.title} — {locator_label(source)}{heading}\n"
            f"{source.snippet}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, sources: list[Source]) -> list[dict]:
    if not sources:
        context = "(nothing was retrieved for this question)"
    else:
        context = build_context(sources)

    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {question}",
        },
    ]


def build_rewrite_messages(question: str, history: list[tuple[str, str]]) -> list[dict]:
    """`history` is (role, content), oldest first, already trimmed."""
    turns = "\n".join(f"{role}: {content}" for role, content in history)
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Conversation so far:\n{turns}\n\nLatest question: {question}"},
    ]


def strip_unsupported_citations(text: str, valid: set[int]) -> tuple[str, int]:
    """Remove citation markers that point at nothing the model was shown.

    The model is asked to cite by number; nothing stops it inventing one. A
    marker with no matching source is not a formatting slip, it is a claim
    presented as sourced when it is not — so it is removed, and the count is
    reported rather than swallowed.

    The answer streams to the client unmodified, because the tokens are already
    on their way by the time the whole answer exists. What this produces is the
    version that gets stored, and the count that gets shown under the answer.
    """
    dropped = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal dropped
        if int(match.group(1)) in valid:
            return match.group(0)
        dropped += 1
        return ""

    cleaned = CITATION.sub(replace, text)
    if dropped:
        cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
        cleaned = _REPEATED_SPACE.sub(" ", cleaned)
    return cleaned, dropped


def cited_numbers(text: str) -> set[int]:
    return {int(m.group(1)) for m in CITATION.finditer(text)}
