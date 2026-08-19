"""Prompt construction and the citation guard."""
from __future__ import annotations

import re
from typing import Sequence

from app.schemas import Source

SYSTEM = """You answer questions about an organisation's internal documents.

Rules:
- Use only the numbered context in the latest message. Never use outside \
knowledge, and never fill a gap in the context with something that sounds \
plausible.
- Mark every factual claim with the number of the context block it came from, \
like [2]. A claim with no number is treated as unsupported.
- If the context does not answer the question, say so plainly and stop. Do not \
guess, and do not pad the answer with what the context does happen to contain.
- Be concise, and prefer the documents' own wording over paraphrase.
- Earlier turns are there so you can tell what the latest question refers to. \
They are not a source: nothing in them may be cited or restated as fact unless \
the numbered context in the latest message says it too."""

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


def strip_citations(text: str) -> str:
    """An earlier answer with its citation markers removed.

    `[2]` in the turn before last pointed at *that* turn's second chunk. This
    turn retrieved its own five, and its `[2]` is a different passage
    entirely. Left in the conversation, those numbers read to the model as
    citations it has already made and can refer back to — and the citation
    guard cannot help, because a re-used `[2]` is a valid number for this
    turn: it would be kept, and it would point at the wrong document.

    Removing them costs nothing. The history is here to resolve "the second
    one", not to be quoted.
    """
    cleaned = CITATION.sub("", text)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    return _REPEATED_SPACE.sub(" ", cleaned).strip()


#: Appended to the context when the re-ranker scored nothing as clearly
#: relevant. The system prompt already says not to answer from context that
#: does not answer the question; this says *now*, at the one moment the model
#: cannot tell on its own — the blocks below look exactly like good ones,
#: because they are the closest rows in the index either way.
LOW_CONFIDENCE_HINT = (
    "\n\n(Retrieval note: none of the blocks above was scored as clearly "
    "relevant to this question. They are the closest matches in the index, "
    "which is not the same as an answer. If they do not actually answer it, "
    "say that plainly instead of assembling something from them.)"
)


def build_messages(
    question: str,
    sources: list[Source],
    history: Sequence[tuple[str, str]] = (),
    *,
    low_confidence: bool = False,
) -> list[dict]:
    """The full request: system rules, the conversation, then context + question.

    `history` is the part that was missing. The rewriter was given the
    conversation so that "what about the second one?" could be turned into
    something retrievable — and then the model that had to *answer* it was
    handed the question exactly as typed, with no conversation at all. It
    could see five chunks about the second thing and a question that never
    says what the second thing is, so the good case was a lucky guess and the
    normal case was "could you clarify what you are asking about?" over a
    corpus that had already found the answer.

    Prior turns go in as real messages rather than pasted into the question,
    because that is the shape a chat model is trained on. The question itself
    stays as the user typed it: the rewritten form exists to search with, and
    putting words in the user's mouth here is how an answer ends up addressing
    something they did not ask.
    """
    if not sources:
        context = "(nothing was retrieved for this question)"
    else:
        context = build_context(sources)
        if low_confidence:
            context += LOW_CONFIDENCE_HINT

    messages: list[dict] = [{"role": "system", "content": SYSTEM}]
    for role, content in history:
        messages.append(
            {
                "role": role,
                "content": strip_citations(content) if role == "assistant" else content,
            }
        )
    messages.append(
        {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"}
    )
    return messages


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
