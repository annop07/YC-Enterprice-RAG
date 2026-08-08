"""Retrieval evaluation.

The point of this file is that every retrieval claim in the README is a number
someone can reproduce. It runs the same question set through four
configurations — vector only, keyword only, both fused, both fused then
re-ranked — and reports Recall@k and MRR for each.

The four runs differ only in which legs are given a non-zero limit and whether
the re-ranker is applied. Everything else, including the SQL, is identical, so
the differences between the columns are the thing being measured and not an
artefact of running four different searches.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.search import hybrid_search
from app.schemas import Source

GOLDEN_PATH = Path(__file__).with_name("golden.json")
READABILITY_PATH = Path(__file__).with_name("readability.json")

CONFIGS: list[tuple[str, dict]] = [
    ("vector only", {"legs": "vector", "rerank": False}),
    ("keyword only", {"legs": "keyword", "rerank": False}),
    ("hybrid (RRF)", {"legs": "hybrid", "rerank": False}),
    ("hybrid + rerank", {"legs": "hybrid", "rerank": True}),
]


@dataclass(frozen=True)
class Question:
    question: str
    path: str
    #: A distinctive phrase from the passage that should answer it. Matching on
    #: this rather than on the document alone means a hit is the right
    #: *passage*, not merely the right file — a five-chunk document would
    #: otherwise score a hit for retrieving any of it.
    contains: str


@dataclass
class Score:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    misses: list[str]


@dataclass(frozen=True)
class ReadabilityCase:
    """A question and whether retrieval is expected to be able to read it.

    Recall says nothing about this axis. Every one of the 30 golden questions is
    English, so the whole table can read 1.00 while questions in another script
    return the same arbitrary chunks at 0.99 — which is exactly what it did.
    """

    question: str
    unreadable: bool
    #: Why this case is in the set. Kept in the data because the interesting
    #: cases are the ones that look like the opposite class: a Thai question
    #: about something the corpus documents, and an English question about
    #: something it does not.
    why: str


@dataclass
class ReadabilityScore:
    total: int
    correct: int
    #: `(question, expected_unreadable)` for each case that came out the other
    #: way. Listed rather than counted: which side it failed on says whether
    #: retrieval went blind or went over-cautious, and those are opposite bugs.
    mismatches: list[tuple[str, bool]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def load_golden(path: Path = GOLDEN_PATH) -> list[Question]:
    raw = json.loads(path.read_text())
    return [Question(**item) for item in raw]


def load_readability(path: Path = READABILITY_PATH) -> list[ReadabilityCase]:
    raw = json.loads(path.read_text())
    return [ReadabilityCase(**item) for item in raw]


async def score_readability(cases: list[ReadabilityCase], k: int = 5) -> ReadabilityScore:
    """Whether retrieval classifies each question the way the set says it should.

    Scored through `hybrid_search` rather than `Embedder.reads` directly, so it
    measures what a caller actually receives — a unit test on the tokenizer
    passes even if nothing is wired to its answer.
    """
    mismatches: list[tuple[str, bool]] = []
    for case in cases:
        result = await hybrid_search(case.question, top_k=k)
        if result.unreadable_query is not case.unreadable:
            mismatches.append((case.question, case.unreadable))

    return ReadabilityScore(
        total=len(cases),
        correct=len(cases) - len(mismatches),
        mismatches=mismatches,
    )


def first_relevant_rank(question: Question, sources: list[Source]) -> int | None:
    """1-based rank of the first source that answers the question, if any."""
    for source in sources:
        if source.path == question.path and question.contains in source.snippet:
            return source.n
    return None


async def score_config(questions: list[Question], k: int, **config) -> Score:
    ranks: list[int | None] = []
    for question in questions:
        result = await hybrid_search(question.question, top_k=k, **config)
        ranks.append(first_relevant_rank(question, result.sources))

    def recall(at: int) -> float:
        hits = sum(1 for r in ranks if r is not None and r <= at)
        return hits / len(ranks) if ranks else 0.0

    reciprocal = [1.0 / r if r else 0.0 for r in ranks]
    return Score(
        recall_at_1=recall(1),
        recall_at_3=recall(3),
        recall_at_5=recall(5),
        mrr=sum(reciprocal) / len(reciprocal) if reciprocal else 0.0,
        misses=[q.question for q, r in zip(questions, ranks) if r is None],
    )


async def run(questions: list[Question], k: int = 5) -> dict[str, Score]:
    return {name: await score_config(questions, k, **config) for name, config in CONFIGS}


def as_markdown(
    results: dict[str, Score],
    total: int,
    readability: ReadabilityScore | None = None,
) -> str:
    lines = [
        f"| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR |",
        f"| --- | --- | --- | --- | --- |",
    ]
    for name, score in results.items():
        lines.append(
            f"| {name} | {score.recall_at_1:.2f} | {score.recall_at_3:.2f} "
            f"| {score.recall_at_5:.2f} | {score.mrr:.3f} |"
        )
    lines.append("")
    lines.append(f"{total} questions.")

    if readability is not None:
        lines.append("")
        # Reported next to recall rather than in a separate document because the
        # two answer different halves of the same question, and recall alone
        # reads as a full account of retrieval when it is not.
        lines.append(
            f"Query readability: **{readability.correct}/{readability.total}** "
            f"classified correctly. A question the embedding model cannot read "
            f"disables the vector leg and the re-ranker instead of returning "
            f"their output, so \"could not look\" is never reported as "
            f"\"looked and found nothing\"."
        )
        for question, expected in readability.mismatches:
            wanted = "unreadable" if expected else "readable"
            lines.append(f"- not classified as {wanted}: {question!r}")

    return "\n".join(lines)


def validate(questions: list[Question], corpus: Mapping[str, Sequence[str]]) -> list[str]:
    """Catch expectations that no document can satisfy.

    A golden set is only as good as its answers, and a typo in one of them
    shows up as a retrieval failure rather than as a broken test.

    A path maps to *every* document indexed under it, not one. Five of the
    repositories in the development corpus each contribute a `README.md`, and
    keying by path alone silently checked whichever of them was written last —
    so an expectation could be validated against a document retrieval would
    never match it to, and the count printed with the table undercounted the
    corpus by four documents.
    """
    problems = []
    for q in questions:
        texts = corpus.get(q.path)
        if not texts:
            problems.append(f"{q.question!r}: no document at {q.path!r}")
        elif not any(q.contains in text for text in texts):
            where = q.path if len(texts) == 1 else f"any of the {len(texts)} at {q.path}"
            problems.append(f"{q.question!r}: {q.contains!r} is not in {where}")
    return problems
