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
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.search import hybrid_search
from app.schemas import Source

GOLDEN_PATH = Path(__file__).with_name("golden.json")

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


def load_golden(path: Path = GOLDEN_PATH) -> list[Question]:
    raw = json.loads(path.read_text())
    return [Question(**item) for item in raw]


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


def as_markdown(results: dict[str, Score], total: int) -> str:
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
    return "\n".join(lines)


def validate(questions: list[Question], corpus: dict[str, str]) -> list[str]:
    """Catch expectations that no document can satisfy.

    A golden set is only as good as its answers, and a typo in one of them
    shows up as a retrieval failure rather than as a broken test.
    """
    problems = []
    for q in questions:
        if q.path not in corpus:
            problems.append(f"{q.question!r}: no document at {q.path!r}")
        elif q.contains not in corpus[q.path]:
            problems.append(f"{q.question!r}: {q.contains!r} is not in {q.path}")
    return problems
