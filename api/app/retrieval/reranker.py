"""Cross-encoder re-ranking.

Fusion orders chunks by *where they appeared* in two independent rankings. A
cross-encoder reads the question and the chunk together and scores how well one
answers the other — slower per pair, but far closer to relevance, which is why
only the survivors of fusion are worth spending it on.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import get_settings


def relevance_probability(logit: float) -> float:
    """Squash a cross-encoder logit into 0..1.

    ms-marco cross-encoders emit unbounded logits — around -11 for an unrelated
    pair, comfortably positive for a good one. The sigmoid of that is the
    model's relevance probability, which is both the standard reading of the
    output and the only form the UI can draw a bar from.
    """
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    # exp(-logit) overflows for very negative logits; this branch is the same
    # function written so it cannot.
    exp = math.exp(logit)
    return exp / (1.0 + exp)


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return [relevance_probability(s) for s in self._model.rerank(query, list(documents))]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(get_settings().rerank_model)
