"""Local embeddings via fastembed, plus the tokenizer the chunker budgets with."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from fastembed import TextEmbedding
from tokenizers import Tokenizer

from app.config import get_settings


class Embedder:
    """One loaded model, used for both counting and embedding.

    Counting and embedding must agree: budgeting a chunk with a different
    tokenizer than the one that will encode it is how text gets silently
    truncated at the 512-token ceiling.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name)

        shared = self._model.model.tokenizer
        # fastembed configures this tokenizer for *inference*: truncation on,
        # padding on. Both are wrong for counting, and both fail silently.
        #
        #   truncation -> anything longer reports 512, hiding the very overflow
        #                 this is meant to detect.
        #   padding    -> encode_batch pads every sequence to the longest one in
        #                 the batch, so a one-word line and a paragraph report
        #                 the same count. Line-by-line budgeting then works off
        #                 a constant and the chunker packs by the wrong number.
        #
        # An independent copy with both switched off counts honestly and leaves
        # the embedding path untouched.
        self._counter = Tokenizer.from_str(shared.to_str())
        self._counter.no_truncation()
        self._counter.no_padding()

        truncation = shared.truncation or {}
        self.max_input_tokens: int = int(truncation.get("max_length", 512))

        # e5 was trained with asymmetric prefixes and loses a lot of retrieval
        # quality without them. bge and the rest do not want them.
        self._needs_e5_prefix = "e5" in model_name.lower()

    # --- counting ---------------------------------------------------------

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encodings = self._counter.encode_batch(list(texts), add_special_tokens=False)
        return [len(e.ids) for e in encodings]

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return list(self._counter.encode(text, add_special_tokens=False).offsets)

    # --- embedding --------------------------------------------------------

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [f"passage: {t}" for t in texts] if self._needs_e5_prefix else list(texts)
        return [v.tolist() for v in self._model.embed(prepared)]

    def embed_query(self, text: str) -> list[float]:
        prepared = f"query: {text}" if self._needs_e5_prefix else text
        return next(iter(self._model.query_embed(prepared))).tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Loading the ONNX model takes seconds; do it once per process."""
    settings = get_settings()
    embedder = Embedder(settings.embed_model)

    probe = embedder.embed_passages(["dimension probe"])[0]
    if len(probe) != settings.embed_dim:
        raise RuntimeError(
            f"EMBED_DIM={settings.embed_dim} but {settings.embed_model} returns "
            f"{len(probe)} dimensions. The vector column is fixed at table "
            f"creation, so this is a re-index, not a config change."
        )
    return embedder
