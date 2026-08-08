"""Local embeddings via fastembed, plus the tokenizer the chunker budgets with."""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from typing import Sequence

from fastembed import TextEmbedding
from tokenizers import Tokenizer

from app.config import get_settings


def _unknown_token_id(tokenizer: Tokenizer) -> int | None:
    """The id this tokenizer emits for text it has no vocabulary for.

    Read off the tokenizer rather than hardcoded. BERT-family vocabularies
    spell it `[UNK]`, the SentencePiece family spells it `<unk>`, and Unigram
    stores an index into its own vocabulary instead of a name — so `EMBED_MODEL`
    could be changed to a perfectly good multilingual model and a hardcoded
    `[UNK]` would quietly stop matching anything.

    None means this tokenizer has no such concept, which `Embedder.reads`
    treats as fail-open.
    """
    spec = json.loads(tokenizer.to_str()).get("model", {})

    # WordPiece and BPE name it; Unigram numbers it.
    named = getattr(tokenizer.model, "unk_token", None) or spec.get("unk_token")
    if isinstance(named, str) and named:
        return tokenizer.token_to_id(named)

    index, vocab = spec.get("unk_id"), spec.get("vocab")
    if isinstance(index, int) and isinstance(vocab, list) and 0 <= index < len(vocab):
        entry = vocab[index]
        name = entry[0] if isinstance(entry, (list, tuple)) else entry
        if isinstance(name, str):
            return tokenizer.token_to_id(name)
    return None


def _has_letter(token: str) -> bool:
    """Letters only — see `Embedder.reads` for why digits do not count."""
    return any(unicodedata.category(ch).startswith("L") for ch in token)


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

        # Resolved once here rather than per query: it costs a serialise and a
        # parse of the whole vocabulary.
        self._unk_id = _unknown_token_id(self._counter)

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

    # --- legibility -------------------------------------------------------

    def reads(self, text: str) -> bool:
        """Does the tokenizer see real words here, or only unknowns?

        An English WordPiece vocabulary contains no Thai, so an all-Thai
        question collapses to a single unknown token — and *every* all-Thai
        question collapses to the same one. Measured on the default
        `BAAI/bge-small-en-v1.5`: unrelated Thai questions embed at pairwise
        cosine 1.000000. The vector leg there is not weak, it is blind, and a
        blind leg that still returns its nearest neighbours is worse than one
        that returns nothing — it returns the same arbitrary documents for
        every question, with no signal that anything went wrong.

        Only *letters* count as read. Punctuation and digits survive the
        vocabulary while carrying none of the question: measured, three
        unrelated Thai questions ending in "?" tokenise to `[UNK] ?` and embed
        at cosine 1.000000, as do two unrelated Thai questions sharing the
        number 100. Counting either would defeat this check on any Thai
        sentence with a question mark in it.

        This detects blindness, not degradation. A script the model happens to
        tokenise character by character reads as legible here and is merely
        poor — which is the right side to err on, since the alternative is
        disabling retrieval over a judgement call about quality.
        """
        if self._unk_id is None:
            return True  # fail-open: see `_unknown_token_id`

        # `add_special_tokens=False` is load-bearing: [CLS] and [SEP] are not
        # unknown tokens, so with them on every text on earth looks readable.
        encoding = self._counter.encode(text, add_special_tokens=False)

        # Nothing was unknown, so nothing was lost. This is also what keeps a
        # letterless-but-legible query like "5433" or "???" on the normal path.
        if self._unk_id not in encoding.ids:
            return True

        return any(
            _has_letter(token)
            for token, token_id in zip(encoding.tokens, encoding.ids)
            if token_id != self._unk_id
        )

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
