"""Embedder tests.

These load the real model, which is the point: the bugs worth catching here
are in how fastembed's tokenizer is configured, and a fake cannot have them.
"""
from __future__ import annotations

from app.ingest.embedder import Embedder

MODEL = "BAAI/bge-small-en-v1.5"


def test_batch_counting_is_not_padded_to_the_longest_item():
    """Regression: every line in a document reported the same token count.

    fastembed configures its tokenizer for inference, with padding on. That
    makes `encode_batch` pad every sequence to the longest one in the batch, so
    a one-word line counted the same as a full paragraph and the chunker packed
    by a constant. It fails silently — the numbers look plausible, just wrong.
    """
    embedder = Embedder(MODEL)
    counts = embedder.count_tokens(["hi", "one two three four five six seven", ""])

    assert counts[2] == 0, "an empty string is zero tokens, not padding"
    assert counts[0] < counts[1]
    # Counting one at a time must agree with counting as a batch.
    assert counts == [embedder.count_tokens([t])[0] for t in ["hi", "one two three four five six seven", ""]]


def test_counting_is_not_truncated_at_the_model_ceiling():
    embedder = Embedder(MODEL)
    long_text = " ".join(["word"] * 2000)
    assert embedder.count_tokens([long_text])[0] > embedder.max_input_tokens


def test_token_offsets_cover_the_text_for_splitting_long_lines():
    embedder = Embedder(MODEL)
    text = "alpha beta gamma delta"
    offsets = embedder.token_offsets(text)

    assert offsets[0][0] == 0
    assert offsets[-1][1] == len(text)
    assert len(offsets) == embedder.count_tokens([text])[0]


def test_embeddings_have_the_model_dimension_and_queries_embed_the_same_way():
    embedder = Embedder(MODEL)
    passages = embedder.embed_passages(["first passage", "second passage"])
    query = embedder.embed_query("a question")

    assert len(passages) == 2
    assert len(passages[0]) == len(query) == 384
