"""Embedder tests.

These load the real model, which is the point: the bugs worth catching here
are in how fastembed's tokenizer is configured, and a fake cannot have them.
"""
from __future__ import annotations

from tokenizers import Tokenizer, models

from app.ingest.embedder import Embedder, _unknown_token_id

MODEL = "BAAI/bge-small-en-v1.5"

THAI_QUESTION = "ใครเป็นคนดูแลระบบนี้"
OTHER_THAI_QUESTION = "ขั้นตอนการติดตั้งมีอะไรบ้าง"


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


def test_thai_really_does_collapse_to_one_vector_on_this_model():
    """The reason `reads` exists — asserted rather than asserted-about.

    Without this the tests below could pass while the premise was false.
    """
    embedder = Embedder(MODEL)
    a, b = (embedder.embed_query(q) for q in (THAI_QUESTION, OTHER_THAI_QUESTION))

    dot = sum(x * y for x, y in zip(a, b))
    norms = sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
    assert dot / norms > 0.9999, "two unrelated Thai questions should be identical here"


def test_reads_separates_a_blind_question_from_a_legible_one():
    embedder = Embedder(MODEL)

    assert embedder.reads("how does chunking work") is True
    assert embedder.reads(THAI_QUESTION) is False
    # One English term is enough: the vector leg has something to work with.
    assert embedder.reads("ตั้งค่า chunk overlap ยังไง") is True
    assert embedder.reads("hnsw.ef_search คืออะไร") is True


def test_punctuation_and_digits_do_not_count_as_reading_the_question():
    """Both survive an English vocabulary while carrying none of the question.

    Measured: three unrelated Thai questions ending in "?" all tokenise to
    `[UNK] ?` and embed at pairwise cosine 1.000000, and so do two unrelated
    Thai questions sharing the number 100. A naive "any non-unknown token"
    check is defeated by a single question mark.
    """
    embedder = Embedder(MODEL)

    assert embedder.reads(THAI_QUESTION + "?") is False
    assert embedder.reads("ระบบนี้รองรับเอกสาร 100 ไฟล์ไหม") is False
    # Nothing was unknown here, so nothing was lost — this stays on the normal
    # path and the re-ranker gets to call it irrelevant, as it always has.
    assert embedder.reads("???") is True
    assert embedder.reads("5433") is True


def test_the_unknown_token_is_discovered_rather_than_hardcoded():
    """`[UNK]` is a BERT spelling. e5 and the SentencePiece family use `<unk>`."""
    embedder = Embedder(MODEL)
    assert embedder._unk_id == embedder._counter.token_to_id("[UNK]")

    angle = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1}, unk_token="<unk>"))
    assert _unknown_token_id(angle) == 0


def test_a_tokenizer_with_no_unknown_token_fails_open():
    """A model we cannot inspect must not silently lose half of retrieval."""
    no_unk = Tokenizer(models.WordLevel({"hello": 0, "world": 1}))
    assert _unknown_token_id(no_unk) is None

    embedder = Embedder(MODEL)
    embedder._unk_id = None
    assert embedder.reads(THAI_QUESTION) is True
