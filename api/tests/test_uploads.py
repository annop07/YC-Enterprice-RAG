"""Uploaded files, and the identity they are filed under.

A document is `source_type:source_id`, and for an upload the `source_id` is
its name — so two files called `README.md` are one document. Ingesting both in
one request wrote the first, then overwrote it with the second, and reported
"2 documents, 2 written" over a corpus that ended up holding one of them.
"""
from __future__ import annotations

import pytest

from app.ingest.uploads import document_from_upload, duplicate_names, upload_name

MARKDOWN = b"# Team A\n\nTeam A deploys on Fridays.\n"


# --- naming ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("sent", "filed_as"),
    [
        ("README.md", "README.md"),
        ("docs/README.md", "README.md"),
        ("../../etc/passwd.md", "passwd.md"),
        (r"C:\docs\notes.md", "notes.md"),
        ("/absolute/notes.md", "notes.md"),
        ("  spaced.md  ", "spaced.md"),
        # Reduces to nothing, and must not become the empty string — every
        # such file would then share one identity.
        ("..", "upload"),
        ("/", "upload"),
        ("", "upload"),
    ],
)
def test_a_filename_is_reduced_to_a_name(sent: str, filed_as: str) -> None:
    assert upload_name(sent) == filed_as


# --- collisions -----------------------------------------------------------


def test_two_files_with_the_same_name_are_reported() -> None:
    assert duplicate_names(["a/README.md", "b/README.md"]) == ["README.md"]


def test_every_colliding_name_is_reported_not_just_the_first() -> None:
    names = ["x/README.md", "y/README.md", "p/notes.md", "q/notes.md", "unique.md"]
    assert duplicate_names(names) == ["README.md", "notes.md"]


def test_distinct_names_do_not_collide() -> None:
    assert duplicate_names(["a.md", "b.md", "handbook.pdf"]) == []


def test_collision_is_judged_on_the_filed_name_not_the_string_sent() -> None:
    """The paths differ; what they are filed under does not."""
    assert duplicate_names(["docs/guide.md", "reference/guide.md"]) == ["guide.md"]


def test_one_file_is_never_a_collision_with_itself() -> None:
    assert duplicate_names(["README.md"]) == []


# --- the identity that made it a data-loss bug ----------------------------


def test_two_different_files_of_the_same_name_are_one_document() -> None:
    """The property the 409 exists to protect the user from.

    Not a bug in itself — it is what makes re-uploading an edited file update
    it in place rather than duplicate it. It only becomes data loss when both
    arrive in the same request, where neither can have been meant as an edit
    of the other.
    """
    a = document_from_upload("a/README.md", MARKDOWN)
    b = document_from_upload("b/README.md", b"# Team B\n\nTeam B never deploys.\n")
    assert a.id == b.id
    assert a.content_hash != b.content_hash


def test_an_unsupported_type_is_still_refused_by_name() -> None:
    with pytest.raises(ValueError):
        document_from_upload("notes.txt", b"plain")
