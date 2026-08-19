"""What an upload is allowed to cost before anything looks at it.

`/ingest/files` used to call `await upload.read()` with no argument, which
materialises the whole file — any file, at any size — in this process before
there is anything to check. The limit only means something if it is enforced on
the way in, which is what these pin.
"""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException, UploadFile

from app.main import _UPLOAD_CHUNK, _read_capped

pytestmark = pytest.mark.asyncio


def upload(data: bytes, name: str = "handbook.md") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=name, size=len(data))


async def test_a_file_within_the_budget_arrives_whole():
    data = b"# Handbook\n\n" + b"x" * 5_000
    assert await _read_capped(upload(data), "handbook.md", 1_000_000) == data


async def test_a_file_over_the_budget_is_refused_with_413():
    with pytest.raises(HTTPException) as raised:
        await _read_capped(upload(b"y" * 4_000), "big.pdf", 1_000)

    assert raised.value.status_code == 413
    assert "big.pdf" in raised.value.detail
    # Both ceilings are named, because "too large" without a number is a bug
    # report the user cannot act on.
    assert "MAX_UPLOAD_BYTES" in raised.value.detail
    assert "MAX_UPLOAD_TOTAL_BYTES" in raised.value.detail


async def test_the_refusal_happens_before_the_whole_file_is_in_memory():
    """The point of the chunked read: a 2 GB upload must not be 2 GB of this
    process's memory first and a 413 second."""
    source = io.BytesIO(b"z" * (8 * _UPLOAD_CHUNK))
    too_big = UploadFile(file=source, filename="huge.pdf")

    with pytest.raises(HTTPException):
        await _read_capped(too_big, "huge.pdf", _UPLOAD_CHUNK)

    assert source.tell() <= 2 * _UPLOAD_CHUNK, "read kept going past the limit"


async def test_the_budget_is_what_is_left_of_the_request_not_just_the_file():
    """`ingest_files` passes `min(per-file limit, what the request has left)`,
    so fifty files each under the per-file limit cannot add up to a gigabyte.
    """
    with pytest.raises(HTTPException) as raised:
        await _read_capped(upload(b"a" * 900), "last.md", 500)
    assert raised.value.status_code == 413


async def test_an_empty_upload_is_not_an_error_here():
    """It is rejected later, by the connector, with a message about the file
    type — not here with one about size."""
    assert await _read_capped(upload(b""), "empty.md", 1_000) == b""
