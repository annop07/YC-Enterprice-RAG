"""Indexing as a job, and what that buys.

Ingestion used to run inside the request that asked for it. The request could
only say "working" until it was over, a proxy was free to time it out and
leave the work running with nobody holding the result, and one unreadable file
took the whole batch down with it.
"""
from __future__ import annotations

import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, Response, UploadFile

from app import main
from app.ingest import jobs
from tests.conftest import requires_db

pytestmark = [requires_db, pytest.mark.asyncio]

GOOD = b"# Job Test A\n\nThe first document, which is perfectly readable.\n"
ALSO_GOOD = b"# Job Test B\n\nThe second document, equally readable.\n"
#: A `.pdf` extension over bytes that are not a PDF. PyMuPDF raises on it,
#: which is exactly the shape of failure that used to abort a whole batch.
CORRUPT_PDF = b"this is not a PDF at all"


def upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(data), filename=name, size=len(data))


async def start_files(files, **kwargs):
    """`ingest_files` takes the `Response` FastAPI injects, so that it can
    answer 200 rather than 202 when the caller asked to wait."""
    return await main.ingest_files(files, Response(), **kwargs)


async def wait_for(job_id: str, timeout: float = 120.0) -> dict:
    """Poll until the job leaves `running`, the way a client does."""
    for _ in range(int(timeout / 0.05)):
        row = await jobs.get(job_id)
        assert row is not None
        if row["status"] != "running":
            return row
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished")


@pytest.fixture(autouse=True)
async def _clean(pool):
    yield
    await jobs.cancel_all()


# --- the shape of the answer ----------------------------------------------


async def test_uploading_answers_with_a_job_before_the_work_is_done(pool):
    job = await start_files([upload("job-a.md", GOOD)], force=True)

    assert job.id.startswith("job_")
    assert job.source == "files"
    assert job.label == "1 file"
    # Answered while the work is still ahead of it — that is the whole point.
    assert job.status == "running"
    assert job.report is None

    finished = await wait_for(job.id)
    assert finished["status"] == "done"
    assert finished["report"]["written"] == 1


async def test_the_finished_job_carries_the_report_the_endpoint_used_to_return(pool):
    # `force` so the assertions do not depend on whether a previous run of
    # this suite left the same document in the corpus.
    job = await start_files([upload("job-b.md", GOOD)], force=True, wait=True)

    assert job.status == "done"
    assert job.report is not None
    assert job.report.documents == 1
    assert job.report.chunk_budget > 0
    assert job.report.written == 1
    assert job.report.results[0].status in {"created", "updated"}


async def test_waiting_is_opt_in_and_returns_the_finished_job(pool):
    job = await start_files([upload("job-c.md", GOOD)], wait=True)
    assert job.status in {"done", "failed"}
    assert job.done == job.total


async def test_the_label_counts_the_files(pool):
    job = await start_files([upload("job-d.md", GOOD), upload("job-e.md", ALSO_GOOD)], force=True, wait=True
    )
    assert job.label == "2 files"
    assert job.total == 2


# --- one bad file no longer takes the batch with it -----------------------


async def test_an_unreadable_file_fails_alone(pool):
    """The eleven readable files are what the user asked for.

    This used to raise `ValueError` out of the endpoint as a 415 for the whole
    request, so a single corrupt PDF meant nothing at all was indexed.
    """
    job = await start_files([
            upload("job-good.md", GOOD),
            upload("job-broken.pdf", CORRUPT_PDF),
            upload("job-good2.md", ALSO_GOOD),
        ],
        force=True,
        wait=True,
    )

    assert job.status == "done", "the job itself succeeded; one document did not"
    assert job.report is not None
    by_path = {r.path: r for r in job.report.results}
    assert by_path["job-broken.pdf"].status == "failed"
    assert by_path["job-broken.pdf"].error
    assert by_path["job-good.md"].status in {"created", "updated"}
    assert by_path["job-good2.md"].status in {"created", "updated"}
    assert job.report.failed == 1
    assert job.report.written == 2


async def test_an_unsupported_type_is_one_failed_row_not_a_dead_request(pool):
    job = await start_files([upload("notes.txt", b"plain text"), upload("job-f.md", GOOD)],
        force=True,
        wait=True,
    )
    assert job.report is not None
    assert job.report.failed == 1
    assert job.report.written == 1


# --- progress -------------------------------------------------------------


async def test_progress_reaches_the_row_while_the_job_runs(pool):
    """`done` has to move for every document, including unchanged ones.

    A re-sync is almost all unchanged documents, and those leave the indexing
    loop by a different path. Counting only the slow path shows a bar that
    sticks at two of sixty and then jumps to finished.
    """
    files = [upload(f"job-p{i}.md", GOOD + str(i).encode()) for i in range(4)]
    first = await start_files(files, force=True, wait=True)
    assert first.done == 4

    # Every one of them is unchanged the second time around.
    again = await start_files([upload(f"job-p{i}.md", GOOD + str(i).encode()) for i in range(4)], wait=True
    )
    assert again.report is not None
    assert again.report.unchanged == 4
    assert again.done == 4


# --- reading them back ----------------------------------------------------


async def test_an_unknown_job_is_a_404(pool):
    with pytest.raises(HTTPException) as e:
        await main.job_detail("job_nosuchthing")
    assert e.value.status_code == 404


async def test_jobs_are_listed_newest_first(pool):
    older = await start_files([upload("job-old.md", GOOD)], wait=True)
    newer = await start_files([upload("job-new.md", ALSO_GOOD)], wait=True)

    listed = [j.id for j in await main.list_jobs(limit=20)]
    assert listed.index(newer.id) < listed.index(older.id)


# --- a process that went away ---------------------------------------------


async def test_a_job_left_running_by_a_dead_process_is_closed_out(pool):
    """Its task lived in an event loop that no longer exists.

    Left as `running`, the row reads as work in progress for ever.
    """
    stale = await jobs.create(source="files", label="stranded", total=3, phase="indexing")
    await jobs.advance(stale, done=1, current="half-way.md")

    closed = await jobs.abandon_running()
    assert closed >= 1

    row = await jobs.get(stale)
    assert row is not None
    assert row["status"] == "failed"
    assert "restarted" in row["error"]


async def test_closing_out_stale_jobs_leaves_finished_ones_alone(pool):
    done = await start_files([upload("job-keep.md", GOOD)], wait=True)
    await jobs.abandon_running()

    row = await jobs.get(done.id)
    assert row is not None and row["status"] == "done" and row["error"] is None


# --- validation still happens in the request ------------------------------


async def test_duplicate_names_are_still_refused_before_a_job_exists(pool):
    """A 409 the caller can act on, not a job they have to poll to discover."""
    before = len(await jobs.recent(limit=100))
    with pytest.raises(HTTPException) as e:
        await start_files([upload("a/dup.md", GOOD), upload("b/dup.md", ALSO_GOOD)])
    assert e.value.status_code == 409
    assert len(await jobs.recent(limit=100)) == before


async def test_a_malformed_repository_is_a_400_not_a_failed_job(pool):
    from app.schemas import GitHubIngestRequest

    before = len(await jobs.recent(limit=100))
    with pytest.raises(HTTPException) as e:
        await main.ingest_github(GitHubIngestRequest(repo="../.."), Response())
    assert e.value.status_code == 400
    assert len(await jobs.recent(limit=100)) == before
