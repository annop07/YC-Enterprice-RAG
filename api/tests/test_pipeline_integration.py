"""What re-ingesting an unchanged document does.

The interesting case is not the one that re-embeds — it is the one that decides
nothing needs re-embedding, because that is where a document can quietly keep
stale metadata forever.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from app import db
from app.ingest.connectors import RawDocument
from app.ingest.pipeline import ingest
from tests.conftest import requires_db

pytestmark = [requires_db, pytest.mark.asyncio]

TEXT = "# Sync\n\n## Blobs\n\nThe tree listing carries a sha for every file.\n"


def document(*, commit: str, blob_sha: str) -> RawDocument:
    return RawDocument(
        source_type="github",
        source_id="acme/handbook@docs/sync.md",
        title="Sync",
        path="docs/sync.md",
        text=TEXT,
        url=f"https://github.com/acme/handbook/blob/{commit}/docs/sync.md",
        meta={"repo": "acme/handbook", "commit": commit, "blob_sha": blob_sha},
    )


@pytest_asyncio.fixture(loop_scope="session")
async def clean(pool):
    async with db.pool().connection() as conn:
        await conn.execute(
            "DELETE FROM document WHERE source_id LIKE %s", ("acme/handbook@%",)
        )
    yield


async def stored() -> tuple[str, dict]:
    row = await db.fetch_one(
        "SELECT url, meta FROM document WHERE source_id = %s",
        ("acme/handbook@docs/sync.md",),
    )
    return row[0], row[1]


async def test_unchanged_text_still_records_where_it_came_from(clean):
    """The trap in skipping work: the blob sha is what lets the *next* sync
    skip fetching this file at all, and it arrives on a document whose text has
    not changed — so a pipeline that writes nothing for an unchanged document
    never records one, and the optimisation never starts.
    """
    await ingest([document(commit="aaa111", blob_sha="blob-1")])
    assert (await stored())[1]["blob_sha"] == "blob-1"

    # Same text at a later commit: the file was moved forward by some other
    # commit in the repository, so both the permalink and the blob sha changed.
    report = await ingest([document(commit="bbb222", blob_sha="blob-2")])

    assert [r.status for r in report.results] == ["unchanged"]
    assert report.chunks == 0, "an unchanged document is not re-chunked"

    url, meta = await stored()
    assert meta["blob_sha"] == "blob-2", "the next sync would fetch this file again"
    assert meta["commit"] == "bbb222"
    assert url.endswith("/blob/bbb222/docs/sync.md")


async def test_nothing_is_written_when_nothing_moved(clean):
    """The refresh is conditional — re-running a sync that changed nothing must
    not rewrite every row in the corpus."""
    await ingest([document(commit="aaa111", blob_sha="blob-1")])
    before = await db.fetch_one(
        "SELECT xmin::text FROM document WHERE source_id = %s",
        ("acme/handbook@docs/sync.md",),
    )

    await ingest([document(commit="aaa111", blob_sha="blob-1")])
    after = await db.fetch_one(
        "SELECT xmin::text FROM document WHERE source_id = %s",
        ("acme/handbook@docs/sync.md",),
    )

    assert before == after, "the row was rewritten by an ingest that changed nothing"


async def test_a_second_sync_of_an_unchanged_repository_fetches_no_blobs(clean):
    """The whole B-22 loop, through the database rather than around it.

    The connector's own tests stub the shas in by hand; this one takes them
    from `document.meta` the way the endpoint does, which is the half that was
    never exercised — a broken lookup there disables the optimisation silently,
    since fetching every blob is exactly what the code did before.
    """
    from app.ingest.github import GitHubConnector
    from app.main import _known_blob_shas
    from tests.test_github import blob_sha, build_client

    first_calls: list[str] = []
    first = GitHubConnector("acme/handbook", client=build_client(calls=first_calls))
    await ingest(list(first.load()))

    assert [c for c in first_calls if "/git/blobs/" in c], "the first sync reads blobs"
    known = await _known_blob_shas("acme/handbook")
    assert known == {
        "README.md": blob_sha("README.md"),
        "docs/setup.md": blob_sha("docs/setup.md"),
    }, "the shas the first sync stored are not what the endpoint reads back"

    second_calls: list[str] = []
    second = GitHubConnector(
        "acme/handbook",
        client=build_client(calls=second_calls),
        known_blob_shas=known,
    )
    docs = list(second.load())

    assert docs == []
    assert sorted(second.skipped) == ["README.md", "docs/setup.md"]
    assert [c for c in second_calls if "/git/blobs/" in c] == []


async def test_a_repository_never_borrows_another_repositorys_shas(clean):
    """`my_repo` and `myXrepo` are the same string to LIKE, and handing one
    repository another's shas would skip a fetch and leave the wrong text
    indexed."""
    from app.ingest.github import GitHubConnector
    from app.main import _known_blob_shas
    from tests.test_github import build_client

    await ingest(list(GitHubConnector("acme/handbook", client=build_client()).load()))

    assert await _known_blob_shas("acme_handbook") == {}
    assert await _known_blob_shas("acme/hand") == {}
    assert await _known_blob_shas("acme/handbook") != {}
