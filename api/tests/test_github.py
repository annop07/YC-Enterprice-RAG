"""GitHub connector, against a stubbed API.

No network: `httpx.MockTransport` answers the four endpoints the connector
uses, so the test pins the *request pattern* — one tree call, blobs by sha —
as much as the parsing.
"""
from __future__ import annotations

import base64
import json
import logging

import httpx
import pytest

from app.ingest.github import GitHubConnector, GitHubError

COMMIT = "7f3a91c0d2e4b6a8c1f3e5d7b9a1c3e5d7f9b1a3"
FILES = {
    "README.md": "# Project\n\nRoot readme.\n",
    "docs/setup.md": "# Setup\n\nHow to set up.\n",
    "docs/img/logo.png": "not markdown",
    "src/main.py": "print()\n",
}
#: A repository that committed its dependencies, which is the case the filter
#: exists for. Only `docs/guide.md` belongs to the corpus.
VENDORED = {
    "docs/guide.md": "# Guide\n\nKept.\n",
    ".venv/lib/site-packages/x/LICENSE.md": "# License\n\nVendored.\n",
    "node_modules/y/README.md": "# Dep\n\nVendored.\n",
    "build/generated/notes.md": "# Generated\n\nVendored.\n",
    ".github/ISSUE_TEMPLATE/bug.md": "# Bug\n\nDot-directory.\n",
}


def blob_sha(path: str) -> str:
    """A fake sha that says which file it came from, so calls can be asserted on."""
    return "blob-" + path.replace("/", "-")


def build_client(
    *,
    truncated: bool = False,
    calls: list[str] | None = None,
    files: dict[str, str] | None = None,
):
    files = FILES if files is None else files

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        path = request.url.path

        if path == "/repos/acme/handbook":
            return httpx.Response(200, json={"default_branch": "main"})
        if path.startswith("/repos/acme/handbook/commits/"):
            return httpx.Response(200, json={"sha": COMMIT})
        if path == f"/repos/acme/handbook/git/trees/{COMMIT}":
            assert request.url.params.get("recursive") == "1"
            return httpx.Response(
                200,
                json={
                    "truncated": truncated,
                    "tree": [
                        {"path": p, "type": "blob", "sha": blob_sha(p)} for p in files
                    ]
                    + [{"path": "docs", "type": "tree", "sha": "treesha"}],
                },
            )
        if path.startswith("/repos/acme/handbook/git/blobs/"):
            sha = path.rsplit("/", 1)[-1]
            source = next(p for p in files if blob_sha(p) == sha)
            return httpx.Response(
                200,
                json={
                    "encoding": "base64",
                    "content": base64.b64encode(files[source].encode()).decode(),
                },
            )
        return httpx.Response(404, json={"message": "Not Found"})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_only_markdown_is_loaded_and_the_tree_is_fetched_once():
    calls: list[str] = []
    docs = list(GitHubConnector("acme/handbook", client=build_client(calls=calls)).load())

    assert sorted(d.path for d in docs) == ["README.md", "docs/setup.md"]
    tree_calls = [c for c in calls if "/git/trees/" in c]
    assert len(tree_calls) == 1, "the whole layout comes from one recursive call"


def test_permalinks_pin_the_commit_not_the_branch():
    """A link against main starts pointing at different lines after any edit."""
    docs = list(GitHubConnector("acme/handbook", client=build_client()).load())
    for doc in docs:
        assert doc.url == f"https://github.com/acme/handbook/blob/{COMMIT}/{doc.path}"
        assert "/blob/main/" not in doc.url
        assert doc.meta["commit"] == COMMIT


def test_source_id_excludes_the_commit_so_a_reindex_updates_in_place():
    docs = {d.path: d for d in GitHubConnector("acme/handbook", client=build_client()).load()}
    assert docs["docs/setup.md"].source_id == "acme/handbook@docs/setup.md"
    assert COMMIT not in docs["docs/setup.md"].source_id


def test_path_prefix_restricts_the_subtree():
    docs = list(
        GitHubConnector(
            "acme/handbook", path_prefix="docs", client=build_client()
        ).load()
    )
    assert [d.path for d in docs] == ["docs/setup.md"]


def test_titles_and_text_survive_the_base64_round_trip():
    docs = {d.path: d for d in GitHubConnector("acme/handbook", client=build_client()).load()}
    assert docs["docs/setup.md"].title == "Setup"
    assert docs["docs/setup.md"].text == FILES["docs/setup.md"]


def test_a_truncated_tree_is_reported_rather_than_looking_complete(caplog):
    with caplog.at_level(logging.WARNING):
        list(GitHubConnector("acme/handbook", client=build_client(truncated=True)).load())
    assert "truncated" in caplog.text


def test_a_rate_limited_response_says_what_to_do_about_it():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, text=json.dumps({"message": "API rate limit exceeded"})
        )

    connector = GitHubConnector(
        "acme/handbook", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        list(connector.load())


def test_vendored_and_dot_directories_do_not_enter_the_corpus():
    """A repo that committed `.venv` or `node_modules` must not drag it in.

    The connector walks the tree itself and so never reached `iter_files`,
    which is where the skip list is applied for every local source.
    """
    docs = list(
        GitHubConnector("acme/handbook", client=build_client(files=VENDORED)).load()
    )
    assert [d.path for d in docs] == ["docs/guide.md"]


def test_a_skipped_path_costs_no_blob_call():
    """Half the bug: filtering after the fetch still spends the rate limit.

    Unauthenticated requests get sixty an hour, and one vendored tree is
    thousands of files.
    """
    calls: list[str] = []
    list(
        GitHubConnector(
            "acme/handbook", client=build_client(files=VENDORED, calls=calls)
        ).load()
    )
    blob_calls = [c for c in calls if "/git/blobs/" in c]
    assert blob_calls == [f"/repos/acme/handbook/git/blobs/{blob_sha('docs/guide.md')}"]


def test_an_explicitly_requested_dot_subtree_is_not_filtered_out():
    """Segments are judged below the prefix, as `iter_files` judges below its root.

    Asking for `.github/ISSUE_TEMPLATE` and getting nothing back would be the
    skip list overruling the caller.
    """
    docs = list(
        GitHubConnector(
            "acme/handbook",
            path_prefix=".github/ISSUE_TEMPLATE",
            client=build_client(files=VENDORED),
        ).load()
    )
    assert [d.path for d in docs] == [".github/ISSUE_TEMPLATE/bug.md"]


def test_a_malformed_repo_is_rejected_before_any_request():
    with pytest.raises(ValueError, match="owner/name"):
        GitHubConnector("handbook")
