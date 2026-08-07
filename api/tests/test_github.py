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
BLOB_SHA = {path: f"blob{i}" for i, path in enumerate(FILES)}


def build_client(*, truncated: bool = False, calls: list[str] | None = None):
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
                        {"path": p, "type": "blob", "sha": BLOB_SHA[p]} for p in FILES
                    ]
                    + [{"path": "docs", "type": "tree", "sha": "treesha"}],
                },
            )
        if path.startswith("/repos/acme/handbook/git/blobs/"):
            sha = path.rsplit("/", 1)[-1]
            source = next(p for p, s in BLOB_SHA.items() if s == sha)
            return httpx.Response(
                200,
                json={
                    "encoding": "base64",
                    "content": base64.b64encode(FILES[source].encode()).decode(),
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


def test_a_malformed_repo_is_rejected_before_any_request():
    with pytest.raises(ValueError, match="owner/name"):
        GitHubConnector("handbook")
