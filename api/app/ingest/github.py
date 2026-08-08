"""GitHub as a document source.

Two decisions shape this connector:

* **One tree call, not a walk.** `git/trees?recursive=1` returns the whole
  repository layout in a single request. Listing directory by directory burns
  the rate limit — 60 requests an hour unauthenticated — on nothing.

* **Permalinks are built from the commit SHA, never the branch.** A link
  against `main` rots the moment someone edits the file: the line numbers in
  the citation start pointing at different text. A link against the SHA keeps
  pointing at exactly what was indexed.
"""
from __future__ import annotations

import base64
import logging
from typing import Iterator

import httpx

from app.ingest.connectors import (
    MARKDOWN_SUFFIXES,
    RawDocument,
    is_vendored,
    title_from_markdown,
)

log = logging.getLogger(__name__)

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


class GitHubConnector:
    name = "github"

    def __init__(
        self,
        repo: str,
        *,
        ref: str | None = None,
        path_prefix: str = "",
        token: str | None = None,
        suffixes: set[str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if repo.count("/") != 1:
            raise ValueError(f"repo must be 'owner/name', got {repo!r}")
        self.repo = repo
        self.ref = ref
        self.path_prefix = path_prefix.strip("/")
        self.suffixes = suffixes or MARKDOWN_SUFFIXES
        self._own_client = client is None
        self._client = client or httpx.Client(timeout=30.0)
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    # --- HTTP ------------------------------------------------------------

    def _get(self, path: str, **params) -> dict:
        response = self._client.get(f"{API}{path}", headers=self._headers, params=params)
        if response.status_code == 404:
            raise GitHubError(f"{path}: not found (private repo without a token?)")
        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise GitHubError(
                "GitHub rate limit reached. Unauthenticated requests get 60 per "
                "hour; set GITHUB_TOKEN for 5000."
            )
        response.raise_for_status()
        return response.json()

    # --- Loading ---------------------------------------------------------

    def resolve_commit(self) -> str:
        ref = self.ref or self._get(f"/repos/{self.repo}")["default_branch"]
        return self._get(f"/repos/{self.repo}/commits/{ref}")["sha"]

    def load(self) -> Iterator[RawDocument]:
        try:
            commit = self.resolve_commit()
            tree = self._get(
                f"/repos/{self.repo}/git/trees/{commit}", recursive="1"
            )

            if tree.get("truncated"):
                # Silent truncation would read as "the repo has no more docs".
                log.warning(
                    "%s: the tree listing was truncated by GitHub; some files "
                    "were not seen. Narrow it with path_prefix.",
                    self.repo,
                )

            for entry in tree.get("tree", []):
                if entry.get("type") != "blob":
                    continue
                path = entry["path"]
                if self.path_prefix and not path.startswith(f"{self.path_prefix}/"):
                    continue
                # Only the segments *below* the prefix are judged, mirroring
                # `iter_files`, which never judges the segments of its own root:
                # a prefix of ".github/workflows" was asked for deliberately and
                # must not filter itself out.
                below = path[len(self.path_prefix) + 1 :] if self.path_prefix else path
                # Before the blob fetch, not after — a repository that committed
                # its `.venv` costs one API call per vendored file otherwise, and
                # unauthenticated requests run out after sixty.
                if is_vendored(below.split("/")):
                    continue
                if not any(path.lower().endswith(s) for s in self.suffixes):
                    continue

                text = self._blob_text(entry["sha"])
                if not text.strip():
                    continue

                yield RawDocument(
                    source_type="github",
                    # Deliberately excludes the commit: the same file at a new
                    # commit must update its row, not add a second one.
                    source_id=f"{self.repo}@{path}",
                    title=title_from_markdown(text, path.rsplit("/", 1)[-1]),
                    path=path,
                    text=text,
                    url=f"https://github.com/{self.repo}/blob/{commit}/{path}",
                    meta={"repo": self.repo, "commit": commit},
                )
        finally:
            if self._own_client:
                self._client.close()

    def _blob_text(self, blob_sha: str) -> str:
        blob = self._get(f"/repos/{self.repo}/git/blobs/{blob_sha}")
        if blob.get("encoding") != "base64":
            raise GitHubError(f"unexpected blob encoding: {blob.get('encoding')}")
        return base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
