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
import re
from typing import Any, Iterator
from urllib.parse import quote

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
    """Anything that went wrong reaching GitHub, in terms a caller can report.

    Every failure this module can have is one of these. That is the point of
    it: the endpoint above maps `GitHubError` to a 502 with the message
    attached, and anything that escapes as a different type reaches the client
    as `500 Internal Server Error` with the reason visible only in the server
    log. That is what a wrong branch name used to do — GitHub answers 422 for
    one, which is neither of the two statuses this class used to be raised
    for, so `raise_for_status` threw `httpx.HTTPStatusError` and the user was
    told nothing at all.
    """


#: `owner/name`. Both segments are as permissive as GitHub itself, and no more:
#: this string is interpolated into an API path, so `../..` was a request to
#: `https://api.github.com/` — which answers 200 with the API root, and the
#: connector then failed on a missing `default_branch` key with a 500. A
#: repository whose *name* is `..` does not exist, so nothing legitimate is
#: refused by ruling it out.
_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")

#: Git's own rules, reduced to what matters here: a ref may contain `/`
#: (`release/2026-08`), and may not contain `..`, whitespace, or any of the
#: characters git reserves. `..` is the one that mattered — a ref of
#: `../../../user` walked out of the repository and issued an authenticated
#: request to a different endpoint entirely.
_REF_REJECT = re.compile(r"\.\.|[\s~^:?*\[\]\\]|^[/-]|/$")


def _check_repo(repo: str) -> None:
    owner, slash, name = repo.partition("/")
    if (
        not slash
        or not _SEGMENT.match(owner)
        or not _SEGMENT.match(name)
        or owner in {".", ".."}
        or name in {".", ".."}
    ):
        raise ValueError(f"repo must be 'owner/name', got {repo!r}")


def _check_ref(ref: str) -> None:
    if not ref or _REF_REJECT.search(ref):
        raise ValueError(f"not a usable git ref: {ref!r}")


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
        known_blob_shas: dict[str, str] | None = None,
    ) -> None:
        _check_repo(repo)
        if ref is not None:
            _check_ref(ref)
        self.repo = repo
        self.ref = ref
        self.path_prefix = path_prefix.strip("/")
        self.suffixes = suffixes or MARKDOWN_SUFFIXES
        #: path -> the blob sha already indexed for it. A git blob sha *is* a
        #: content hash, and the tree listing carries one per file, so a match
        #: means the file cannot have changed and its content never has to be
        #: fetched. Without this, re-syncing costs one request per file every
        #: time even when nothing moved — sixty files exhaust the whole
        #: unauthenticated hourly budget on files the store already has.
        self.known_blob_shas = known_blob_shas or {}
        #: Paths skipped because of the above, for the caller to report as
        #: unchanged. They are not yielded: there is no text to yield.
        self.skipped: list[str] = []
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
        """One GET, with every way it can fail turned into a `GitHubError`.

        The old version named three outcomes — 404, 403-with-"rate limit", and
        success — and left every other one to `raise_for_status`. GitHub uses
        422 for a ref that does not resolve, 451 for a repository taken down,
        409 for an empty one, 429 for a secondary rate limit, and 5xx for its
        own bad days; a network that is simply down raises before there is a
        status at all. All of those reached the user as a bare 500.
        """
        try:
            response = self._client.get(
                f"{API}{path}", headers=self._headers, params=params
            )
        except httpx.TimeoutException as e:
            raise GitHubError(f"GitHub did not answer in time ({path})") from e
        except httpx.HTTPError as e:
            raise GitHubError(f"could not reach GitHub: {e}") from e

        if response.status_code >= 400:
            raise self._failure(response, path)

        try:
            body = response.json()
        except ValueError as e:
            raise GitHubError(f"{path}: GitHub returned a body that is not JSON") from e
        if not isinstance(body, dict):
            # Every endpoint this connector calls returns an object. Anything
            # else means the path was not the one intended — which is exactly
            # what a traversal in `repo` or `ref` produced.
            raise GitHubError(f"{path}: unexpected response shape from GitHub")
        return body

    def _failure(self, response: httpx.Response, path: str) -> GitHubError:
        """GitHub's own explanation of a >=400, in one sentence."""
        try:
            payload = response.json()
            message = str(payload.get("message", "")).strip() if isinstance(payload, dict) else ""
        except ValueError:
            message = ""

        status = response.status_code
        # A depleted budget is reported as 403 or 429 depending on which limit
        # was hit, and the header is the reliable half of the signal — the
        # wording of the message is not.
        depleted = response.headers.get("x-ratelimit-remaining") == "0"
        if status in (403, 429) and (depleted or "rate limit" in message.lower()):
            authenticated = "Authorization" in self._headers
            budget = (
                "Authenticated requests get 5000 an hour"
                if authenticated
                else "Unauthenticated requests get 60 an hour; set GITHUB_TOKEN for 5000"
            )
            retry = response.headers.get("retry-after")
            when = f" Retry after {retry}s." if retry else ""
            return GitHubError(f"GitHub rate limit reached. {budget}.{when}")

        if status == 404:
            return GitHubError(
                f"{path}: not found on GitHub (a typo, or a private repository "
                "without GITHUB_TOKEN set)"
            )
        if status == 401:
            return GitHubError("GITHUB_TOKEN was rejected by GitHub (expired or revoked?)")
        if status == 422:
            # What a branch, tag or SHA that does not exist looks like.
            return GitHubError(f"{path}: GitHub could not resolve that — {message or 'unprocessable'}")

        return GitHubError(f"{path}: GitHub returned {status}{f' — {message}' if message else ''}")

    # --- Loading ---------------------------------------------------------

    def _expect(self, body: dict[str, Any], key: str, path: str) -> Any:
        """Read a field GitHub is contracted to send, or say which one was missing.

        `body["default_branch"]` used to be read straight off the response, so
        a request that reached some *other* endpoint — which is what a repo of
        `../..` did — died on `KeyError: 'default_branch'` and a 500.
        """
        if key not in body:
            raise GitHubError(f"{path}: GitHub's response has no {key!r}")
        return body[key]

    def resolve_commit(self) -> str:
        repo = quote(self.repo, safe="/")
        if self.ref:
            # `safe="/"` keeps `release/2026-08` addressable as a path, which
            # is how GitHub resolves a branch with a slash in it. `_check_ref`
            # has already refused the shapes that would leave the repository.
            ref = quote(self.ref, safe="/")
        else:
            path = f"/repos/{repo}"
            ref = quote(str(self._expect(self._get(path), "default_branch", path)), safe="/")

        path = f"/repos/{repo}/commits/{ref}"
        return str(self._expect(self._get(path), "sha", path))

    def load(self) -> Iterator[RawDocument]:
        try:
            commit = self.resolve_commit()
            tree = self._get(
                f"/repos/{quote(self.repo, safe='/')}/git/trees/{commit}",
                recursive="1",
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

                sha = entry["sha"]
                if self.known_blob_shas.get(path) == sha:
                    self.skipped.append(path)
                    continue

                text = self._blob_text(sha)
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
                    # `blob_sha` is what the next sync compares against; it is
                    # stored even when the text turns out to be unchanged.
                    meta={"repo": self.repo, "commit": commit, "blob_sha": sha},
                )
        finally:
            if self._own_client:
                self._client.close()

    def _blob_text(self, blob_sha: str) -> str:
        blob = self._get(f"/repos/{quote(self.repo, safe='/')}/git/blobs/{blob_sha}")
        if blob.get("encoding") != "base64":
            raise GitHubError(f"unexpected blob encoding: {blob.get('encoding')}")
        return base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
