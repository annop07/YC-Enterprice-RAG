"""Every way a GitHub ingest can fail, and what the caller is told.

Separate from `test_github.py`, which pins the happy-path request pattern.
This file is about the other half: before it, `_get` named 404 and
403-with-"rate limit" and left everything else to `raise_for_status`, so a
mistyped branch — which GitHub answers with 422 — reached the browser as
`500 Internal Server Error` with the reason visible only in the server log.
"""
from __future__ import annotations

import httpx
import pytest

from app.ingest.github import GitHubConnector, GitHubError


def connector(handler, **kwargs) -> GitHubConnector:
    return GitHubConnector(
        kwargs.pop("repo", "owner/name"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def answering(status: int, body: dict | str = "", headers: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, dict):
            return httpx.Response(status, json=body, headers=headers or {})
        return httpx.Response(status, text=body, headers=headers or {})

    return handler


# --- repository and ref validation ----------------------------------------


@pytest.mark.parametrize(
    "repo",
    [
        "../..",          # addressed https://api.github.com/ — the API root
        "..",
        "a/../b",
        "owner",          # no name at all
        "owner/name/x",
        "own er/name",
        "owner/na me",
        "/name",
        "owner/",
        "",
        "./.",
    ],
)
def test_a_repository_that_is_not_owner_slash_name_is_refused(repo: str) -> None:
    """The value is interpolated into an api.github.com path.

    `../..` normalised to the API root, which answers 200 with a JSON object
    that has no `default_branch` — so the connector died on a KeyError and the
    endpoint returned 500 for what is a typo in a form field.
    """
    with pytest.raises(ValueError):
        GitHubConnector(repo)


@pytest.mark.parametrize("repo", ["pgvector/pgvector", "owner/name", "a.b/c-d_e", "o/r.git"])
def test_a_real_repository_name_is_accepted(repo: str) -> None:
    GitHubConnector(repo)


@pytest.mark.parametrize(
    "ref",
    ["../../../user", "a..b", "feature branch", "-x", "/main", "main/", "he~ad", "x^", "a:b", "", "q?", "s*", "[y]"],
)
def test_a_ref_that_git_would_not_accept_is_refused(ref: str) -> None:
    """A ref reaches the URL too, and `../../../user` is a request to /user.

    It goes out with this server's `GITHUB_TOKEN` attached, which makes it an
    authenticated request to an endpoint the caller chose.
    """
    with pytest.raises(ValueError):
        GitHubConnector("owner/name", ref=ref)


@pytest.mark.parametrize("ref", ["main", "v1.0", "release/2026-08", "refs/heads/main", "7f3a91c"])
def test_a_real_ref_is_accepted(ref: str) -> None:
    GitHubConnector("owner/name", ref=ref)


# --- HTTP failures --------------------------------------------------------


def test_an_unresolvable_ref_is_a_github_error_not_a_status_exception() -> None:
    """422 is what GitHub answers for a branch that does not exist.

    This is the case that produced the 500: it is neither 404 nor
    403-rate-limit, so it fell through to `raise_for_status`.
    """
    c = connector(answering(422, {"message": "No commit found for SHA: nope"}), ref="nope")
    with pytest.raises(GitHubError) as e:
        list(c.load())
    assert "No commit found" in str(e.value)


@pytest.mark.parametrize("status", [400, 401, 409, 451, 500, 502, 503])
def test_every_error_status_becomes_a_github_error(status: int) -> None:
    c = connector(answering(status, {"message": f"stub {status}"}))
    with pytest.raises(GitHubError):
        list(c.load())


def test_a_404_says_what_to_check() -> None:
    c = connector(answering(404, {"message": "Not Found"}))
    with pytest.raises(GitHubError) as e:
        list(c.load())
    assert "GITHUB_TOKEN" in str(e.value)


def test_a_rejected_token_is_named_as_such() -> None:
    c = connector(answering(401, {"message": "Bad credentials"}), token="rotten")
    with pytest.raises(GitHubError) as e:
        list(c.load())
    assert "GITHUB_TOKEN" in str(e.value)


def test_a_depleted_rate_limit_is_read_off_the_header_not_the_message() -> None:
    """GitHub reports a secondary limit as 429 and does not always say "rate limit"."""
    c = connector(
        answering(429, {"message": "You have exceeded a secondary rate limit"},
                  {"x-ratelimit-remaining": "0", "retry-after": "60"}),
    )
    with pytest.raises(GitHubError) as e:
        list(c.load())
    assert "rate limit" in str(e.value).lower()
    assert "60s" in str(e.value)


def test_the_unauthenticated_budget_is_the_one_quoted_when_there_is_no_token() -> None:
    c = connector(answering(403, {"message": "API rate limit exceeded"},
                            {"x-ratelimit-remaining": "0"}))
    with pytest.raises(GitHubError) as e:
        list(c.load())
    assert "60 an hour" in str(e.value)


def test_a_network_failure_is_a_github_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nowhere to connect to", request=request)

    with pytest.raises(GitHubError) as e:
        list(connector(refuse).load())
    assert "could not reach GitHub" in str(e.value)


def test_a_timeout_says_it_timed_out() -> None:
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(GitHubError) as e:
        list(connector(stall).load())
    assert "in time" in str(e.value)


def test_a_body_that_is_not_json_is_a_github_error() -> None:
    c = connector(answering(200, "<html>a proxy login page</html>"))
    with pytest.raises(GitHubError) as e:
        list(c.load())
    assert "not JSON" in str(e.value)


def test_a_response_missing_the_field_names_the_field() -> None:
    """What the API root returned: a valid JSON object, with nothing expected in it."""
    c = connector(answering(200, {"current_user_url": "https://api.github.com/user"}))
    with pytest.raises(GitHubError) as e:
        list(c.load())
    assert "default_branch" in str(e.value)
