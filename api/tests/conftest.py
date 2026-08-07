"""Shared fixtures.

Two decisions here:

* **Integration tests use a real Postgres.** The SQL is the interesting part of
  retrieval, and a mocked database would test the mock. They skip when no
  database is reachable.

* **They use their own database, not the development one.** Search results
  depend on everything in the corpus, so a test asserting "this query finds
  that document" silently becomes a test of whatever else happens to be
  indexed. `rag_test` is created on demand and is the only thing these tests
  can see or damage.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
import pytest_asyncio

from app import db
from app.config import Settings, get_settings

TEST_DATABASE = "rag_test"


def _swap_database(url: str, name: str) -> str:
    parts = urlparse(url)
    return urlunparse(parts._replace(path=f"/{name}"))


def _ensure_test_database(base_url: str) -> bool:
    """Create `rag_test` if it is missing. False when Postgres is not up."""
    try:
        admin = _swap_database(base_url, "postgres")
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DATABASE,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{TEST_DATABASE}"')
        return True
    except Exception:  # noqa: BLE001 — any failure means "no database here"
        return False


_base_url = Settings().database_url
_available = _ensure_test_database(_base_url)

if _available:
    # Set before anything reads the cached settings, so the pool, the schema
    # bootstrap and every query in the suite land on the test database.
    os.environ["DATABASE_URL"] = _swap_database(_base_url, TEST_DATABASE)
    get_settings.cache_clear()

requires_db = pytest.mark.skipif(
    not _available,
    reason="no Postgres on DATABASE_URL — run `docker compose up -d`",
)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pool():
    await db.bootstrap()
    await db.open_pool()
    try:
        yield
    finally:
        await db.close_pool()
