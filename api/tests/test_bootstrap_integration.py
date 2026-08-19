"""Schema bootstrap against a real Postgres.

The check worth having here is the one the schema file cannot make itself:
`CREATE TABLE IF NOT EXISTS` is a no-op against a table that already exists, so
it will happily leave a `vector(384)` column in place under an EMBED_MODEL that
produces 1024 dimensions.
"""
from __future__ import annotations

import pytest

from app import db
from app.config import get_settings
from tests.conftest import requires_db

pytestmark = [requires_db, pytest.mark.asyncio]


async def test_bootstrap_is_safe_to_run_again(pool):
    """It runs on every startup, so it has to be idempotent."""
    await db.bootstrap()

    row = await db.fetch_one(db.EMBEDDING_DIM_SQL)
    assert row[0] == get_settings().embed_dim


async def test_a_database_built_for_another_model_fails_at_startup(pool, monkeypatch):
    """Without this the mismatch surfaces later and elsewhere: startup
    succeeds, `get_embedder` succeeds because it only checks the model against
    settings, and the first insert or search dies inside pgvector with a type
    error that says nothing about which setting to change.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "embed_dim", 1024)

    with pytest.raises(db.SchemaMismatch) as raised:
        await db.bootstrap()

    message = str(raised.value)
    assert "1024" in message and "vector(384)" in message
    # A failure that does not say what to run is a failure the reader has to
    # take to the source.
    assert "DROP TABLE" in message
    assert "message_citation" in message, "say that history survives the drop"


async def test_the_mismatch_check_does_not_damage_the_database(pool):
    """The failing bootstrap above ran the schema file first. Nothing it did
    may have touched the existing tables."""
    row = await db.fetch_one(db.EMBEDDING_DIM_SQL)
    assert row[0] == get_settings().embed_dim

    counts = await db.corpus_stats()
    assert counts["documents"] >= 0
