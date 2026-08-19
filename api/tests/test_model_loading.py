"""How the ONNX models get loaded — once, and not on the event loop.

No model is loaded here: what is under test is the locking around the load, so
the build step is replaced by a slow fake that counts how many times it ran.
That is the whole failure mode — `lru_cache` looks up its cache *before* the
function body runs, so two threads arriving together both missed and both
loaded, which is two ONNX sessions and twice the memory for one model.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.ingest import embedder as embedder_module
from app.retrieval import reranker as reranker_module


def race(get, workers: int = 6) -> list[object]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda _: get(), range(workers)))


def test_threads_arriving_together_load_one_embedder(monkeypatch):
    builds: list[float] = []

    def slow_build():
        builds.append(time.monotonic())
        time.sleep(0.2)  # long enough for the other threads to arrive
        return object()

    monkeypatch.setattr(embedder_module, "_build", slow_build)
    monkeypatch.setattr(embedder_module, "_instance", None)

    instances = race(embedder_module.get_embedder)

    assert len(builds) == 1, f"the model was loaded {len(builds)} times"
    assert len({id(i) for i in instances}) == 1


def test_threads_arriving_together_load_one_reranker(monkeypatch):
    builds: list[str] = []

    class SlowReranker:
        def __init__(self, model_name: str) -> None:
            builds.append(model_name)
            time.sleep(0.2)

    monkeypatch.setattr(reranker_module, "Reranker", SlowReranker)
    monkeypatch.setattr(reranker_module, "_instance", None)

    instances = race(reranker_module.get_reranker)

    assert len(builds) == 1
    assert len({id(i) for i in instances}) == 1


@pytest.mark.asyncio
async def test_the_warm_up_keeps_the_event_loop_answering(monkeypatch):
    """The point of doing this at startup, on a thread: the first question used
    to load the models inline, so the whole server — `/health` included — was
    unavailable for as long as it took, with nothing to say why.
    """
    from app import main

    def slow_build():
        time.sleep(0.3)
        return object()

    monkeypatch.setattr(embedder_module, "_build", slow_build)
    monkeypatch.setattr(embedder_module, "_instance", None)
    monkeypatch.setattr(main, "get_reranker", lambda: object())
    monkeypatch.setattr(main, "_models_ready", False)

    warm = asyncio.create_task(main._warm_models())
    # While that runs, the loop is still free — if the load were inline this
    # would not get a turn until it finished.
    ticks = 0
    while not warm.done():
        await asyncio.sleep(0.01)
        ticks += 1

    await warm
    assert ticks > 1, "the event loop was blocked for the whole load"
    assert main._models_ready is True


@pytest.mark.asyncio
async def test_a_failed_warm_up_does_not_stop_the_application(monkeypatch):
    """A missing model file or an offline Hub is a reason to start degraded and
    say so in `/health`, not a reason to refuse to start."""
    from app import main

    def explode():
        raise RuntimeError("the Hub is unreachable")

    monkeypatch.setattr(embedder_module, "_build", explode)
    monkeypatch.setattr(embedder_module, "_instance", None)
    monkeypatch.setattr(main, "_models_ready", False)

    await main._warm_models()

    assert main._models_ready is False
