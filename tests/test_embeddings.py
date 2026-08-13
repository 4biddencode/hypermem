"""Tests for the EmbeddingClient: success, graceful None fallback, and the
transient-vs-permanent failure distinction (a cold Ollama model load must
not permanently disable semantic recall)."""

import httpx
import pytest

from hypermem.embeddings import EmbeddingClient


def make_ec(handler) -> EmbeddingClient:
    return EmbeddingClient(
        provider="ollama", model="nomic-embed-text",
        endpoint="http://stub.local:11434",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_success_probes_available():
    ec = make_ec(lambda r: httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]}))
    vec = await ec.embed("hello")
    assert vec == [0.1, 0.2, 0.3]
    assert ec._available is True
    assert ec.available is True


@pytest.mark.asyncio
async def test_5xx_is_transient_and_recovers():
    """A 5xx (e.g. model still loading) parks the client briefly instead of
    latching it off forever — the first call after backoff retries."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": "model loading"})
        return httpx.Response(200, json={"embedding": [1.0]})

    ec = make_ec(handler)
    assert await ec.embed("hello") is None
    assert ec._available is not False   # not permanently broken
    assert ec.available is False        # inside the backoff window
    assert calls["n"] == 1              # no hammering during backoff
    assert await ec.embed("hello") is None  # short-circuits while parked

    ec._retry_at = 0  # backoff window elapsed
    assert await ec.embed("hello") == [1.0]
    assert ec.available is True


@pytest.mark.asyncio
async def test_404_is_permanent():
    """A 4xx (model not installed / bad config) is a real misconfig — no point
    retrying, semantic recall stays off with graceful LLM+lexical fallback."""
    ec = make_ec(lambda r: httpx.Response(404, json={"error": "model not found"}))
    assert await ec.embed("hello") is None
    assert ec._available is False
    assert ec.available is False
    assert await ec.embed("hello") is None  # short-circuits


@pytest.mark.asyncio
async def test_disabled_provider_never_calls():
    ec = EmbeddingClient(provider="none")
    assert ec.enabled is False
    assert ec.available is False
    assert await ec.embed("hello") is None
