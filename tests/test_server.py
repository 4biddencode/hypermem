"""Tests for the HyperMEM REST server (hermetic: stubbed Ollama API)."""

import asyncio
import pytest
import httpx
from pathlib import Path

from hypermem import HyperMemConfig
from hypermem.server import create_app
from conftest import make_llm


@pytest.fixture
def client(tmp_path: Path):
    """FastAPI app wired to a stubbed Ollama transport and a temp data dir."""
    llm, _ = make_llm()
    app = create_app(HyperMemConfig(), tmp_path, llm=llm)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["sessions"] == 0


@pytest.mark.asyncio
async def test_session_lifecycle(client):
    # Create with fixed id
    resp = await client.post("/sessions", json={"session_id": "rp1"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["session_id"] == "rp1"
    assert data["total_messages"] == 0

    # Duplicate gives 409
    resp = await client.post("/sessions", json={"session_id": "rp1"})
    assert resp.status_code == 409

    # List
    resp = await client.get("/sessions")
    assert [s["conversation_id"] for s in resp.json()["sessions"]] == ["rp1"]

    # Missing gives 404
    resp = await client.get("/sessions/nope")
    assert resp.status_code == 404

    # Delete
    resp = await client.delete("/sessions/rp1")
    assert resp.status_code == 204
    resp = await client.get("/sessions/rp1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_add_message_and_recall(client):
    await client.post("/sessions", json={"session_id": "s1"})

    resp = await client.post("/sessions/s1/messages", json={
        "role": "user",
        "content": "My name is Emanuel and I love hiking in the mountains",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["tagged"] is not None
    assert data["total_messages"] == 1
    assert "emanuel" in " ".join(data["tagged"]["keywords"])

    resp = await client.get("/sessions/s1/recall", params={"query": "What's my name?"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["relevant"]) == 1
    assert "Emanuel" in data["relevant"][0]["content"]

    # Recall without a query gives 400
    resp = await client.get("/sessions/s1/recall")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_remember_and_memories(client):
    await client.post("/sessions", json={"session_id": "s2"})

    resp = await client.post("/sessions/s2/remember", json={
        "content": "The vault password is Starlight",
        "memory_type": "static",
    })
    assert resp.status_code == 200
    mem = resp.json()["memory"]
    assert mem["pinned"] is True
    assert mem["memory_type"] == "static"

    resp = await client.get("/sessions/s2/memories")
    mems = resp.json()["memories"]
    assert len(mems) == 1
    assert mems[0]["content"] == "The vault password is Starlight"


@pytest.mark.asyncio
async def test_context_endpoint(client):
    await client.post("/sessions", json={"session_id": "s3"})
    await client.post("/sessions/s3/messages", json={
        "role": "user", "content": "I am a wizard from Sunhaven",
    })

    resp = await client.get("/sessions/s3/context", params={"message": "Where am I from?"})
    assert resp.status_code == 200
    assert "Sunhaven" in resp.text


@pytest.mark.asyncio
async def test_persona_endpoint(client):
    await client.post("/sessions", json={"session_id": "s4"})
    resp = await client.put("/sessions/s4/persona", json={
        "name": "Elena", "description": "Elven rogue", "traits": ["witty"],
    })
    assert resp.status_code == 200

    state = (await client.get("/sessions/s4/state")).json()
    assert state["persona"]["name"] == "Elena"


@pytest.mark.asyncio
async def test_world_ida_endpoint(client):
    await client.post("/sessions", json={"session_id": "s5"})

    resp = await client.post("/sessions/s5/world-ida/update", json={
        "user_msg": "Let's go into the forest",
        "ai_msg": "I follow you into the woods.",
    })
    assert resp.status_code == 200
    ida = resp.json()["world_ida"]
    assert ida is not None
    assert "scene" in ida and "meta" in ida

    resp = await client.get("/sessions/s5/world-ida")
    assert resp.status_code == 200
    assert resp.json()["world_ida"]["meta"]["turn_count_in_scene"] == 1


@pytest.mark.asyncio
async def test_persistence_across_restart(tmp_path: Path):
    """Sessions survive a server restart (data dir)."""
    llm, _ = make_llm()

    app1 = create_app(HyperMemConfig(), tmp_path, llm=llm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app1), base_url="http://test"
    ) as c:
        await c.post("/sessions", json={"session_id": "persist1"})
        await c.post("/sessions/persist1/messages", json={
            "role": "user", "content": "My sister lives in Oakvale",
        })

    app2 = create_app(HyperMemConfig(), tmp_path, llm=llm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app2), base_url="http://test"
    ) as c:
        resp = await c.get("/sessions")
        assert any(s["conversation_id"] == "persist1" for s in resp.json()["sessions"])
        mems = (await c.get("/sessions/persist1/memories")).json()["memories"]
        assert any("Oakvale" in m["content"] for m in mems)


@pytest.mark.asyncio
async def test_memory_provenance(client):
    """GET /sessions/{id}/memories/{id} explains why a memory exists and
    optionally why it would surface (live score breakdown)."""
    await client.post("/sessions", json={"session_id": "s7"})
    await client.post("/sessions/s7/messages", json={
        "role": "user", "content": "My name is Emanuel and I love hiking",
    })

    mems = (await client.get("/sessions/s7/memories")).json()["memories"]
    mid = mems[0]["id"]

    resp = await client.get(f"/sessions/s7/memories/{mid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lifecycle"] == "active"
    assert data["source_message_id"] is not None
    assert data["stored_from"] == "My name is Emanuel and I love hiking"
    assert "embedding" not in data  # internal vector never leaks

    # With a query, the live hybrid score breakdown
    resp = await client.get(f"/sessions/s7/memories/{mid}",
                            params={"query": "What's my name?"})
    ranking = resp.json()["ranking"]
    assert set(ranking) == {"cosine", "lexical", "importance", "recency",
                            "identity_boost", "echo_penalty", "total"}
    assert ranking["identity_boost"] == 1.5  # identity query boosts it

    # Unknown memory gives 404
    assert (await client.get("/sessions/s7/memories/nope")).status_code == 404


@pytest.mark.asyncio
async def test_auto_session_persisted_on_create(tmp_path: Path):
    """An auto-generated (no fixed id) session survives a restart even before
    its first mutation — persisted at create time."""
    llm, _ = make_llm()
    app1 = create_app(HyperMemConfig(), tmp_path, llm=llm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app1), base_url="http://test"
    ) as c:
        resp = await c.post("/sessions", json={})
        assert resp.status_code == 201
        sid = resp.json()["session_id"]
        assert sid.startswith("session_")

    app2 = create_app(HyperMemConfig(), tmp_path, llm=llm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app2), base_url="http://test"
    ) as c:
        resp = await c.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["total_messages"] == 0


@pytest.mark.asyncio
async def test_concurrent_messages_do_not_lose_writes(client):
    """The per-session lock serializes mutate+persist: 20 concurrent message
    posts must all land (no read-modify-write race)."""
    await client.post("/sessions", json={"session_id": "s8"})

    # Distinct token sets per message — identical wording would be near-dup
    # deduped into one memory, which is not what this test is about.
    tomes = ["griffin", "phoenix", "wyvern", "basilisk", "manticore",
             "chimera", "sphinx", "kraken", "gorgon", "harpy",
             "centaur", "minotaur", "hippogriff", "pegasus", "siren",
             "griffon", "leviathan", "behemoth", "roc", "jackalope"]

    async def post(tome: str):
        resp = await client.post("/sessions/s8/messages", json={
            "role": "user",
            "content": f"I own a library in Corvus where I catalog the {tome} scroll",
        })
        return resp.status_code

    codes = await asyncio.gather(*[post(t) for t in tomes])
    assert all(c == 200 for c in codes)

    # The lock guarantees every message was processed: total_messages counts
    # each add_message, so a lost write would show up here. (Memory count can
    # be < 20 — near-duplicate wording is legitimately deduped.)
    summary = (await client.get("/sessions/s8")).json()
    assert summary["total_messages"] == 20
    mems = (await client.get("/sessions/s8/memories")).json()["memories"]
    assert 1 <= len(mems) <= 20
    assert all("Corvus" in m["content"] for m in mems)


@pytest.mark.asyncio
async def test_invalid_session_id_rejected(client):
    resp = await client.post("/sessions", json={"session_id": "../evil"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_invalid_message_body(client):
    await client.post("/sessions", json={"session_id": "s6"})
    resp = await client.post("/sessions/s6/messages", json={
        "role": "admin",  # not allowed
        "content": "x",
    })
    assert resp.status_code == 422