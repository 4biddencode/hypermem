"""HyperMEM REST server.

Exposes the engine over HTTP so any application (chat bots, games, CLI
tools) can use persistent AI memory without writing Python.

Run with ``python -m hypermem.server`` or the ``hypermem-server`` script.

Requires the optional extra: ``pip install hypermem[server]``.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .engine import HyperMEM
from .types import HyperMemConfig, MemoryType, Persona

logger = logging.getLogger("hypermem.server")

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The HyperMEM server requires the optional dependencies. "
        "Install with: pip install 'hypermem[server]'"
    ) from e


# ---------------------------------------------------------------------------
# Session store — in-memory cache + JSON persistence on disk
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionStore:
    """Owns HyperMEM instances, one per conversation, persisted as JSON."""

    def __init__(self, config: HyperMemConfig, data_dir: Path, llm=None):
        self._config = config
        self._data_dir = data_dir
        self._llm = llm  # shared LLM client or None → one per session
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, HyperMEM] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        for path in self._data_dir.glob("*.json"):
            sid = path.stem
            hm = HyperMEM(self._config, llm=self._llm)
            try:
                hm.load(str(path))
                self._sessions[sid] = hm
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning("skipping corrupt session file %s", path)

    def _path(self, session_id: str) -> Path:
        return self._data_dir / f"{session_id}.json"

    def create(self, session_id: Optional[str] = None) -> HyperMEM:
        if session_id and not _ID_RE.match(session_id):
            raise HTTPException(400, "session_id must match [A-Za-z0-9_-]{1,64}")
        if session_id and session_id in self._sessions:
            raise HTTPException(409, f"session '{session_id}' already exists")
        sid = session_id or f"session_{int(time.time() * 1000)}"
        hm = HyperMEM(self._config, llm=self._llm)
        hm.state.conversation_id = sid
        if session_id is None:
            hm.state.recent_messages = []
        # persist immediately if a fixed id was requested
        if session_id is not None:
            self._write(sid, hm)
        with self._lock:
            self._sessions[sid] = hm
        return hm

    def get(self, session_id: str) -> HyperMEM:
        with self._lock:
            hm = self._sessions.get(session_id)
        if hm is None:
            raise HTTPException(404, f"session '{session_id}' not found")
        return hm

    def delete(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise HTTPException(404, f"session '{session_id}' not found")
            del self._sessions[session_id]
        self._path(session_id).unlink(missing_ok=True)

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def _write(self, session_id: str, hm: HyperMEM) -> None:
        hm.save(str(self._path(session_id)))

    def persist(self, session_id: str, hm: HyperMEM) -> None:
        self._write(session_id, hm)


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def _mem_to_dict(mem) -> dict:
    d = asdict(mem)
    d["memory_type"] = mem.memory_type.value
    d["effective_importance"] = round(mem.importance, 4)
    return d


def _recall_to_dict(recall) -> dict:
    return {
        "relevant": [_mem_to_dict(m) for m in recall.relevant],
        "relevance": recall.relevance,
    }


def _summary(hm: HyperMEM) -> dict:
    return {
        "conversation_id": hm.state.conversation_id,
        "total_messages": hm.state.total_messages,
        "active_memories": len(hm.state.active),
        "archived_memories": len(hm.state.archive),
    }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MessageIn(BaseModel):
    role: str = Field(default="user", pattern="^(user|assistant|system)$")
    content: str
    memory_type: str = Field(default="episodic", pattern="^(static|episodic|temporal)$")


class RememberIn(BaseModel):
    content: str
    memory_type: str = Field(default="static", pattern="^(static|episodic|temporal)$")


class PersonaIn(BaseModel):
    name: str = ""
    description: str = ""
    traits: list[str] = []
    backstory: str = ""
    boundaries: list[str] = []


class WorldIDAIn(BaseModel):
    user_msg: str
    ai_msg: str = ""
    persona_context: Optional[str] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[HyperMemConfig] = None,
               data_dir: Optional[Path] = None, llm=None) -> FastAPI:
    cfg = config or HyperMemConfig()
    store = SessionStore(cfg, data_dir or Path("./.hypermem_data"), llm=llm)
    app = FastAPI(
        title="HyperMEM",
        version="0.1.0",
        description="AI memory system — never forgets what matters.",
    )
    app.state.store = store

    @app.get("/health")
    async def health():
        return {"status": "ok", "sessions": len(store.list_ids())}

    # ---- Sessions ----

    @app.post("/sessions", status_code=201)
    async def create_session(body: dict | None = None):
        body = body or {}
        sid = body.get("session_id")
        hm = store.create(sid)
        return {"session_id": hm.state.conversation_id, **_summary(hm)}

    @app.get("/sessions")
    async def list_sessions():
        return {"sessions": [_summary(store.get(s)) for s in store.list_ids()]}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return _summary(store.get(session_id))

    @app.delete("/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        store.delete(session_id)

    # ---- Messages & memory ----

    @app.post("/sessions/{session_id}/messages")
    async def add_message(session_id: str, msg: MessageIn):
        hm = store.get(session_id)
        result = await hm.add_message(
            msg.role, msg.content,
            memory_type=MemoryType(msg.memory_type),
        )
        store.persist(session_id, hm)
        return {
            "tagged": _mem_to_dict(result.tagged) if result.tagged else None,
            "recalled": _recall_to_dict(result.recalled) if result.recalled else None,
            **_summary(hm),
        }

    @app.post("/sessions/{session_id}/remember")
    async def remember(session_id: str, body: RememberIn):
        hm = store.get(session_id)
        mem = hm.remember(body.content, MemoryType(body.memory_type))
        store.persist(session_id, hm)
        return {"memory": _mem_to_dict(mem), **_summary(hm)}

    @app.get("/sessions/{session_id}/recall")
    async def recall(session_id: str, query: str = ""):
        if not query:
            raise HTTPException(400, "query parameter is required")
        hm = store.get(session_id)
        result = await hm.recall(query)
        store.persist(session_id, hm)
        return _recall_to_dict(result)

    @app.get("/sessions/{session_id}/context")
    async def get_context(session_id: str, message: str = ""):
        hm = store.get(session_id)
        ctx = await hm.get_context(message)
        store.persist(session_id, hm)
        return PlainTextResponse(ctx)

    @app.get("/sessions/{session_id}/memories")
    async def memories(session_id: str):
        hm = store.get(session_id)
        return {"memories": hm.memories()}

    @app.put("/sessions/{session_id}/persona")
    async def set_persona(session_id: str, body: PersonaIn):
        hm = store.get(session_id)
        hm.set_persona(Persona(**body.model_dump()))
        store.persist(session_id, hm)
        return {"persona_set": True}

    # ---- World state ----

    @app.get("/sessions/{session_id}/world-ida")
    async def world_ida(session_id: str):
        hm = store.get(session_id)
        ida = hm.get_world_ida()
        if ida is None:
            return {"world_ida": None}
        from .world_ida import _ida_to_dict
        return {"world_ida": _ida_to_dict(ida)}

    @app.post("/sessions/{session_id}/world-ida/update")
    async def update_world_ida(session_id: str, body: WorldIDAIn):
        hm = store.get(session_id)
        await hm.update_world_ida(body.user_msg, body.ai_msg, body.persona_context)
        store.persist(session_id, hm)
        from .world_ida import _ida_to_dict
        return {"world_ida": _ida_to_dict(hm.get_world_ida())}

    # ---- Export / import ----

    @app.get("/sessions/{session_id}/state")
    async def get_state(session_id: str):
        hm = store.get(session_id)
        return JSONResponse(hm.to_dict())

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_config(args) -> HyperMemConfig:
    return HyperMemConfig(
        auto_tag_threshold=args.auto_tag_threshold,
        max_active_memories=args.max_active_memories,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_endpoint=args.llm_endpoint,
        llm_api_key=args.llm_api_key,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="hypermem-server", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", default=".hypermem_data")
    parser.add_argument("--llm-provider", default="auto",
                        choices=["auto", "ollama", "openai", "anthropic"],
                        help="auto detects from model/endpoint (default)")
    parser.add_argument("--llm-model", default="qwen2.5:7b")
    parser.add_argument("--llm-endpoint", default="http://localhost:11434")
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--auto-tag-threshold", type=float, default=0.4)
    parser.add_argument("--max-active-memories", type=int, default=100)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config = _build_config(args)
    app = create_app(config, Path(args.data_dir))

    print(f"HyperMEM server listening on http://{args.host}:{args.port}")
    print(f"  LLM:      {args.llm_provider} / {args.llm_model}")
    print(f"  Data dir: {os.path.abspath(args.data_dir)}")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()