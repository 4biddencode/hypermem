"""HyperMEM - Core type definitions."""
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


class MemoryType(str, Enum):
    """Type of memory, determines decay and conflict behavior."""
    STATIC = "static"       # Facts that don't change (name, backstory). Never decay.
    EPISODIC = "episodic"   # Events that happened. Decay over time/access count.
    TEMPORAL = "temporal"   # Current state (mood, location). worldIDA handles this.


@dataclass
class Message:
    """A single message in a conversation."""
    id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float


@dataclass
class HyperMem:
    """A memory that HyperMEM has tagged as important."""
    id: str
    content: str
    created_at: float
    last_accessed_at: float
    access_count: int
    keywords: list[str]
    importance: float
    source: str  # "user" | "auto"
    memory_type: MemoryType = MemoryType.EPISODIC
    superseded_by: Optional[str] = None  # ID of memory that replaced this one
    trigger: Optional[str] = None
    pinned: bool = False
    subject: str = ""                    # short entity this fact is about ("user", "lyra", "vault password")
    embedding: Optional[list] = None     # semantic vector (list[float]) for recall
    source_message_id: Optional[str] = None  # the chat message this memory came from
    consolidated_from: Optional[list] = None  # IDs of episodic memories merged into this one


@dataclass
class Persona:
    """A persona/character definition. NEVER modified by memory operations.

    Persona data is isolated from memory — extraction prompts explicitly
    exclude persona fields, and worldIDA update prompts have a hard rule
    against modifying persona-level traits.
    """
    name: str = ""
    description: str = ""
    traits: list[str] = field(default_factory=list)
    backstory: str = ""
    boundaries: list[str] = field(default_factory=list)


@dataclass
class RecallResult:
    """Result of a recall operation."""
    relevant: list[HyperMem]
    relevance: str


@dataclass
class AddMessageResult:
    """Result of adding a message to HyperMEM."""
    state: 'HyperMemState'
    tagged: Optional[HyperMem]
    recalled: Optional[RecallResult]


@dataclass
class HyperMemState:
    """The full memory state of a conversation."""
    conversation_id: str
    active: list[HyperMem] = field(default_factory=list)
    archive: list[HyperMem] = field(default_factory=list)
    recent_messages: list[Message] = field(default_factory=list)
    total_messages: int = 0
    persona: Optional[Persona] = None


@dataclass
class HyperMemConfig:
    """Configuration for HyperMEM behavior."""
    max_context_messages: int = 20
    auto_tag_threshold: float = 0.4
    max_active_memories: int = 100
    auto_tagging: bool = True
    auto_tag_roles: tuple = ("user",)  # message roles the judge processes
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_endpoint: str = "http://localhost:11434"
    llm_api_key: Optional[str] = None
    # ---- recall ----
    embedding_provider: str = "auto"   # "auto" | "ollama" | "openai" | "none"
    embedding_model: Optional[str] = None      # e.g. "nomic-embed-text"
    embedding_endpoint: Optional[str] = None   # override embed base URL
    embedding_api_key: Optional[str] = None
    recall_use_llm: bool = False       # when embeddings are on, also run the LLM rank
    max_recall_tokens: int = 300       # context-window budget for recalled memories
    search_archive: bool = False       # whether decay-archived memories stay recallable
    recall_ambiguity_gap: float = 0.7  # leader-to-runner-up score gap below which the
                                       # LLM adjudicates near-tied candidates (0 = never)
    # ---- ingestion / lifecycle ----
    max_memory_chars: int = 1000       # verbatim content cap
    consolidation_threshold: int = 6   # episodic memories per subject before consolidation (0=off)
    consolidation_interval: int = 20   # min messages between consolidation runs


def state_to_dict(state: HyperMemState) -> dict:
    """Serialize state to JSON-compatible dict."""
    result = {
        "conversation_id": state.conversation_id,
        "active": [asdict(m) for m in state.active],
        "archive": [asdict(m) for m in state.archive],
        "recent_messages": [asdict(m) for m in state.recent_messages],
        "total_messages": state.total_messages,
    }
    if state.persona:
        result["persona"] = asdict(state.persona)
    return result


def _build_from_fields(cls, data: dict, defaults: dict) -> object:
    """Build a dataclass from a dict, tolerating stale or hand-edited JSON.

    Only known fields are passed through; unknown keys are ignored and
    missing ones fall back to the dataclass default.
    """
    from dataclasses import fields
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    return cls(**kwargs)


def state_from_dict(data: dict) -> HyperMemState:
    """Deserialize state from dict.

    Tolerant of pre-rework or hand-edited JSON: unknown keys are ignored and
    missing fields fall back to defaults, so a memory saved by a different
    version never crashes a load.
    """
    mem_defaults = {
        "id": "unknown", "content": "", "created_at": 0.0,
        "last_accessed_at": 0.0, "access_count": 0, "keywords": [],
        "importance": 0.0, "source": "user",
    }

    def _mem(m: dict) -> HyperMem:
        mt = m.get("memory_type", "episodic")
        if not isinstance(mt, MemoryType):
            try:
                mt = MemoryType(str(mt).lower())
            except ValueError:
                mt = MemoryType.EPISODIC
        m = dict(m)
        m["memory_type"] = mt
        return _build_from_fields(HyperMem, m, mem_defaults)

    active = [_mem(m) for m in data.get("active", []) if isinstance(m, dict)]
    archive = [_mem(m) for m in data.get("archive", []) if isinstance(m, dict)]

    persona = None
    if isinstance(data.get("persona"), dict):
        persona = _build_from_fields(Persona, data["persona"], {})

    return HyperMemState(
        conversation_id=data.get("conversation_id", ""),
        active=active,
        archive=archive,
        recent_messages=[
            _build_from_fields(Message, m, {
                "id": "", "role": "user", "content": "", "timestamp": 0.0,
            })
            for m in data.get("recent_messages", [])
            if isinstance(m, dict)
        ],
        total_messages=data.get("total_messages", 0),
        persona=persona,
    )
