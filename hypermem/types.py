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


def state_from_dict(data: dict) -> HyperMemState:
    """Deserialize state from dict."""
    active = []
    for m in data.get("active", []):
        m["memory_type"] = MemoryType(m.get("memory_type", "episodic"))
        active.append(HyperMem(**m))

    archive = []
    for m in data.get("archive", []):
        m["memory_type"] = MemoryType(m.get("memory_type", "episodic"))
        archive.append(HyperMem(**m))

    persona = None
    if data.get("persona"):
        persona = Persona(**data["persona"])

    return HyperMemState(
        conversation_id=data["conversation_id"],
        active=active,
        archive=archive,
        recent_messages=[Message(**m) for m in data.get("recent_messages", [])],
        total_messages=data.get("total_messages", 0),
        persona=persona,
    )
