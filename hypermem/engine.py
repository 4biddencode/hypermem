"""HyperMEM - Core engine."""
import time
import re
import logging
from typing import Optional
from .types import (
    Message, HyperMem, HyperMemState, HyperMemConfig, MemoryType, Persona,
    RecallResult, AddMessageResult, state_to_dict, state_from_dict,
)
from .llm import LLMClient

logger = logging.getLogger("hypermem")

_mem_id_counter = 0


def _next_id() -> str:
    global _mem_id_counter
    _mem_id_counter += 1
    return f"hm_{int(time.time() * 1000)}_{_mem_id_counter}"


def _extract_keywords(text: str) -> list[str]:
    """Extract keywords from text for fallback matching."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9]{4,}", text.lower())
    stopwords = {
        "the", "this", "that", "with", "from", "have", "been", "were",
        "what", "when", "where", "which", "their", "there", "about",
        "would", "could", "should", "after", "before", "into", "over",
        "only", "other", "than", "then", "very", "just", "also", "more",
        "some", "such", "like",
    }
    return list(dict.fromkeys(w for w in words if w not in stopwords))


# ---- Contradiction resolution ----

def _resolve_conflict(existing: HyperMem, new_content: str,
                      new_importance: float) -> tuple[bool, Optional[HyperMem]]:
    """Resolve a conflict between existing memory and new information.

    Args:
        existing: The currently stored memory.
        new_content: The new information that conflicts.
        new_importance: Importance score of the new information.

    Returns:
        (should_replace, replacement_memory):
        - (True, new_mem) → existing should be superseded
        - (False, None) → append both (they coexist)
    """
    # Static facts: new supersedes old (user corrected themselves)
    if existing.memory_type == MemoryType.STATIC:
        return True, HyperMem(
            id=_next_id(),
            content=new_content,
            created_at=time.time(),
            last_accessed_at=time.time(),
            access_count=0,
            keywords=_extract_keywords(new_content),
            importance=new_importance,
            source="auto",
            memory_type=MemoryType.STATIC,
            superseded_by=None,
            pinned=existing.pinned,
        )

    # Temporal: old gets archived, new is current
    if existing.memory_type == MemoryType.TEMPORAL:
        return True, HyperMem(
            id=_next_id(),
            content=new_content,
            created_at=time.time(),
            last_accessed_at=time.time(),
            access_count=0,
            keywords=_extract_keywords(new_content),
            importance=new_importance,
            source="auto",
            memory_type=MemoryType.TEMPORAL,
            superseded_by=None,
            pinned=False,
        )

    # Episodic: append both (multiple events can coexist)
    return False, None


def _find_conflicts(memories: list[HyperMem], new_keywords: list[str],
                    new_content: str) -> list[int]:
    """Find memories that might conflict with new information.

    Looks for memories sharing keywords with the new content.
    """
    conflicts = []
    new_lower = new_content.lower()
    for i, mem in enumerate(memories):
        # Check keyword overlap
        if any(kw in new_lower for kw in mem.keywords):
            # Check if they're about the same topic (name, location, etc.)
            shared = set(kw for kw in mem.keywords if kw in new_lower)
            if len(shared) >= 1:
                conflicts.append(i)
    return conflicts


# ---- Persona isolation ----

def _build_judge_prompt(content: str, persona: Optional[Persona] = None) -> str:
    """Build the judge prompt with persona isolation baked in.

    The prompt explicitly excludes persona fields from being stored as memories.
    """
    persona_block = ""
    if persona:
        persona_block = f"""
Persona context (DO NOT store these as memories — they are already known):
- Name: {persona.name}
- Description: {persona.description}
- Traits: {', '.join(persona.traits)}
- Backstory: {persona.backstory}
"""

    return f"""Message: "{content}"{persona_block}

Extract the key factual information from this message as a short memory.

Rules:
- If the message contains ANY personal information (name, relationship, location, profession, event, preference, plan, object, description), store it with importance 0.5-1.0
- Only return importance=0 for pure greetings, filler, or questions with no self-disclosure
- Be generous — when in doubt, store it
- NEVER store persona information (it's already known)

Return JSON. Example: {{"memory":"User name is Emanuel","keywords":["emanuel","name"],"importance":0.7}}

ENGLISH ONLY."""


# ---- Decay ----

def _apply_decay(mem: HyperMem) -> float:
    """Calculate the effective importance after decay.

    Static memories never decay.
    Episodic memories decay with time and access count.
    Temporal memories are handled by worldIDA.
    """
    if mem.memory_type == MemoryType.STATIC:
        return mem.importance

    if mem.memory_type == MemoryType.EPISODIC:
        hours_since_access = (time.time() - mem.last_accessed_at) / 3600
        access_decay = 1.0 / (1.0 + 0.1 * mem.access_count)
        time_decay = max(0.5, 1.0 - 0.01 * hours_since_access)
        return mem.importance * access_decay * time_decay

    return mem.importance


# ---- Main engine ----

class HyperMEM:
    """HyperMEM - AI memory system that never forgets what matters."""

    def __init__(self, config: Optional[HyperMemConfig] = None,
                 llm: Optional[LLMClient] = None):
        self.config = config or HyperMemConfig()
        self.state = HyperMemState(conversation_id=f"hm_{int(time.time())}")
        self._llm = llm or LLMClient(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            endpoint=self.config.llm_endpoint,
            api_key=self.config.llm_api_key,
        )
        self._world_ida = None
        self._world_ida_store = None

    async def __aenter__(self) -> "HyperMEM":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self):
        """Release the underlying HTTP client."""
        close = getattr(self._llm, "close", None)
        if close is not None:
            await close()

    def _get_ida_store(self):
        if self._world_ida_store is None:
            from .world_ida import WorldIDAStore
            self._world_ida_store = WorldIDAStore()
        return self._world_ida_store

    def set_persona(self, persona: Persona):
        """Set the persona definition. Protected from memory operations."""
        self.state.persona = persona

    def set_world_ida(self, ida) -> None:
        self._world_ida = ida
        store = self._get_ida_store()
        store.set(self.state.conversation_id, ida)

    def get_world_ida(self):
        return self._world_ida

    # ---- Public API ----

    def _is_filler(self, content: str) -> bool:
        return content.strip() in {
            "Ok.", "Hmm.", "I see.", "Yes.", "No.", "Maybe.", "Sure.",
            "Wait.", "Oh.", "Right.", "Let's go.", "Alright.", "Cool.",
            "Nice.", "Good.", "Hey", "Hi", "Hello", "Bye", "Thanks",
        }

    async def add_message(self, role: str, content: str,
                          message_id: Optional[str] = None,
                          memory_type: MemoryType = MemoryType.EPISODIC) -> AddMessageResult:
        """Add a message. Auto-tags, recalls, resolves conflicts."""
        msg = Message(
            id=message_id or _next_id(),
            role=role, content=content, timestamp=time.time(),
        )

        self.state.recent_messages.append(msg)
        if len(self.state.recent_messages) > self.config.max_context_messages:
            if self.config.max_context_messages > 0:
                self.state.recent_messages = self.state.recent_messages[-self.config.max_context_messages:]
            else:
                self.state.recent_messages = []
        self.state.total_messages += 1

        if self._is_filler(content):
            return AddMessageResult(self.state, None, None)

        tagged = None
        recalled = None

        # 1. Recall check
        if self.state.active:
            relevant_indices = await self._llm.find_relevant(content, self.state.active)
            if relevant_indices:
                relevant_mems = []
                for idx in relevant_indices:
                    if idx < len(self.state.active):
                        mem = self.state.active[idx]
                        mem.last_accessed_at = time.time()
                        mem.access_count += 1
                        relevant_mems.append(mem)
                if relevant_mems:
                    recalled = RecallResult(relevant_mems, f"Recalled {len(relevant_mems)} memories")

        # 2. Importance scoring + extraction (user messages only by default)
        if self.config.auto_tagging and role in self.config.auto_tag_roles:
            prompt = _build_judge_prompt(content, self.state.persona)
            result = await self._llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=100,
            )

            if result:
                parsed = self._parse_judge_result(result)
                if parsed:
                    try:
                        importance = float(parsed.get("importance", 0) or 0)
                    except (TypeError, ValueError):
                        importance = 0.0
                    memory_text = parsed.get("memory", "")
                    keywords = parsed.get("keywords", [])

                    if memory_text and memory_text.strip():
                        # If the judge extracted a fact, store it. Importance
                        # only ranks it — never below the tagging threshold.
                        importance = max(importance, self.config.auto_tag_threshold)
                        mem_content = memory_text if len(memory_text) > 5 and memory_text != "name" else content
                        new_keywords = keywords or _extract_keywords(content)

                        # Check for conflicts with existing memories
                        conflict_indices = _find_conflicts(self.state.active, new_keywords, mem_content)
                        if conflict_indices:
                            for ci in conflict_indices:
                                existing = self.state.active[ci]
                                should_replace, replacement = _resolve_conflict(existing, mem_content, importance)
                                if should_replace and replacement:
                                    # Mark old as superseded
                                    existing.superseded_by = replacement.id
                                    # Archive the old one
                                    self.state.archive.append(existing)
                                    # Replace in active
                                    self.state.active[ci] = replacement
                                    tagged = replacement
                                    break
                        else:
                            # No conflict — create new memory
                            tagged = HyperMem(
                                id=_next_id(),
                                content=mem_content,
                                created_at=time.time(),
                                last_accessed_at=time.time(),
                                access_count=0,
                                keywords=new_keywords,
                                importance=importance,
                                source="auto",
                                memory_type=memory_type,
                                pinned=False,
                            )
                            self.state.active.append(tagged)

        # 3. Archive if over limit (respecting decay)
        if len(self.state.active) > self.config.max_active_memories:
            scored = [(m, _apply_decay(m)) for m in self.state.active]
            scored.sort(key=lambda x: (not x[0].pinned, x[1]))
            to_archive = scored[:len(scored) - self.config.max_active_memories]
            for mem, _ in to_archive:
                self.state.archive.append(mem)
                self.state.active.remove(mem)

        return AddMessageResult(self.state, tagged, recalled)

    def _parse_judge_result(self, raw: str) -> Optional[dict]:
        """Parse LLM judge response, handling markdown fences.

        Returns None if the response is not a JSON object (e.g. a list or
        malformed output) — callers treat None as "nothing to store".
        """
        import json
        try:
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, IndexError):
            return None

    async def update_world_ida(self, user_msg: str, ai_msg: str,
                                persona_context: Optional[str] = None) -> None:
        from .world_ida import update_world_ida as _update, scene_transition_summary

        old_ida = self._world_ida
        new_ida = await _update(
            previous_ida=old_ida,
            last_user_message=user_msg,
            last_ai_response=ai_msg,
            persona_context=persona_context or (
                self.state.persona.description if self.state.persona else None
            ),
            llm_complete=self._llm.complete,
        )

        if old_ida is not None and new_ida.meta.scene_changed:
            summary = scene_transition_summary(old_ida)
            self.remember(summary)

        self.set_world_ida(new_ida)

    def remember(self, content: str, memory_type: MemoryType = MemoryType.STATIC) -> HyperMem:
        """Explicitly tell HyperMEM to remember something."""
        mem = HyperMem(
            id=_next_id(),
            content=content,
            created_at=time.time(),
            last_accessed_at=time.time(),
            access_count=0,
            keywords=_extract_keywords(content),
            importance=1.0,
            source="user",
            memory_type=memory_type,
            pinned=True,
        )
        self.state.active.append(mem)
        return mem

    async def recall(self, query: str) -> RecallResult:
        """Search all memories for relevance to a query."""
        all_mems = self.state.active + self.state.archive
        if not all_mems:
            return RecallResult([], "No memories stored")

        relevant_indices = await self._llm.find_relevant(query, all_mems)
        relevant = []
        for idx in relevant_indices:
            if idx < len(all_mems):
                mem = all_mems[idx]
                for active_mem in self.state.active:
                    if active_mem.id == mem.id:
                        active_mem.last_accessed_at = time.time()
                        active_mem.access_count += 1
                        break
                relevant.append(mem)

        return RecallResult(relevant, f"Found {len(relevant)} relevant memories")

    async def get_context(self, current_message: str) -> str:
        """Build context prompt with world state + relevant memories."""
        parts = []

        if hasattr(self, '_world_ida') and self._world_ida is not None:
            from .world_ida import world_ida_to_context_string
            ctx_str = world_ida_to_context_string(self._world_ida)
            if ctx_str:
                parts.append(f"[WORLD STATE]\n{ctx_str}\n[/WORLD STATE]")

        recall = await self.recall(current_message)
        if recall.relevant:
            mem_block = "[RELEVANT MEMORIES]\n" + "\n".join(
                f"- {m.content} (importance: {round(m.importance * 100)}%)"
                for m in recall.relevant
            ) + "\n[/RELEVANT MEMORIES]"
            parts.append(mem_block)

        if self.state.recent_messages:
            chat = "\n".join(
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
                for m in self.state.recent_messages[-10:]
            )
            parts.append(chat)

        return "\n\n".join(parts)

    def memories(self) -> list[dict]:
        return [
            {
                "id": m.id,
                "content": m.content,
                "keywords": m.keywords,
                "importance": _apply_decay(m),
                "memory_type": m.memory_type.value,
                "source": m.source,
                "pinned": m.pinned,
                "access_count": m.access_count,
                "superseded_by": m.superseded_by,
                "created_at": m.created_at,
            }
            for m in self.state.active + self.state.archive
        ]

    # ---- Persistence ----

    def to_dict(self) -> dict:
        """Serialize the full engine state (memories + worldIDA) to a dict."""
        from .world_ida import _ida_to_dict
        data = state_to_dict(self.state)
        if self._world_ida is not None:
            data["world_ida"] = _ida_to_dict(self._world_ida)
        if self._world_ida_store is not None:
            data["world_ida_store"] = self._world_ida_store.to_dict()
        return data

    def from_dict(self, data: dict) -> None:
        """Restore the full engine state from a dict."""
        from .world_ida import _ida_from_dict, WorldIDAStore
        self.state = state_from_dict(data)
        if "world_ida" in data:
            self._world_ida = _ida_from_dict(data["world_ida"])
        if "world_ida_store" in data:
            store = WorldIDAStore()
            store.from_dict(data["world_ida_store"])
            self._world_ida_store = store

    def save(self, path: str):
        """Persist full engine state to JSON (atomic write)."""
        import json
        import os
        import tempfile

        target = os.path.abspath(path)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self, path: str):
        """Restore full engine state from a JSON file."""
        import json
        with open(path, encoding="utf-8") as f:
            self.from_dict(json.load(f))
