"""
worldIDA — a compact, always-current "world state" object for roleplay.

Separate from long-term memory (which is searched/ranked and grows over time),
worldIDA is ONE small object per session, fully overwritten every turn,
never searched — always injected in full into context.
"""

import copy
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger("hypermem.world_ida")


# ---------------------------------------------------------------------------
# 1. SCHEMA
# ---------------------------------------------------------------------------

@dataclass
class Scene:
    location: str = ""
    sub_location: Optional[str] = None
    time_of_day: str = ""
    ambient_conditions: Optional[str] = None
    ongoing_action: str = ""
    last_completed_action: Optional[str] = None
    interrupted_action: Optional[str] = None


@dataclass
class UserState:
    physical_state: str = ""


@dataclass
class CharacterState:
    physical_state: str = ""
    appearance: Optional[str] = None
    mood: str = ""
    mood_trajectory: Optional[str] = None
    energy_level: Optional[str] = None


@dataclass
class Relationship:
    stage: str = ""
    trust_level: Optional[str] = None
    unresolved_thread: Optional[str] = None


@dataclass
class Meta:
    scene_changed: bool = False
    turn_count_in_scene: int = 0
    last_updated_turn_index: int = 0
    confidence: float = 1.0


@dataclass
class WorldIDA:
    scene: Scene = field(default_factory=Scene)
    user: UserState = field(default_factory=UserState)
    character: CharacterState = field(default_factory=CharacterState)
    relationship: Relationship = field(default_factory=Relationship)
    meta: Meta = field(default_factory=Meta)


# ---------------------------------------------------------------------------
# 2. UPDATE FUNCTION
# ---------------------------------------------------------------------------

def _build_update_prompt(previous_ida: Optional[WorldIDA],
                         last_user_message: str,
                         last_ai_response: str,
                         persona_context: Optional[str] = None) -> str:
    """Build the system prompt for the worldIDA update LLM call."""
    if previous_ida is not None:
        previous_json = json.dumps(_ida_to_dict(previous_ida))
    else:
        previous_json = "null (first turn — initialize the scene from the exchange)"

    ctx = persona_context or "No persona context provided."

    return f"""You are tracking the current state of a roleplay scene. Output a compact JSON object with ONLY the fields that changed.

SCHEMA (field names):
{{
  "scene": {{"location": "", "sub_location": null, "time_of_day": "", "ambient_conditions": null, "ongoing_action": "", "last_completed_action": null, "interrupted_action": null}},
  "user": {{"physical_state": ""}},
  "character": {{"physical_state": "", "appearance": null, "mood": "", "mood_trajectory": null, "energy_level": null}},
  "relationship": {{"stage": "", "trust_level": null, "unresolved_thread": null}},
  "meta": {{"scene_changed": false, "turn_count_in_scene": 0, "last_updated_turn_index": 0, "confidence": 1.0}}
}}

RULES:
1. Output ONLY fields that changed or are new — omit unchanged fields entirely. An unchanged turn can be just {{"meta": {{"turn_count_in_scene": 5}}}}.
2. scene_changed=true ONLY if location, time_of_day, or fundamental activity changed. Natural continuation is NOT a scene change.
3. If scene_changed=true, reset turn_count_in_scene to 0. Otherwise increment by 1.
4. mood and physical_state are snapshots after the latest exchange.
5. If exchange contradicts previous state, trust the latest exchange but lower confidence.
6. Never modify persona-level traits.
7. Output ONLY valid JSON. No preamble, no explanation.

Previous state: {previous_json}
Persona: {ctx}
User: {last_user_message}
Character: {last_ai_response}

Output JSON:"""


def _ida_from_dict(data: dict) -> WorldIDA:
    """Build WorldIDA from a dict (partial — missing fields get defaults)."""
    return WorldIDA(
        scene=Scene(**{k: data.get("scene", {}).get(k) for k in Scene.__dataclass_fields__}),
        user=UserState(**{k: data.get("user", {}).get(k) for k in UserState.__dataclass_fields__}),
        character=CharacterState(**{k: data.get("character", {}).get(k) for k in CharacterState.__dataclass_fields__}),
        relationship=Relationship(**{k: data.get("relationship", {}).get(k) for k in Relationship.__dataclass_fields__}),
        meta=Meta(**{k: data.get("meta", {}).get(k) for k in Meta.__dataclass_fields__}),
    )


def _ida_to_dict(ida: WorldIDA) -> dict:
    """Serialize WorldIDA to a dict, omitting None values."""
    d = asdict(ida)
    # Strip None values for cleaner JSON
    def _strip_none(obj):
        if isinstance(obj, dict):
            return {k: _strip_none(v) for k, v in obj.items() if v is not None}
        return obj
    return _strip_none(d)


def _validate_ida(data: dict) -> bool:
    """Validate that the LLM response is a usable WorldIDA update.

    Accepts partial output (only changed fields) — at minimum a meta block
    or one known section. Unchanged fields are carried over from the
    previous state during the merge.
    """
    if not isinstance(data, dict):
        return False
    known = {"scene", "user", "character", "relationship", "meta"}
    provided = [k for k in data if k in known]
    if not provided:
        return False
    for k in provided:
        if not isinstance(data[k], dict):
            return False
    return True


async def update_world_ida(
    previous_ida: Optional[WorldIDA],
    last_user_message: str,
    last_ai_response: str,
    persona_context: Optional[str] = None,
    llm_complete=None,
) -> WorldIDA:
    """
    Update worldIDA with the latest exchange.

    Args:
        previous_ida: Previous state, or None to initialize fresh.
        last_user_message: The user's latest message.
        last_ai_response: The AI's latest response.
        persona_context: Optional persona/character description.
        llm_complete: Async function(messages, **kwargs) -> str | None.
                      The model is already configured in the LLM client.

    Returns:
        Updated WorldIDA. On failure, returns previous_ida unchanged (or fresh).
    """
    # Track consecutive failures for this call
    failure_count = getattr(update_world_ida, "_failure_count", 0)

    prompt = _build_update_prompt(previous_ida, last_user_message, last_ai_response, persona_context)

    try:
        if llm_complete is None:
            # Can't update without an LLM
            logger.warning("worldIDA: no llm_complete provided, returning previous state")
            update_world_ida._failure_count = failure_count + 1
            return previous_ida or WorldIDA()

        result = await llm_complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        if not result:
            logger.warning("worldIDA: LLM returned None")
            update_world_ida._failure_count = failure_count + 1
            return previous_ida or WorldIDA()

        # Strip markdown fences if present
        cleaned = result.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        parsed = json.loads(cleaned)

        if not _validate_ida(parsed):
            logger.warning(f"worldIDA: validation failed, raw: {cleaned[:200]}")
            update_world_ida._failure_count = failure_count + 1
            return previous_ida or WorldIDA()

        # Success — reset failure counter
        update_world_ida._failure_count = 0

        # Merge partial output over the previous state, so unchanged fields
        # survive a compact "only what changed" response.
        if previous_ida is not None:
            base = _ida_to_dict(previous_ida)
            for section, fields in parsed.items():
                if isinstance(fields, dict):
                    base.setdefault(section, {}).update(fields)
            new_ida = _ida_from_dict(base)
        else:
            new_ida = _ida_from_dict(parsed)

        return new_ida

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"worldIDA: parse error: {e}")
        update_world_ida._failure_count = failure_count + 1

        # 3+ consecutive failures → log error
        if update_world_ida._failure_count >= 3:
            logger.error("worldIDA: 3+ consecutive failures, returning minimal state")

        return previous_ida or WorldIDA()


# ---------------------------------------------------------------------------
# 3. CONTEXT INJECTION
# ---------------------------------------------------------------------------

def world_ida_to_context_string(ida: WorldIDA) -> str:
    """
    Compact natural language representation, ~80 tokens, omitting null fields.

    Example:
        "Current scene: corner booth at a restaurant, evening. Elena is
        leaning forward, telling a story about her sister — mood: wistful,
        slightly tipsy. You are seated, holding a wine glass. Relationship:
        flirtatious, early stage."
    """
    parts = []

    # Scene
    scene_parts = []
    if ida.scene.location:
        scene_parts.append(ida.scene.location)
    if ida.scene.sub_location:
        scene_parts.append(ida.scene.sub_location)
    if ida.scene.time_of_day:
        scene_parts.append(ida.scene.time_of_day)
    if scene_parts:
        s = f"Current scene: {', '.join(scene_parts)}."
        if ida.scene.ambient_conditions:
            s += f" {ida.scene.ambient_conditions}."
        parts.append(s)

    # Character
    char_parts = []
    if ida.character.physical_state:
        char_parts.append(ida.character.physical_state)
    if ida.character.mood:
        mood = f"mood: {ida.character.mood}"
        if ida.character.mood_trajectory:
            mood += f", {ida.character.mood_trajectory}"
        char_parts.append(mood)
    if ida.character.energy_level:
        char_parts.append(f"energy: {ida.character.energy_level}")
    if char_parts:
        parts.append("Character is " + ". ".join(char_parts) + ".")

    # User
    if ida.user.physical_state:
        parts.append(f"You are {ida.user.physical_state}.")

    # Relationship
    rel_parts = []
    if ida.relationship.stage:
        rel_parts.append(ida.relationship.stage)
    if ida.relationship.trust_level:
        rel_parts.append(f"trust: {ida.relationship.trust_level}")
    if ida.relationship.unresolved_thread:
        rel_parts.append(f"unresolved: {ida.relationship.unresolved_thread}")
    if rel_parts:
        parts.append("Relationship: " + ", ".join(rel_parts) + ".")

    # Ongoing action
    if ida.scene.ongoing_action:
        parts.append(f"Currently: {ida.scene.ongoing_action}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# 4. SCENE TRANSITION → LONG-TERM MEMORY SNAPSHOT
# ---------------------------------------------------------------------------

def scene_transition_summary(old_ida: WorldIDA) -> str:
    """Build a summary of the old scene for storage in long-term memory."""
    parts = []
    if old_ida.scene.location:
        parts.append(f"at {old_ida.scene.location}")
    if old_ida.relationship.stage:
        parts.append(f"relationship: {old_ida.relationship.stage}")
    if old_ida.relationship.unresolved_thread:
        parts.append(f"unresolved: {old_ida.relationship.unresolved_thread}")
    if old_ida.scene.ongoing_action:
        parts.append(f"was: {old_ida.scene.ongoing_action}")
    return f"Scene ended: {', '.join(parts)}." if parts else "Scene ended."


# ---------------------------------------------------------------------------
# 5. STORAGE (in-memory per session, persist-ready)
# ---------------------------------------------------------------------------

class WorldIDAStore:
    """In-memory store for worldIDA objects, one per session."""

    def __init__(self):
        self._store: dict[str, WorldIDA] = {}
        self._history: dict[str, list[WorldIDA]] = {}  # last 3-5 versions for debugging

    def get(self, session_id: str) -> Optional[WorldIDA]:
        return self._store.get(session_id)

    def set(self, session_id: str, ida: WorldIDA):
        # Keep history for debugging
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append(copy.deepcopy(ida))
        if len(self._history[session_id]) > 5:
            self._history[session_id] = self._history[session_id][-5:]

        self._store[session_id] = ida

    def get_history(self, session_id: str, limit: int = 3) -> list[WorldIDA]:
        return self._history.get(session_id, [])[-limit:]

    def diff_versions(self, session_id: str, version_a: int = -2,
                      version_b: int = -1) -> Optional[dict]:
        """
        Diff any two versions from history (default: last two).

        Args:
            session_id: Session to inspect.
            version_a: Index from end (-2 = second-to-last, -1 = last, etc.)
            version_b: Index from end.

        Returns:
            Dict of changes, or None if versions aren't available.
        """
        history = self._history.get(session_id, [])
        if len(history) < max(abs(version_a), abs(version_b)):
            return None

        a = history[version_a]
        b = history[version_b]
        return self._diff(a, b)

    @staticmethod
    def _diff(before: WorldIDA, after: WorldIDA) -> dict:
        """Return structured diff between two WorldIDA states."""
        b = _ida_to_dict(before)
        a = _ida_to_dict(after)
        changes = {}

        for section in ["scene", "user", "character", "relationship", "meta"]:
            b_sec = b.get(section, {})
            a_sec = a.get(section, {})
            sec_changes = {}
            for k in set(b_sec.keys()) | set(a_sec.keys()):
                bv = b_sec.get(k)
                av = a_sec.get(k)
                if bv != av:
                    sec_changes[k] = {"from": bv, "to": av}
            if sec_changes:
                changes[section] = sec_changes

        return changes

    def to_dict(self) -> dict:
        """Serialize all stores for persistence."""
        return {
            "current": {k: _ida_to_dict(v) for k, v in self._store.items()},
            "history": {k: [_ida_to_dict(h) for h in v] for k, v in self._history.items()},
        }

    def from_dict(self, data: dict):
        """Load stores from dict."""
        self._store = {k: _ida_from_dict(v) for k, v in data.get("current", {}).items()}
        self._history = {
            k: [_ida_from_dict(h) for h in v]
            for k, v in data.get("history", {}).items()
        }
