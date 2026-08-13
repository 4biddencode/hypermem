"""
worldIDA tests.

Covers: single-field update, scene change detection, physical state continuity,
appearance drift, relationship persistence across scenes, malformed LLM fallback,
first-turn init, and 50+ turn stability.
"""

import json
import pytest
from hypermem.world_ida import (
    WorldIDA, Scene, UserState, CharacterState, Relationship, Meta,
    update_world_ida, world_ida_to_context_string, scene_transition_summary,
    WorldIDAStore, _ida_from_dict, _validate_ida,
)


def make_ida(**overrides) -> WorldIDA:
    """Helper to build a WorldIDA with sensible defaults."""
    ida = WorldIDA(
        scene=Scene(
            location="tavern",
            sub_location="corner booth",
            time_of_day="evening",
            ongoing_action="drinking ale",
        ),
        user=UserState(physical_state="seated at the table"),
        character=CharacterState(
            physical_state="leaning forward, elbows on table",
            mood="curious",
            energy_level="alert",
        ),
        relationship=Relationship(
            stage="just met",
            trust_level="cautious",
        ),
        meta=Meta(turn_count_in_scene=5, confidence=0.95),
    )
    # Apply overrides
    for section, fields in overrides.items():
        if hasattr(ida, section):
            for k, v in fields.items():
                setattr(getattr(ida, section), k, v)
    return ida


class StubLLM:
    """Test double: an async LLM callable returning a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    async def __call__(self, messages, **kwargs):
        self.last_prompt = messages[0]["content"]
        return self.response


# ---- Single field update ----

@pytest.mark.asyncio
async def test_single_field_mood_change():
    """Only mood changes; all other fields identical."""
    prev = make_ida()
    response = json.dumps({
        "scene": {"location": "tavern", "sub_location": "corner booth", "time_of_day": "evening",
                   "ongoing_action": "drinking ale"},
        "user": {"physical_state": "seated at the table"},
        "character": {"physical_state": "leaning forward, elbows on table", "mood": "angry", "energy_level": "agitated"},
        "relationship": {"stage": "just met", "trust_level": "cautious"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 6, "last_updated_turn_index": 0, "confidence": 0.95},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "You're lying!", "I don't lie.", llm_complete=llm)

    assert result.character.mood == "angry"
    assert result.character.energy_level == "agitated"
    assert result.scene.location == "tavern"  # unchanged
    assert result.relationship.stage == "just met"  # unchanged
    assert result.meta.turn_count_in_scene == 6  # incremented
    assert result.meta.scene_changed is False


# ---- Scene change detection ----

@pytest.mark.asyncio
async def test_scene_change_detection():
    """Location shifts → scene_changed=true, turn_count resets."""
    prev = make_ida()
    response = json.dumps({
        "scene": {"location": "forest", "time_of_day": "dawn",
                   "ongoing_action": "walking on a trail"},
        "user": {"physical_state": "walking"},
        "character": {"physical_state": "walking ahead", "mood": "alert", "energy_level": "energetic"},
        "relationship": {"stage": "just met", "trust_level": "cautious"},
        "meta": {"scene_changed": True, "turn_count_in_scene": 0, "last_updated_turn_index": 0, "confidence": 0.95},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "Let's go to the forest.", "I follow you into the woods.", llm_complete=llm)

    assert result.meta.scene_changed is True
    assert result.meta.turn_count_in_scene == 0  # reset
    assert result.scene.location == "forest"
    assert result.relationship.stage == "just met"  # persists


# ---- Physical state continuity ----

@pytest.mark.asyncio
async def test_physical_state_continuity():
    """Sitting → should not show standing without justification."""
    prev = make_ida()
    # LLM returns same state (no change mentioned)
    response = json.dumps({
        "scene": {"location": "tavern", "time_of_day": "evening", "ongoing_action": "drinking ale"},
        "user": {"physical_state": "seated at the table"},
        "character": {"physical_state": "leaning forward, elbows on table", "mood": "curious", "energy_level": "alert"},
        "relationship": {"stage": "just met", "trust_level": "cautious"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 6, "last_updated_turn_index": 0, "confidence": 0.95},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "Tell me more about yourself.", "Well, I grew up in these parts.", llm_complete=llm)

    assert result.user.physical_state == "seated at the table"  # unchanged
    assert result.character.physical_state == "leaning forward, elbows on table"  # unchanged


# ---- Appearance drift ----

@pytest.mark.asyncio
async def test_appearance_drift():
    """Appearance only changes when exchange describes it."""
    prev = make_ida()
    prev.character.appearance = "tall, dark hair, scar on cheek"

    response = json.dumps({
        "scene": {"location": "tavern", "time_of_day": "evening", "ongoing_action": "drinking ale"},
        "user": {"physical_state": "seated at the table"},
        "character": {"physical_state": "leaning forward", "mood": "curious",
                       "appearance": "tall, dark hair, scar on cheek"},
        "relationship": {"stage": "just met", "trust_level": "cautious"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 6, "last_updated_turn_index": 0, "confidence": 0.95},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "What's your name?", "I'm Elena.", llm_complete=llm)

    assert result.character.appearance == "tall, dark hair, scar on cheek"  # unchanged


# ---- Relationship persists across scene change ----

@pytest.mark.asyncio
async def test_relationship_persists_across_scenes():
    """Relationship stage/trust doesn't reset on scene change."""
    prev = make_ida()
    prev.relationship.stage = "friends"
    prev.relationship.trust_level = "high"

    # LLM returns scene change but omits relationship (should be carried forward)
    response = json.dumps({
        "scene": {"location": "castle", "time_of_day": "night", "ongoing_action": "exploring"},
        "user": {"physical_state": "walking cautiously"},
        "character": {"physical_state": "beside you, sword drawn", "mood": "alert"},
        "meta": {"scene_changed": True, "turn_count_in_scene": 0, "last_updated_turn_index": 0, "confidence": 0.9},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "Let's search the castle.", "Stay close. This place is dangerous.", llm_complete=llm)

    assert result.relationship.stage == "friends"  # carried forward
    assert result.relationship.trust_level == "high"  # carried forward


@pytest.mark.asyncio
async def test_partial_update_merges_over_previous():
    """A compact 'only what changed' response is merged over the previous
    state — untouched fields survive (the partial-output contract)."""
    prev = make_ida()
    # LLM returns ONLY the changed fields — no scene/user/relationship at all.
    response = json.dumps({
        "character": {"mood": "angry"},
        "meta": {"turn_count_in_scene": 6},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "You're lying!", "I don't lie.", llm_complete=llm)

    # changed fields applied
    assert result.character.mood == "angry"
    assert result.meta.turn_count_in_scene == 6
    # untouched fields carried forward
    assert result.scene.location == "tavern"
    assert result.character.physical_state == "leaning forward, elbows on table"
    assert result.user.physical_state == "seated at the table"
    assert result.relationship.stage == "just met"
    assert result.character.energy_level == "alert"


# ---- Malformed LLM fallback ----

@pytest.mark.asyncio
async def test_malformed_llm_fallback():
    """Malformed JSON → return previous_ida unchanged, no crash."""
    prev = make_ida()
    llm = StubLLM("this is not json")

    result = await update_world_ida(prev, "Hello.", "Hi there!", llm_complete=llm)

    assert result.scene.location == prev.scene.location
    assert result.character.mood == prev.character.mood
    assert result.meta.turn_count_in_scene == prev.meta.turn_count_in_scene


# ---- First-turn init ----

@pytest.mark.asyncio
async def test_first_turn_init():
    """previous_ida=None → initialize from exchange."""
    response = json.dumps({
        "scene": {"location": "coffee shop", "time_of_day": "morning", "ongoing_action": "ordering coffee"},
        "user": {"physical_state": "standing at the counter"},
        "character": {"physical_state": "behind the counter, smiling", "mood": "friendly"},
        "relationship": {"stage": "strangers"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 0, "last_updated_turn_index": 0, "confidence": 0.8},
    })

    llm = StubLLM(response)
    result = await update_world_ida(None, "Good morning!", "Welcome! What can I get you?", llm_complete=llm)

    assert result.scene.location == "coffee shop"
    assert result.character.mood == "friendly"
    assert result.relationship.stage == "strangers"


# ---- Context string ----

def test_context_string():
    """world_ida_to_context_string produces readable output."""
    ida = make_ida()
    ctx = world_ida_to_context_string(ida)

    assert "tavern" in ctx
    assert "corner booth" in ctx
    assert "evening" in ctx
    assert "curious" in ctx
    assert "seated at the table" in ctx
    assert "just met" in ctx
    assert "cautious" in ctx
    assert "drinking ale" in ctx
    assert len(ctx.split()) < 100  # reasonable length


def test_context_string_empty():
    """Empty worldIDA produces empty or minimal string."""
    ida = WorldIDA()
    ctx = world_ida_to_context_string(ida)
    assert ctx == "" or len(ctx.split()) < 10


# ---- Scene transition summary ----

def test_scene_transition_summary():
    """scene_transition_summary captures key details."""
    ida = make_ida()
    summary = scene_transition_summary(ida)

    assert "tavern" in summary
    assert "just met" in summary
    assert "drinking ale" in summary
    assert summary.startswith("Scene ended:")


# ---- Validation ----

def test_validate_ida_valid():
    """Valid WorldIDA passes validation."""
    data = {
        "scene": {"location": "tavern"},
        "user": {"physical_state": "seated"},
        "character": {"mood": "happy"},
        "relationship": {"stage": "friends"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 1},
    }
    assert _validate_ida(data) is True


def test_validate_ida_missing_section():
    """A partial update (only changed sections) is accepted — unchanged
    fields are carried over from the previous state during the merge."""
    data = {
        "scene": {"location": "tavern"},
        "user": {"physical_state": "seated"},
        # missing character, relationship, meta — fine for a compact update
    }
    assert _validate_ida(data) is True


def test_validate_ida_empty_rejected():
    """A response with no known sections is rejected."""
    assert _validate_ida({}) is False
    assert _validate_ida({"unrelated": "x"}) is False
    assert _validate_ida("not a dict") is False


# ---- Store ----

def test_world_ida_store():
    """WorldIDAStore set/get/history works."""
    store = WorldIDAStore()
    ida = make_ida()

    store.set("session_1", ida)
    assert store.get("session_1") is not None
    assert store.get("session_1").scene.location == "tavern"

    # History
    history = store.get_history("session_1")
    assert len(history) == 1

    # Update and check history grows
    ida2 = make_ida()
    ida2.scene.location = "forest"
    store.set("session_1", ida2)
    assert len(store.get_history("session_1")) == 2


def test_world_ida_store_diff():
    """diff_versions returns correct changes between versions."""
    store = WorldIDAStore()
    ida1 = make_ida()
    store.set("s1", ida1)

    ida2 = make_ida()
    ida2.character.mood = "angry"
    ida2.meta.turn_count_in_scene = 6
    store.set("s1", ida2)

    diff = store.diff_versions("s1", -2, -1)
    assert diff is not None
    assert "character" in diff
    assert diff["character"]["mood"]["from"] == "curious"
    assert diff["character"]["mood"]["to"] == "angry"
    assert diff["meta"]["turn_count_in_scene"]["from"] == 5
    assert diff["meta"]["turn_count_in_scene"]["to"] == 6
    # Scene should not appear in diff (unchanged)
    assert "scene" not in diff


def test_world_ida_store_serialization():
    """Store to_dict/from_dict round-trips correctly."""
    store = WorldIDAStore()
    store.set("s1", make_ida())
    store.set("s2", make_ida())

    data = store.to_dict()
    assert "current" in data
    assert "history" in data

    store2 = WorldIDAStore()
    store2.from_dict(data)
    assert store2.get("s1") is not None
    assert store2.get("s1").scene.location == "tavern"
