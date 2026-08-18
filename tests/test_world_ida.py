"""
worldIDA tests.

worldIDA tracks only the PHYSICAL state: where characters are, how they sit,
what they physically do, whether a described action is physically possible.
Emotional/social state and narrative time live elsewhere (narrative_time).

Covers: single-field update, scene change detection, physical state continuity,
appearance drift, position persistence, physically-possible flag, malformed LLM
fallback, first-turn init, and 50+ turn stability.
"""

import json
import pytest
from hypermem.world_ida import (
    WorldIDA, Scene, UserState, CharacterState, Meta,
    update_world_ida, world_ida_to_context_string, scene_transition_summary,
    WorldIDAStore, _ida_from_dict, _validate_ida,
)


def make_ida(**overrides) -> WorldIDA:
    """Helper to build a WorldIDA with sensible defaults."""
    ida = WorldIDA(
        scene=Scene(
            location="tavern",
            sub_location="corner booth",
            ongoing_action="drinking ale",
        ),
        user=UserState(physical_state="seated at the table",
                       position="left side of the booth"),
        character=CharacterState(
            physical_state="leaning forward, elbows on table",
            position="across from you, right side of the booth",
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
async def test_single_field_physical_state_change():
    """Only physical_state changes; all other fields identical."""
    prev = make_ida()
    response = json.dumps({
        "scene": {"location": "tavern", "sub_location": "corner booth",
                   "ongoing_action": "drinking ale"},
        "user": {"physical_state": "seated at the table", "position": "left side of the booth"},
        "character": {"physical_state": "standing up, hand raised", "position": "across from you, right side of the booth"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 6, "last_updated_turn_index": 0, "confidence": 0.95},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "You're lying!", "I don't lie.", llm_complete=llm)

    assert result.character.physical_state == "standing up, hand raised"
    assert result.scene.location == "tavern"  # unchanged
    assert result.user.position == "left side of the booth"  # unchanged
    assert result.meta.turn_count_in_scene == 6  # incremented
    assert result.meta.scene_changed is False


# ---- Scene change detection ----

@pytest.mark.asyncio
async def test_scene_change_detection():
    """A location shift sets scene_changed=true and resets turn_count."""
    prev = make_ida()
    response = json.dumps({
        "scene": {"location": "forest",
                   "ongoing_action": "walking on a trail"},
        "user": {"physical_state": "walking", "position": "on the trail ahead"},
        "character": {"physical_state": "walking behind you", "position": "a few steps back"},
        "meta": {"scene_changed": True, "turn_count_in_scene": 0, "last_updated_turn_index": 0, "confidence": 0.95},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "Let's go to the forest.", "I follow you into the woods.", llm_complete=llm)

    assert result.meta.scene_changed is True
    assert result.meta.turn_count_in_scene == 0  # reset
    assert result.scene.location == "forest"


# ---- Physical state continuity ----

@pytest.mark.asyncio
async def test_physical_state_continuity():
    """Sitting should not become standing without justification."""
    prev = make_ida()
    # LLM returns same state (no change mentioned)
    response = json.dumps({
        "scene": {"location": "tavern", "sub_location": "corner booth", "ongoing_action": "drinking ale"},
        "user": {"physical_state": "seated at the table", "position": "left side of the booth"},
        "character": {"physical_state": "leaning forward, elbows on table", "position": "across from you, right side of the booth"},
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
        "scene": {"location": "tavern", "sub_location": "corner booth", "ongoing_action": "drinking ale"},
        "user": {"physical_state": "seated at the table", "position": "left side of the booth"},
        "character": {"physical_state": "leaning forward", "position": "across from you",
                       "appearance": "tall, dark hair, scar on cheek"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 6, "last_updated_turn_index": 0, "confidence": 0.95},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "What's your name?", "I'm Elena.", llm_complete=llm)

    assert result.character.appearance == "tall, dark hair, scar on cheek"  # unchanged


# ---- Position persists across scene change ----

@pytest.mark.asyncio
async def test_position_persists_across_scenes():
    """Character position doesn't reset on scene change when omitted."""
    prev = make_ida()
    prev.character.position = "at the far end of the bar"

    # LLM returns scene change but omits position (should be carried forward)
    response = json.dumps({
        "scene": {"location": "castle", "ongoing_action": "exploring"},
        "user": {"physical_state": "walking cautiously"},
        "character": {"physical_state": "beside you, sword drawn"},
        "meta": {"scene_changed": True, "turn_count_in_scene": 0, "last_updated_turn_index": 0, "confidence": 0.9},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "Let's search the castle.", "Stay close. This place is dangerous.", llm_complete=llm)

    assert result.character.position == "at the far end of the bar"  # carried forward


# ---- Physically possible flag ----

@pytest.mark.asyncio
async def test_physically_impossible_flag():
    """A described action that can't happen in the current state sets
    physically_possible=false."""
    prev = make_ida()
    response = json.dumps({
        "character": {"physical_state": "reaching for the bottle"},
        "meta": {"physically_possible": False},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "Elena grabs the bottle from across the room.", "She reaches.", llm_complete=llm)

    assert result.meta.physically_possible is False

@pytest.mark.asyncio
async def test_physically_possible_defaults_true_when_omitted():
    """Omission of physically_possible reads as 'unchanged' — the prior True
    stays, never fabricates an impossibility."""
    prev = make_ida()  # physically_possible defaults True
    response = json.dumps({"meta": {"scene_changed": False}})
    llm = StubLLM(response)
    result = await update_world_ida(prev, "Hello.", "Hi!", llm_complete=llm)
    assert result.meta.physically_possible is True


@pytest.mark.asyncio
async def test_partial_update_merges_over_previous():
    """A compact 'only what changed' response is merged over the previous
    state — untouched fields survive (the partial-output contract)."""
    prev = make_ida()
    # LLM returns ONLY the changed fields — no scene/user at all.
    response = json.dumps({
        "character": {"physical_state": "standing up"},
        "meta": {"turn_count_in_scene": 6},
    })

    llm = StubLLM(response)
    result = await update_world_ida(prev, "You're lying!", "I don't lie.", llm_complete=llm)

    # changed fields applied
    assert result.character.physical_state == "standing up"
    assert result.meta.turn_count_in_scene == 6
    # untouched fields carried forward
    assert result.scene.location == "tavern"
    assert result.user.physical_state == "seated at the table"
    assert result.character.position == "across from you, right side of the booth"


# ---- Malformed LLM fallback ----

@pytest.mark.asyncio
async def test_malformed_llm_fallback():
    """Malformed JSON returns previous_ida unchanged, without crashing."""
    prev = make_ida()
    llm = StubLLM("this is not json")

    result = await update_world_ida(prev, "Hello.", "Hi there!", llm_complete=llm)

    assert result.scene.location == prev.scene.location
    assert result.character.physical_state == prev.character.physical_state
    assert result.meta.turn_count_in_scene == prev.meta.turn_count_in_scene


@pytest.mark.asyncio
async def test_single_quoted_json_still_updates():
    """A model that emits single-quoted JSON (a common failure mode) must
    still update the world state, matching the judge/recall lenient parsing."""
    prev = make_ida()
    llm = StubLLM("{'meta': {'scene_changed': True, 'turn_count_in_scene': 9}}")

    result = await update_world_ida(prev, "Hello.", "Hi there!", llm_complete=llm)

    assert result.meta.turn_count_in_scene == 9
    assert result.scene.location == prev.scene.location  # unchanged field carried over


# ---- First-turn init ----

@pytest.mark.asyncio
async def test_first_turn_init():
    """With previous_ida=None, initialize from the exchange."""
    response = json.dumps({
        "scene": {"location": "coffee shop", "ongoing_action": "ordering coffee"},
        "user": {"physical_state": "standing at the counter", "position": "at the register"},
        "character": {"physical_state": "behind the counter", "position": "at the espresso machine"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 0, "last_updated_turn_index": 0, "confidence": 0.8},
    })

    llm = StubLLM(response)
    result = await update_world_ida(None, "Good morning!", "Welcome! What can I get you?", llm_complete=llm)

    assert result.scene.location == "coffee shop"
    assert result.user.position == "at the register"


# ---- Context string ----

def test_context_string():
    """world_ida_to_context_string produces readable physical output."""
    ida = make_ida()
    ctx = world_ida_to_context_string(ida)

    assert "tavern" in ctx
    assert "corner booth" in ctx
    assert "leaning forward" in ctx
    assert "seated at the table" in ctx
    assert "position: left side of the booth" in ctx
    assert "drinking ale" in ctx
    assert len(ctx.split()) < 100  # reasonable length


def test_context_string_physically_impossible_note():
    """physically_possible=false surfaces a note in the context string."""
    ida = make_ida()
    ida.meta.physically_possible = False
    ctx = world_ida_to_context_string(ida)
    assert "physically impossible" in ctx


def test_context_string_empty():
    """Empty worldIDA produces empty or minimal string."""
    ida = WorldIDA()
    ctx = world_ida_to_context_string(ida)
    assert ctx == "" or len(ctx.split()) < 10


# ---- Scene transition summary ----

def test_scene_transition_summary():
    """scene_transition_summary captures physical details."""
    ida = make_ida()
    summary = scene_transition_summary(ida)

    assert "tavern" in summary
    assert "drinking ale" in summary
    assert summary.startswith("Scene ended:")


# ---- Validation ----

def test_validate_ida_valid():
    """Valid WorldIDA passes validation."""
    data = {
        "scene": {"location": "tavern"},
        "user": {"physical_state": "seated"},
        "character": {"physical_state": "standing"},
        "meta": {"scene_changed": False, "turn_count_in_scene": 1},
    }
    assert _validate_ida(data) is True


def test_validate_ida_missing_section():
    """A partial update (only changed sections) is accepted — unchanged
    fields are carried over from the previous state during the merge."""
    data = {
        "scene": {"location": "tavern"},
        "user": {"physical_state": "seated"},
        # missing character, meta — fine for a compact update
    }
    assert _validate_ida(data) is True


def test_validate_ida_empty_rejected():
    """A response with no known sections is rejected."""
    assert _validate_ida({}) is False
    assert _validate_ida({"unrelated": "x"}) is False
    assert _validate_ida("not a dict") is False


def test_validate_ida_rejects_relationship_section():
    """relationship is no longer a worldIDA section — a model that still emits
    it must not pass validation (it would be silently dropped)."""
    assert _validate_ida({"relationship": {"stage": "friends"}}) is False


def test_validate_ida_rejects_nested_values():
    """A field whose value is itself a dict/list must be rejected so context
    injection never crashes joining a non-string element."""
    assert _validate_ida({"scene": {"location": {"coords": [1, 2]}}}) is False


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
    ida2.character.physical_state = "standing up"
    ida2.meta.turn_count_in_scene = 6
    store.set("s1", ida2)

    diff = store.diff_versions("s1", -2, -1)
    assert diff is not None
    assert "character" in diff
    assert diff["character"]["physical_state"]["from"] == "leaning forward, elbows on table"
    assert diff["character"]["physical_state"]["to"] == "standing up"
    assert diff["meta"]["turn_count_in_scene"]["from"] == 5
    assert diff["meta"]["turn_count_in_scene"]["to"] == 6
    # Scene should not appear in diff (unchanged)
    assert "scene" not in diff


def test_world_ida_store_serialization():
    """Store to_dict/from_dict round-trips correctly."""
    store = WorldIDAStore()
    store.set("s1", make_ida())
    store.set("s2", make_ida())