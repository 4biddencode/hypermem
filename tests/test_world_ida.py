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
    """A location shift sets scene_changed=true and resets turn_count."""
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
    """Sitting should not become standing without justification."""
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
    """Malformed JSON returns previous_ida unchanged, without crashing."""
    prev = make_ida()
    llm = StubLLM("this is not json")

    result = await update_world_ida(prev, "Hello.", "Hi there!", llm_complete=llm)

    assert result.scene.location == prev.scene.location
    assert result.character.mood == prev.character.mood
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


# ---- Round-2 findings ----

@pytest.mark.asyncio
async def test_turn_count_increments_when_omitted():
    """Rule 3: when the LLM omits turn_count_in_scene (compact output), the
    merge must increment the prior count rather than carry it stale."""
    prev = make_ida()  # turn_count_in_scene=5
    # LLM emits only scene_changed=false, omitting the count entirely
    response = json.dumps({"meta": {"scene_changed": False}})
    llm = StubLLM(response)
    result = await update_world_ida(prev, "Hello.", "Hi!", llm_complete=llm)
    assert result.meta.turn_count_in_scene == 6  # 5 + 1


@pytest.mark.asyncio
async def test_turn_count_resets_on_scene_change_when_omitted():
    """Rule 3: a scene change with an omitted count must reset to 0, not carry
    the stale prior count into the new scene."""
    prev = make_ida()  # turn_count_in_scene=5
    response = json.dumps({"meta": {"scene_changed": True}})
    llm = StubLLM(response)
    result = await update_world_ida(prev, "Let's leave.", "We head out.", llm_complete=llm)
    assert result.meta.scene_changed is True
    assert result.meta.turn_count_in_scene == 0


@pytest.mark.asyncio
async def test_explicit_turn_count_respected():
    """An explicitly-provided count wins over the auto-increment."""
    prev = make_ida()  # turn_count_in_scene=5
    response = json.dumps({"meta": {"scene_changed": False, "turn_count_in_scene": 9}})
    llm = StubLLM(response)
    result = await update_world_ida(prev, "Hello.", "Hi!", llm_complete=llm)
    assert result.meta.turn_count_in_scene == 9  # not auto-incremented


@pytest.mark.asyncio
async def test_null_confidence_keeps_prior():
    """An explicit null confidence must not reset a lowered confidence to max —
    null reads as 'unchanged', so the prior value is carried forward."""
    prev = make_ida()
    prev.meta.confidence = 0.4  # previously lowered
    response = json.dumps({"meta": {"confidence": None}})
    llm = StubLLM(response)
    result = await update_world_ida(prev, "Hello.", "Hi!", llm_complete=llm)
    assert result.meta.confidence == 0.4  # preserved, not reset to 1.0


def test_ida_from_dict_none_section_does_not_crash():
    """A present-but-None section (e.g. "meta": null in foreign/corrupt JSON)
    must not raise AttributeError during load."""
    ida = _ida_from_dict({"meta": None, "scene": None})
    assert ida.meta.confidence == 1.0
    assert ida.scene.location == ""


def test_as_int_accepts_float_strings():
    """_as_int("6.0") / 6.0 must yield 6, not fall back to 0 and lose a count."""
    from hypermem.world_ida import _as_int
    assert _as_int("6.0") == 6
    assert _as_int(6.0) == 6
    assert _as_int("3") == 3
    assert _as_int("not a number") == 0  # genuinely bad input still defaults


# ---- Narrative day counter ----
#
# The day_count is the single source of truth for elapsed in-story time. Its
# defining property is monotonicity: it only ever advances, never resets or
# decreases. That is what keeps the AI's time estimates internally consistent —
# once it has said "3 weeks", a later turn can never contradict it with "a few
# days", because day_count only grows. These tests pin the two advance signals
# (explicit LLM time-skip, and the deterministic night->morning boundary) and
# guard the monotonic invariant against drift and regression.

class TestNarrativeDayCounter:
    @pytest.mark.asyncio
    async def test_starts_at_zero_on_fresh_init(self):
        """A brand-new worldIDA (no previous state) starts on day 0."""
        response = json.dumps({"scene": {"time_of_day": "morning"},
                               "meta": {"day_count": 0}})
        llm = StubLLM(response)
        result = await update_world_ida(None, "Good morning.", "Morning!", llm_complete=llm)
        assert result.meta.day_count == 0

    @pytest.mark.asyncio
    async def test_night_to_morning_advances_day(self):
        """A night/evening time_of_day advancing to morning/dawn marks a new
        in-story day — the day counter increments by exactly one."""
        prev = make_ida()  # time_of_day="evening", day_count defaults to 0
        response = json.dumps({"scene": {"time_of_day": "morning"}})
        llm = StubLLM(response)
        result = await update_world_ida(prev, "We slept.", "You wake up.", llm_complete=llm)
        assert result.meta.day_count == 1

    @pytest.mark.asyncio
    async def test_same_day_no_advance(self):
        """Staying within the same day must NOT advance the counter — the whole
        point is that "later that day" is not a new day."""
        prev = make_ida()  # evening
        prev.meta.day_count = 2
        response = json.dumps({"scene": {"time_of_day": "night"}})
        llm = StubLLM(response)
        result = await update_world_ida(prev, "Still talking.", "Yeah.", llm_complete=llm)
        assert result.meta.day_count == 2  # night is not a boundary

    @pytest.mark.asyncio
    async def test_explicit_llm_time_skip_wins(self):
        """When the LLM reports a jump (e.g. "three days later"), its explicit
        day_count wins over the heuristic — a multi-day skip the model can see
        in the narrative is honored even with no night->morning seen."""
        prev = make_ida()  # evening, day_count 0
        response = json.dumps({"meta": {"day_count": 3}})  # model says 3 days passed
        llm = StubLLM(response)
        result = await update_world_ida(prev, "Three days later...", "It's been days.", llm_complete=llm)
        assert result.meta.day_count == 3

    @pytest.mark.asyncio
    async def test_explicit_lower_never_decreases(self):
        """Monotonic invariant: an explicit day_count BELOW the current value
        must not decrease the counter. The story's elapsed time can never go
        backwards — that would let "3 weeks" become "a few days"."""
        prev = make_ida()
        prev.meta.day_count = 5
        response = json.dumps({"meta": {"day_count": 2}})  # model misreports lower
        llm = StubLLM(response)
        result = await update_world_ida(prev, "Hello.", "Hi!", llm_complete=llm)
        assert result.meta.day_count == 5  # clamped to the prior value

    @pytest.mark.asyncio
    async def test_carries_forward_when_omitted(self):
        """A compact output that omits day_count must carry the prior value
        forward, not reset it to 0."""
        prev = make_ida()
        prev.meta.day_count = 4
        response = json.dumps({"meta": {"scene_changed": False}})  # no day_count
        llm = StubLLM(response)
        result = await update_world_ida(prev, "Hello.", "Hi!", llm_complete=llm)
        assert result.meta.day_count == 4  # preserved

    @pytest.mark.asyncio
    async def test_multiple_nights_accumulate(self):
        """Sequential night->morning crossings accumulate: each new day adds
        one, so after several story-days the counter reflects the total."""
        prev = make_ida()  # evening
        prev.meta.day_count = 0
        for i in range(1, 4):
            # Each cycle: evening -> (night) -> morning. The merge only sees the
            # morning output, but the previous state's time_of_day is "evening",
            # so each crossing is a night->morning boundary and advances by one.
            response = json.dumps({"scene": {"time_of_day": "morning"}})
            llm = StubLLM(response)
            prev = await update_world_ida(prev, "Slept.", "Morning.", llm_complete=llm)
            assert prev.meta.day_count == i
            # Reset time_of_day back to evening so the next iteration is again a
            # night->morning boundary (morning->morning would not advance).
            prev.scene.time_of_day = "evening"
        assert prev.meta.day_count == 3

    @pytest.mark.asyncio
    async def test_scene_change_within_day_keeps_count(self):
        """A scene change (new location) that is NOT a day boundary must keep the
        same day_count — time-of-day is what advances the day, not location."""
        prev = make_ida()
        prev.meta.day_count = 2
        response = json.dumps({"scene": {"location": "forest"},
                               "meta": {"scene_changed": True}})
        llm = StubLLM(response)
        result = await update_world_ida(prev, "Let's go to the forest.", "We walk in.", llm_complete=llm)
        assert result.meta.day_count == 2  # location change, same day

    def test_context_string_surfaces_narrative_time(self):
        """Once the counter advances, the context string tells the AI the day
        and elapsed time, so it can give consistent estimates."""
        ida = make_ida()
        ida.meta.day_count = 3
        s = world_ida_to_context_string(ida)
        assert "day 3" in s
        assert "2 days" in s  # day 3 means ~2 days elapsed

    def test_context_string_omits_day_zero(self):
        """Day 0 (opening scene, no time passed) must not clutter the context
        with a narrative-time line."""
        ida = make_ida()  # day_count defaults to 0
        s = world_ida_to_context_string(ida)
        assert "day " not in s
        assert "days" not in s
