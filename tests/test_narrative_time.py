"""
Narrative time tests.

Narrative time (time_of_day + the monotonic day counter) lives outside worldIDA
in its own module and context block. The day counter only ever advances — it
never resets or decreases — so the AI's estimate of elapsed story time is
always internally consistent.

Covers: first-turn init, partial merge, monotonic clamp, night->morning
boundary, explicit time-skip, malformed LLM fallback, context string, validation.
"""

import json
import pytest
from hypermem.narrative_time import (
    NarrativeTime, _tod_bucket, _validate,
    update_narrative_time, narrative_time_to_context_string,
)


def make_time(time_of_day: str = "", day_count: int = 0) -> NarrativeTime:
    return NarrativeTime(time_of_day=time_of_day, day_count=day_count)


class StubLLM:
    """Test double: an async LLM callable returning a canned response."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    async def __call__(self, messages, **kwargs):
        self.last_prompt = messages[0]["content"]
        return self.response


# ---- First-turn init ----

@pytest.mark.asyncio
async def test_first_turn_init():
    """With previous=None, initialize from the exchange."""
    llm = StubLLM(json.dumps({"time_of_day": "evening", "day_count": 0}))
    result = await update_narrative_time(None, "The sun is setting.", "We sit by the fire.", llm_complete=llm)
    assert result.time_of_day == "evening"
    assert result.day_count == 0


# ---- Partial merge ----

@pytest.mark.asyncio
async def test_partial_update_merges():
    """An 'only what changed' response merges over the previous state."""
    prev = make_time(time_of_day="morning", day_count=3)
    llm = StubLLM(json.dumps({"time_of_day": "afternoon"}))
    result = await update_narrative_time(prev, "The hours pass.", "Time flies.", llm_complete=llm)
    assert result.time_of_day == "afternoon"  # changed
    assert result.day_count == 3  # carried forward


# ---- Monotonic counter ----

@pytest.mark.asyncio
async def test_day_count_never_decreases():
    """An LLM-reported day_count lower than the current one is clamped — the
    counter only ever advances (3 weeks never becomes 'a few days')."""
    prev = make_time(time_of_day="morning", day_count=21)
    llm = StubLLM(json.dumps({"day_count": 4}))  # model regressed
    result = await update_narrative_time(prev, "What day is it?", "A new day.", llm_complete=llm)
    assert result.day_count == 21


@pytest.mark.asyncio
async def test_explicit_higher_day_count_wins():
    """An explicit time skip the LLM reports wins over the heuristic."""
    prev = make_time(time_of_day="morning", day_count=1)
    llm = StubLLM(json.dumps({"day_count": 4}))  # "three days passed"
    result = await update_narrative_time(prev, "Three days pass.", "We travel on.", llm_complete=llm)
    assert result.day_count == 4


@pytest.mark.asyncio
async def test_night_to_morning_advances():
    """A night->morning boundary (deterministic bucket detection) advances the
    counter even when the LLM omits day_count."""
    prev = make_time(time_of_day="night", day_count=2)
    llm = StubLLM(json.dumps({"time_of_day": "morning"}))
    result = await update_narrative_time(prev, "We wake at dawn.", "Morning already.", llm_complete=llm)
    assert result.day_count == 3
    assert result.time_of_day == "morning"


@pytest.mark.asyncio
async def test_same_day_no_advance():
    """Continuing within the same day leaves the counter unchanged."""
    prev = make_time(time_of_day="morning", day_count=1)
    llm = StubLLM(json.dumps({"time_of_day": "afternoon"}))
    result = await update_narrative_time(prev, "Later that day.", "Still afternoon.", llm_complete=llm)
    assert result.day_count == 1


# ---- Malformed LLM fallback ----

@pytest.mark.asyncio
async def test_malformed_llm_fallback():
    """Malformed JSON returns previous unchanged, without crashing."""
    prev = make_time(time_of_day="night", day_count=5)
    llm = StubLLM("this is not json")
    result = await update_narrative_time(prev, "Hello.", "Hi!", llm_complete=llm)
    assert result.time_of_day == prev.time_of_day
    assert result.day_count == prev.day_count


@pytest.mark.asyncio
async def test_unknown_keys_rejected():
    """A response carrying non-time fields fails validation and falls back."""
    prev = make_time(day_count=2)
    llm = StubLLM(json.dumps({"time_of_day": "morning", "mood": "happy"}))
    result = await update_narrative_time(prev, "Hello.", "Hi!", llm_complete=llm)
    assert result.day_count == 2  # unchanged
    assert result.time_of_day == ""  # unchanged


@pytest.mark.asyncio
async def test_nested_values_rejected():
    """Nested values can't poison the counter."""
    llm = StubLLM(json.dumps({"time_of_day": {"now": "morning"}}))
    result = await update_narrative_time(None, "Hello.", "Hi!", llm_complete=llm)
    assert result.day_count == 0
    assert result.time_of_day == ""


# ---- Bucket classification ----

def test_tod_bucket():
    assert _tod_bucket("night") == "night"
    assert _tod_bucket("Evening") == "night"
    assert _tod_bucket("midnight") == "night"
    assert _tod_bucket("morning") == "morning"
    assert _tod_bucket("dawn") == "morning"
    assert _tod_bucket("afternoon") == "day"
    assert _tod_bucket("noon") == "day"
    assert _tod_bucket("") is None
    assert _tod_bucket("the tavern") is None


# ---- Validation ----

def test_validate():
    assert _validate({"time_of_day": "morning"}) is True
    assert _validate({"day_count": 3}) is True
    assert _validate({}) is True  # empty = nothing changed, valid
    assert _validate({"mood": "happy"}) is False
    assert _validate({"time_of_day": {"now": "morning"}}) is False
    assert _validate("not a dict") is False


# ---- Context string ----

def test_context_string_day_zero_empty():
    """Day 0 (opening scene) produces no block — no elapsed time to report."""
    assert narrative_time_to_context_string(make_time()) == ""
    assert narrative_time_to_context_string(make_time(time_of_day="night")) == ""


def test_context_string_day_one():
    t = make_time(time_of_day="morning", day_count=1)
    ctx = narrative_time_to_context_string(t)
    assert "morning" in ctx
    assert "day 1" in ctx
    assert "day has passed" in ctx


def test_context_string_multi_day():
    t = make_time(time_of_day="night", day_count=22)
    ctx = narrative_time_to_context_string(t)
    assert "night" in ctx
    assert "day 22" in ctx
    assert "21 days have passed" in ctx