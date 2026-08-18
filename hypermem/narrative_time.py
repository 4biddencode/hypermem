"""
Narrative time — how much in-story time has passed, separate from worldIDA.

worldIDA tracks only the PHYSICAL state. Narrative time (time_of_day + the
monotonic day counter) lives here, in its own object injected as a separate
context block, so the AI still knows it's morning, evening, or that roughly
21 days have passed since the story began — without polluting the physical
world state.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from .llm import extract_json_object

logger = logging.getLogger("hypermem.narrative_time")


# 1. SCHEMA

@dataclass
class NarrativeTime:
    time_of_day: str = ""
    day_count: int = 0


# Narrative time-of-day buckets. A transition from a night-time bucket to a
# morning bucket marks a new in-story day (a day/night boundary) — the signal
# the day counter advances on, so the AI's sense of elapsed story time is
# monotonic and can never drift backward.
_NIGHT_BUCKETS = {"night", "midnight", "evening", "dusk", "late night"}
_MORNING_BUCKETS = {"morning", "dawn", "sunrise", "day"}


def _tod_bucket(time_of_day: str) -> Optional[str]:
    """Classify a time_of_day string into a coarse bucket, or None if it's a
    non-temporal value (a location, empty, or something unclassifiable)."""
    t = (time_of_day or "").strip().lower()
    if t in _NIGHT_BUCKETS:
        return "night"
    if t in _MORNING_BUCKETS:
        return "morning"
    # Heuristic: "afternoon"/"noon"/"midday" are within-day, not a boundary.
    if any(w in t for w in ("afternoon", "noon", "midday", "mid-day")):
        return "day"
    return None


def _as_int(value, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        # Accept "6.0" / 6.0 as 6 — a small model may emit a float as a string
        # for an int field, and int("6.0") would raise and silently reset the
        # count to default.
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_str(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _from_dict(data: dict) -> NarrativeTime:
    src = data if isinstance(data, dict) else {}
    return NarrativeTime(
        time_of_day=_as_str(src.get("time_of_day")),
        day_count=_as_int(src.get("day_count")),
    )


def _to_dict(t: NarrativeTime) -> dict:
    d = {"time_of_day": t.time_of_day}
    if t.day_count:
        d["day_count"] = t.day_count
    return d


# 2. UPDATE

def _build_update_prompt(previous: Optional[NarrativeTime],
                         last_user_message: str,
                         last_ai_response: str) -> str:
    if previous is not None:
        previous_json = json.dumps(_to_dict(previous))
    else:
        previous_json = "null (first turn — infer the time of day from the exchange)"

    return f"""You are tracking the TIME in a roleplay story — what time of day it is and how many in-story days have passed. Output a compact JSON object with ONLY the fields that changed.

SCHEMA (field names):
{{
  "time_of_day": "",
  "day_count": 0
}}

RULES:
1. Output ONLY fields that changed or are new — omit unchanged fields entirely.
2. time_of_day is the current in-story time ("morning", "evening", "night", "dawn"). Omit it if unchanged.
3. day_count is how many in-story days have passed. Increment it when a new day begins — a time skip ("the next morning", "three days later", "we slept"), a night->morning advance, or an explicit passage of time. If the story continues within the same day, omit it.
4. Output ONLY valid JSON. No preamble, no explanation.

Previous state: {previous_json}
User: {last_user_message}
Character: {last_ai_response}

Output JSON:"""


def _validate(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    # Only time_of_day and day_count are valid keys.
    for k in data:
        if k not in ("time_of_day", "day_count"):
            return False
    # Reject nested values.
    if any(isinstance(v, (dict, list, set, tuple)) for v in data.values()):
        return False
    return True


def _advance_day_count(previous: Optional[NarrativeTime], base: dict) -> int:
    """Return the day_count for the merged state.

    The day counter is monotonic — it only ever advances, never resets or
    decreases — so the AI's estimate of elapsed story time is always internally
    consistent (3 weeks never becomes "a few days"). It advances when:
      - the LLM explicitly reports a higher day_count (a time skip it can see
        in the narrative, e.g. "three days later"), OR
      - a night/evening time_of_day advances to morning/dawn (a day/night
        boundary the deterministic bucketing detects).
    An explicit LLM day_count wins over the heuristic so a multi-day skip the
    model calls out is honored even when the intermediate nights weren't seen.
    """
    prev_day = previous.day_count if previous is not None else 0
    merged_day = _as_int(base.get("day_count", 0) or 0)
    if merged_day > prev_day:
        return merged_day

    prev_tod = previous.time_of_day if previous is not None else ""
    new_tod = _as_str(base.get("time_of_day", ""))
    prev_bucket = _tod_bucket(prev_tod)
    new_bucket = _tod_bucket(new_tod)
    if prev_bucket == "night" and new_bucket == "morning":
        return prev_day + 1
    return prev_day


async def update_narrative_time(
    previous: Optional[NarrativeTime],
    last_user_message: str,
    last_ai_response: str,
    llm_complete=None,
) -> NarrativeTime:
    """
    Update narrative time with the latest exchange.

    Returns:
        Updated NarrativeTime. On failure, returns previous unchanged (or fresh).
    """
    failure_count = getattr(update_narrative_time, "_failure_count", 0)

    prompt = _build_update_prompt(previous, last_user_message, last_ai_response)

    try:
        if llm_complete is None:
            logger.warning("narrative_time: no llm_complete provided, returning previous state")
            update_narrative_time._failure_count = failure_count + 1
            return previous or NarrativeTime()

        result = await llm_complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )

        if not result:
            logger.warning("narrative_time: LLM returned None")
            update_narrative_time._failure_count = failure_count + 1
            return previous or NarrativeTime()

        parsed = extract_json_object(result)

        if parsed is None or not _validate(parsed):
            logger.warning(f"narrative_time: validation failed, raw: {result[:200]}")
            update_narrative_time._failure_count = failure_count + 1
            return previous or NarrativeTime()

        update_narrative_time._failure_count = 0

        # Merge partial output over the previous state.
        if previous is not None:
            base = _to_dict(previous)
            # Drop explicit nulls so they can't clobber a carried-over value.
            base.update({k: v for k, v in parsed.items() if v is not None})
            # Monotonic day counter: only ever advances.
            base["day_count"] = _advance_day_count(previous, base)
            new = _from_dict(base)
        else:
            new = _from_dict(parsed)

        return new

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"narrative_time: parse error: {e}")
        update_narrative_time._failure_count = failure_count + 1
        return previous or NarrativeTime()


# 3. CONTEXT INJECTION

def narrative_time_to_context_string(t: NarrativeTime) -> str:
    """
    Compact natural language representation of narrative time. Omit on day 0
    (the opening scene) to avoid cluttering the context before any time has
    passed.
    """
    if t.day_count <= 0:
        return ""
    elapsed = t.day_count - 1
    if elapsed == 0:
        return f"It is {t.time_of_day or 'a new day'} of day {t.day_count} — about a day has passed since the story began."
    return f"It is {t.time_of_day or 'a new day'} of day {t.day_count} — roughly {elapsed} day{'s' if elapsed != 1 else ''} have passed since the story began."