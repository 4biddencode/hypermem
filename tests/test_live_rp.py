"""
Live roleplay session test for worldIDA.

Runs 20-30 turns of actual RP through Ollama, logs every worldIDA diff,
checks for field drift, confidence tracking, and scene change behavior.
"""

import asyncio
import json
import time
import pytest
from hypermem import HyperMEM, HyperMemConfig
from hypermem.world_ida import (
    WorldIDA, world_ida_to_context_string,
    _ida_to_dict, _ida_from_dict,
)

pytestmark = pytest.mark.live


def diff_ida(before: WorldIDA, after: WorldIDA) -> dict:
    """Return a dict of what changed between two WorldIDA states."""
    b = _ida_to_dict(before)
    a = _ida_to_dict(after)
    changes = {}

    for section in ["scene", "user", "character", "relationship", "meta"]:
        b_sec = b.get(section, {})
        a_sec = a.get(section, {})
        sec_changes = {}
        all_keys = set(b_sec.keys()) | set(a_sec.keys())
        for k in all_keys:
            bv = b_sec.get(k)
            av = a_sec.get(k)
            if bv != av:
                sec_changes[k] = {"from": bv, "to": av}
        if sec_changes:
            changes[section] = sec_changes

    return changes


async def run_live_session():
    config = HyperMemConfig(
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        llm_endpoint="http://localhost:11434",
        auto_tag_threshold=0.3,
    )
    hm = HyperMEM(config)

    # Persona for the RP
    persona = """Elena is a elven rogue in her late 20s, sharp-witted and guarded.
    She has a dry sense of humor but warms up slowly. She's on the run from a
    crime syndicate she crossed in the capital. She doesn't trust easily."""

    # Opening scene
    opening = """*The tavern is dimly lit, filled with the murmur of late-night drinkers.
    A woman in a hooded cloak sits alone in a corner booth, nursing a mug of ale.
    She glances up as you approach, her eyes scanning you warily before she offers
    a small, tired nod.*"""

    print("=" * 70)
    print("worldIDA Live RP Session Test")
    print("=" * 70)
    print(f"\nPersona: {persona}")
    print(f"\nOpening: {opening}\n")

    # Conversation script — 25 turns with various events
    script = [
        # Phase 1: Tavern introduction (turns 1-8)
        "Mind if I sit here? Other tables are full.",
        "Thanks. I'm a traveler passing through. What brings you to this corner of the world?",
        "You seem like you've seen better days. Rough road?",
        "I couldn't help but notice the dagger hilt under your cloak. You know how to use that thing?",
        "Fair enough. I'm not looking for trouble either. Just some company for the night.",
        "The barmaid says you've been here for three days. Waiting for someone?",
        "You flinched when I mentioned the capital. That's where you're from, isn't it?",
        "Alright, I'll drop it. But if you're hiding from something, this tavern won't shield you forever.",

        # Phase 2: Moving to a booth (turns 9-12)
        "The corner booth by the fire just opened up. Want to move there? More private.",
        "Cozy spot. Good view of the door too — you can see everyone who comes in.",
        "So if you are in trouble... maybe I can help. I've got connections.",
        "You're smiling. That's the first time tonight. It's a good look on you.",

        # Phase 3: Time skip — next morning (turns 13-17)
        "*The next morning. Sunlight streams through the tavern windows. You're both at a breakfast table.*",
        "Morning. You look better in daylight. Sleep well?",
        "I meant what I said last night about helping. I know a safe route to the coast.",
        "Why are you being kind to me? You don't even know my name.",
        "Elena. That's a pretty name. I'm Marcus.",

        # Phase 4: Forest journey (turns 18-22) — scene change
        "*Later that day, you're both walking through a dense forest on the road to the coast.*",
        "We should reach the coast by nightfall if we keep this pace.",
        "So what did you do in the capital? Before... whatever happened?",
        "You were a court scribe? That's not what I expected. I pictured something more dangerous.",
        "Wait — do you hear that? Footsteps. Someone's following us.",

        # Phase 5: Ambush (turns 23-25) — scene change
        "*Two armed men step onto the path ahead, blocking the way.*",
        "Elena, get behind me. These aren't bandits — they're wearing city guard colors.",
        "They found us. I'm sorry, Marcus. I should have told you everything.",
    ]

    # Initialize worldIDA from opening
    from hypermem.world_ida import update_world_ida as update_ida
    from hypermem.llm import LLMClient

    llm = LLMClient(provider="ollama", model="qwen2.5:7b", endpoint="http://localhost:11434")

    # First turn: initialize worldIDA
    ai_response = opening
    hm._world_ida = await update_ida(
        None, "", opening, persona_context=persona,
        llm_complete=llm.complete,
    )

    print(f"{'Turn':<6} {'Fields Changed':<50} {'Confidence':<12} {'Scene Changed':<15}")
    print("-" * 85)

    prev_ida = hm._world_ida

    for turn, user_msg in enumerate(script, 1):
        # Get AI response
        context = await hm.get_context(user_msg)
        full_prompt = f"{context}\n\nUser: {user_msg}\nCharacter (Elena):"

        ai_resp = await llm.complete(
            [{"role": "system", "content": f"You are Elena. {persona}"},
             {"role": "user", "content": full_prompt}],
            temperature=0.8, max_tokens=200,
        )
        ai_resp = ai_resp or "[no response]"

        # Update HyperMEM
        await hm.add_message("user", user_msg)
        await hm.add_message("assistant", ai_resp)

        # Update worldIDA
        await hm.update_world_ida(user_msg, ai_resp, persona_context=persona)

        # Diff
        changes = diff_ida(prev_ida, hm._world_ida)
        changed_fields = sum(len(v) for v in changes.values())
        conf = hm._world_ida.meta.confidence if hm._world_ida else 0
        scene_changed = hm._world_ida.meta.scene_changed if hm._world_ida else False

        # Log compactly
        fields_str = ", ".join(
            f"{s}.{k}" for s, v in changes.items() for k in v
        )[:50] if changes else "(none)"
        print(f"{turn:<6} {fields_str:<50} {conf:<12.2f} {str(scene_changed):<15}")

        # Every 5 turns, print full state
        if turn % 5 == 0:
            print(f"\n  State after turn {turn}:")
            print(f"  {world_ida_to_context_string(hm._world_ida)}")
            print()

        prev_ida = hm._world_ida

    # Final summary
    print("\n" + "=" * 70)
    print("Session Summary")
    print("=" * 70)
    print(f"Total turns: {len(script)}")
    print(f"Active memories: {len(hm.state.active)}")
    print(f"Total messages: {hm.state.total_messages}")
    print(f"\nFinal world state:")
    print(world_ida_to_context_string(hm._world_ida))

    # Check for field drift
    print("\n\nDrift Analysis:")
    print("-" * 40)
    # Re-run and track all unique field values
    print("(Run complete — check output above for unexpected field changes)")


if __name__ == "__main__":
    asyncio.run(run_live_session())
