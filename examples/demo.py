"""
HyperMEM — "Memory that actually remembers"

An end-to-end demo of the memory layer, using ONLY the public API. It tells a
little story: two people meet, chat for weeks (hundreds of filler turns),
one fact changes mid-way, and HyperMEM keeps the story straight the whole
time — then it explains itself.

Run with a local Ollama (the default) or any OpenAI-compatible endpoint:

    python examples/demo.py                     # qwen2.5:7b via Ollama
    python examples/demo.py --model gemma3:12b
    python examples/demo.py --filler 1000       # simulate ~1000 messages
    python examples/demo.py --embedding none    # force the LLM+lexical path

Everything printed is real output from the pipeline. If Ollama isn't running,
you'll get a clear message instead of a traceback.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# The demo prints ✓/✗/·/—/→ which need UTF-8. Windows consoles default to
# cp1252 and would raise UnicodeEncodeError on ✗ — force UTF-8 so the demo
# runs cleanly anywhere (Windows Terminal, VS Code, POSIX).
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypermem import HyperMEM, HyperMemConfig
from hypermem.types import Persona
from hypermem.llm import LLMClient

PERSONA = "Eldrin, elven ranger from Silverwood. Brave, focused, terse."

# (message, the keyword a correct recall must surface)
FACTS = [
    ("My name is Eldrin, an elven ranger from Silverwood.", "eldrin"),
    ("I carry a bow named Moonwhisper from my father.", "moonwhisper"),
    ("We are searching for the Lost Crown of Aetheria in the Dragon's Maw.",
     "aetheria"),
    ("The crown can control the weather when worn.", "weather"),
    ("My sister Lyra lives in Oakvale village.", "lyra"),
    ("The vault password is 'Starlight through the darkness'.", "starlight"),
    ("I'm afraid of fire since our village burned down.", "fire"),
    ("Shadow King's true name is Malachar.", "malachar"),
]

QUERIES = [
    ("What's my name?", "eldrin"),
    ("What's my bow?", "moonwhisper"),
    ("What are we searching for?", "aetheria"),
    ("What does the crown do?", "weather"),
    ("What's my sister's name?", "lyra"),
    ("What's the vault password?", "starlight"),
    ("What am I afraid of?", "fire"),
    ("Shadow King's true name?", "malachar"),
]

FILLER = ["Ok.", "Hmm.", "I see.", "Yes.", "No.", "Maybe.", "Sure.", "Wait.",
          "Oh.", "Right.", "Alright.", "Cool.", "Nice.", "Good."]

# ---------- display helpers -------------------------------------------------

def line():
    print("-" * 70)


def heading(text: str):
    print()
    line()
    print(f"  {text}")
    line()


def mem_line(m, extra: str = ""):
    tag = {
        "static": "STATIC",
        "episodic": "EPISODIC",
        "temporal": "TEMPORAL",
    }.get(m.memory_type.value if hasattr(m.memory_type, "value") else m.memory_type,
          str(m.memory_type))
    print(f"  [{tag:<9}] imp {m.importance:.2f}  {m.content[:80]}{extra}")


def lifecycle(m, hm) -> str:
    """active / superseded / archived — derived the same way the server does."""
    if m in hm.state.active:
        return "active"
    if m.superseded_by:
        return "superseded"
    return "archived"


# ---------- the story -------------------------------------------------------

async def answer(llm: LLMClient, query: str, context: str) -> tuple[str, float]:
    """Ask the model to answer from HyperMEM's injected context."""
    start = time.perf_counter()
    resp = await llm.complete(
        [
            {"role": "system",
             "content": "Answer in 1-3 words using ONLY the context. "
                        "If it does not contain the answer, reply UNKNOWN."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.0, max_tokens=20,
    )
    elapsed = (time.perf_counter() - start) * 1000
    return (resp or "UNKNOWN").strip(), elapsed


async def demo(filler_count: int, model: str, endpoint: str,
               embedding: str) -> int:
    cfg = HyperMemConfig(
        llm_model=model,
        llm_endpoint=endpoint,
        embedding_provider=embedding,
        auto_tag_threshold=0.3,
        max_active_memories=500,
    )
    hm = HyperMEM(cfg)
    hm.set_persona(Persona(name="Eldrin",
                           description="Elven ranger from Silverwood",
                           traits=["brave", "focused", "terse"]))
    llm = LLMClient(provider=cfg.llm_provider, model=cfg.llm_model,
                    endpoint=cfg.llm_endpoint)
    embed_note = ("semantic (embeddings)" if cfg.embedding_provider != "none"
                  else "LLM + lexical (no embeddings)")

    print("HyperMEM  ·  v1.0.0  ·  an AI memory layer")
    print(f"model      {cfg.llm_model} @ {cfg.llm_endpoint}")
    print(f"recall     {embed_note}")

    try:
        await hm.recall("ping")  # probe the backend
    except Exception as e:
        print(f"\n! Could not reach the model backend: {e}")
        print("  Start Ollama (`ollama serve`) and pull the model, then retry.")
        await llm.close()
        return 1

    # -- Episode 1: the facts ------------------------------------------------
    heading("Episode 1 · Meeting Eldrin — 8 facts planted")
    for msg, _kw in FACTS:
        res = await hm.add_message("user", msg)
        if res.tagged is not None:
            mem_line(res.tagged)
    print(f"\n  stored {len(hm.state.active)} active memories, "
          f"{hm.state.total_messages} messages")

    # -- Episode 2: the long wait --------------------------------------------
    heading(f"Episode 2 · {filler_count} filler messages (the weeks pass)")
    t0 = time.perf_counter()
    for i in range(filler_count):
        await hm.add_message("user", FILLER[i % len(FILLER)])
    dt = time.perf_counter() - t0
    print(f"  added {filler_count} chit-chat messages in {dt:.1f}s — "
          f"filler is gated out of extraction, so it costs almost nothing.")

    # -- Episode 3: does it still remember? ----------------------------------
    heading("Episode 3 · Recall under pressure (facts planted 600+ msgs ago)")
    passed = total = 0
    for query, keyword in QUERIES:
        r = await hm.recall(query)
        hit = any(keyword in m.content.lower() for m in r.relevant)
        passed += int(hit); total += 1
        print(f"  {'✓' if hit else '✗'} \"{query}\"  →  {len(r.relevant)} "
              f"memory(ies)")
    print(f"\n  recall pass@1: {passed}/{total}")

    # -- Episode 4: an LLM actually answers ----------------------------------
    heading("Episode 4 · The model answers from injected context")
    for query, keyword in QUERIES:
        ctx = await hm.get_context(query)
        answer_text, ms = await answer(llm, query, ctx)
        hit = keyword in answer_text.lower()
        print(f"  {'✓' if hit else '✗'} {query:<28} → {answer_text:<14} "
              f"({ms:.0f} ms)")

    # -- Episode 5: the fact that changed ------------------------------------
    heading("Episode 5 · A changed fact must win (supersession)")
    await hm.add_message(
        "user", "Actually, I changed the vault password to 'Midnight'.")
    # find both the old and new password memories
    old = next(m for m in hm.state.active + hm.state.archive
               if "Starlight" in m.content)
    new = next(m for m in hm.state.active
               if "Midnight" in m.content)
    print(f"  old \"Starlight…\"  → lifecycle {lifecycle(old, hm)}, "
          f"superseded_by={old.superseded_by}")
    print(f"  new \"Midnight\"    → lifecycle {lifecycle(new, hm)}")
    r = await hm.recall("What's the vault password?")
    print("  recall returns: " +
          ", ".join(m.content for m in r.relevant))
    stale = any("Starlight" in m.content for m in r.relevant)
    print(f"  stale leak: {stale}  (must be False)")

    # -- Episode 6: provenance ----------------------------------------------
    heading("Episode 6 · Why did that surface? (provenance)")
    print(f"  memory id    {new.id}")
    print(f"  stored from  \"{new.content[:60]}\"  "
          f"(message {new.source_message_id})")
    breakdown = await hm.explain_recall("What's the vault password?", new.id)
    if breakdown:
        print("  live ranking vs the query:")
        for k, v in breakdown.items():
            print(f"      {k:<15} {v:+.2f}")

    # -- Episode 7: the world keeps moving -----------------------------------
    heading("Episode 7 · worldIDA — the world state updates every turn")
    await hm.update_world_ida(
        "Let's go into the forest toward the Dragon's Maw.",
        "*Eldrin checks his bow and nods.*", PERSONA,
    )
    ida = hm.get_world_ida()
    print(f"  scene     {ida.scene.location} — {ida.scene.ongoing_action}")
    print(f"  meta      turn {ida.meta.turn_count_in_scene}, "
          f"scene_changed={ida.meta.scene_changed}")

    # -- closing ---------------------------------------------------------------
    heading("Closing")
    n_active = len(hm.state.active)
    n_archive = len(hm.state.archive)
    kb = len(json.dumps(hm.to_dict()).encode()) / 1024
    print(f"  {hm.state.total_messages} messages processed → "
          f"{n_active} active + {n_archive} archived memories ({kb:.1f} KB state)")
    print(f"  public API used: add_message, recall, get_context, "
          f"explain_recall,\n                    update_world_ida, "
          f"set_persona, save/load, memories")
    await hm.close()
    await llm.close()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--endpoint", default="http://localhost:11434")
    p.add_argument("--filler", type=int, default=600,
                   help="filler messages to simulate (default 600)")
    p.add_argument("--embedding", default="auto",
                   choices=["auto", "ollama", "openai", "none"])
    args = p.parse_args()
    try:
        sys.exit(asyncio.run(demo(args.filler, args.model, args.endpoint,
                                  args.embedding)))
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
