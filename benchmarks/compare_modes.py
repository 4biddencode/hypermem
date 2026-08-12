"""
HyperMEM — Mode Comparison Benchmark

End-to-end comparison of three context-injection modes against a REAL LLM:

1. normal       — no memory system. The model sees only the recent chat
                  window. Facts planted long ago are out of reach.
2. hypermem     — HyperMEM recalls relevant memories and injects them
                  alongside the recent chat.
3. hypermem_ida — hypermem + worldIDA: a live world-state object is also
                  injected in full on every turn.

Metric: can the model ANSWER a question about a fact planted N messages
ago (keyword hit in its answer)? Plus per-query latency and context size.

Usage:
    python benchmarks/compare_modes.py                # full scales
    python benchmarks/compare_modes.py --quick        # [100, 500, 1000]
    python benchmarks/compare_modes.py --scales 100,1000,5000
    python benchmarks/compare_modes.py --model gemma3:12b
    python benchmarks/compare_modes.py --only normal,hypermem

Prerequisites: a running model backend (default: Ollama on localhost:11434).

Output: JSON written to benchmarks/benchmark_results_compare.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypermem import HyperMEM, HyperMemConfig

RESULTS: dict = {}

FACTS = [
    "My name is Eldrin, an elven ranger from Silverwood.",
    "I carry a bow named Moonwhisper from my father.",
    "Searching for the Lost Crown of Aetheria in the Dragon's Maw.",
    "Father killed by the Shadow King five years ago.",
    "The crown can control the weather when worn.",
    "My sister Lyra lives in Oakvale village.",
    "Vault password is 'Starlight through the darkness'.",
    "I'm afraid of fire since our village burned down.",
    "Moonwhisper was blessed by the High Elves of Sunhaven.",
    "Shadow King's true name is Malachar.",
]

QUERIES = [
    ("What's my name?", "eldrin"),
    ("What's my bow?", "moonwhisper"),
    ("What are we searching for?", "crown of aetheria"),
    ("Where is the crown hidden?", "dragon's maw"),
    ("Who killed my father?", "shadow king"),
    ("What does the crown do?", "control the weather"),
    ("What's my sister's name?", "lyra"),
    ("Where does my sister live?", "oakvale"),
    ("What's the vault password?", "starlight"),
    ("What am I afraid of?", "fire"),
    ("Who blessed Moonwhisper?", "sunhaven"),
    ("Shadow King's true name?", "malachar"),
]

FILLER = ["Ok.", "Hmm.", "I see.", "Yes.", "No.", "Maybe.", "Sure.", "Wait.", "Oh.", "Right."]

PERSONA = "Eldrin, elven ranger from Silverwood. Brave, focused, terse."
ANSWER_SYSTEM = (
    "Answer the question in 1-3 words using ONLY the provided context. "
    "If the context does not contain the answer, reply exactly UNKNOWN."
)


def make_config() -> HyperMemConfig:
    return HyperMemConfig(auto_tag_threshold=0.3, max_active_memories=500)


async def plant(hm: HyperMEM, scale: int) -> None:
    """Plant facts, then fill the conversation to `scale` messages."""
    for i, f in enumerate(FACTS):
        await hm.add_message("user", f)
        if hm.get_world_ida() is not None:
            await hm.update_world_ida(f, "*Listens attentively.*", PERSONA)
    for i in range(scale - len(FACTS)):
        await hm.add_message("user", FILLER[i % len(FILLER)])


def recent_chat(hm: HyperMEM) -> str:
    msgs = hm.state.recent_messages[-10:]
    return "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in msgs
    )


async def answer_question(hm: HyperMEM, llm, query: str, mode: str) -> tuple[str, str, float]:
    if mode == "normal":
        ctx = recent_chat(hm)
    else:
        ctx = await hm.get_context(query)

    start = time.perf_counter()
    resp = await llm.complete(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
        ],
        temperature=0.0, max_tokens=20,
    )
    elapsed = (time.perf_counter() - start) * 1000
    answer = (resp or "UNKNOWN").strip()
    return ctx, answer, elapsed


async def bench_mode(mode: str, scales: list[int]):
    print(f"\n{'=' * 60}")
    print(f"MODE: {mode}")
    print("=" * 60)

    cfg = make_config()
    if mode == "normal":
        cfg = HyperMemConfig(auto_tagging=False, max_active_memories=500)

    results = []
    for scale in scales:
        hm = HyperMEM(cfg)
        llm = hm._llm
        if mode == "hypermem_ida":
            from hypermem.world_ida import WorldIDA
            hm.set_world_ida(WorldIDA())  # enable the world-state block in context

        await plant(hm, scale)

        passed, total, lat_ms, ctx_tokens = 0, 0, 0.0, 0
        misses = []
        for query, keyword in QUERIES:
            ctx, answer, elapsed = await answer_question(hm, llm, query, mode)
            hit = keyword in answer.lower()
            passed += int(hit)
            total += 1
            lat_ms += elapsed
            ctx_tokens += max(1, len(ctx) // 4)
            if not hit:
                misses.append((query, answer))

        avg_lat = lat_ms / total
        avg_ctx_tokens = ctx_tokens / total
        pct = round(passed / total * 100)
        print(f"  {scale:>6} msgs: {passed}/{total} = {pct}%  "
              f"({avg_lat:.0f}ms/q, {avg_ctx_tokens:.0f} tok ctx)")
        if misses:
            print(f"         misses: {', '.join(q for q, _ in misses)}")

        results.append({
            "scale": scale, "passed": passed, "total": total,
            "pct": pct, "avg_query_ms": round(avg_lat, 1),
            "avg_context_tokens": round(avg_ctx_tokens, 1),
            "misses": [{"query": q, "answer": a} for q, a in misses],
        })
        await hm.close()

    RESULTS[mode] = results


async def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated modes: normal,hypermem,hypermem_ida")
    parser.add_argument("--out", default=str(Path(__file__).parent / "benchmark_results_compare.json"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpoint", default=None)
    args = parser.parse_args()

    scales = [int(s) for s in args.scales.split(",") if s.strip()] or (
        [100, 500, 1000] if args.quick else [100, 500, 1000, 2000, 5000, 10000]
    )
    modes = [m for m in args.only.split(",") if m.strip()] or ["normal", "hypermem", "hypermem_ida"]
    invalid = set(modes) - {"normal", "hypermem", "hypermem_ida"}
    if invalid:
        parser.error(f"unknown modes: {', '.join(sorted(invalid))}")

    overrides = {}
    if args.model:
        overrides["llm_model"] = args.model
    if args.endpoint:
        overrides["llm_endpoint"] = args.endpoint

    print("=" * 60)
    print("HyperMEM — Mode Comparison")
    print("=" * 60)
    print(f"Modes:  {modes}")
    print(f"Scales: {scales}")
    print(f"Model:  {overrides.get('llm_model', HyperMemConfig().llm_model)}")
    print(f"Python: {sys.version.split()[0]}")

    for mode in modes:
        await bench_mode(mode, scales)

    RESULTS["meta"] = {
        "model": overrides.get("llm_model", HyperMemConfig().llm_model),
        "endpoint": overrides.get("llm_endpoint", HyperMemConfig().llm_endpoint),
        "scales": scales,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }

    out = Path(args.out)
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {out}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
