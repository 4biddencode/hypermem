"""
HyperMEM — Benchmark Suite

Runs against a REAL LLM via LLMClient (Ollama/OpenAI/Anthropic) and reports:

1. Recall accuracy vs. distance (scales)
2. Recall accuracy with distractors (similar/competing facts)
3. Latency per call type (extraction, recall, worldIDA update, filler fast-path)
4. Storage growth curve
5. worldIDA stability over N turns

Usage:
    python benchmarks/run_benchmarks.py                # full scales
    python benchmarks/run_benchmarks.py --quick         # small scales
    python benchmarks/run_benchmarks.py --scales 100,1000,5000
    python benchmarks/run_benchmarks.py --out results.json

Prerequisites: a running model backend (default: Ollama on localhost:11434).

Output: JSON written to benchmarks/benchmark_results.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypermem import HyperMEM, HyperMemConfig
from hypermem.world_ida import update_world_ida

RESULTS: dict = {}


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

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

DISTRACTORS = [
    "My name is Eldrin's cousin, from the same forest.",
    "There is a different bow called Starfall.",
    "Another crown was hidden in the Sunken Temple.",
    "The Ice King also wanted the crown.",
    "The crown can also grant invisibility.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> HyperMemConfig:
    return HyperMemConfig(auto_tag_threshold=0.3, **overrides)


def make_engine(config: HyperMemConfig) -> HyperMEM:
    return HyperMEM(config)


# ---------------------------------------------------------------------------
# 1. Recall accuracy vs. distance
# ---------------------------------------------------------------------------

async def bench_recall_accuracy(scales: list[int]):
    print("\n=== Benchmark 1: Recall Accuracy vs. Distance ===")

    results = []
    cfg = make_config(max_active_memories=500)
    for scale in scales:
        hm = make_engine(cfg)
        for f in FACTS:
            await hm.add_message("user", f)
        for i in range(scale - len(FACTS)):
            await hm.add_message("user", FILLER[i % len(FILLER)])

        elapsed_all, (passed, hits) = await _score_queries(hm)
        pct = round(passed / len(QUERIES) * 100)
        print(f"  {scale:>6} messages: {passed}/{len(QUERIES)} = {pct}%  ({elapsed_all:.0f}ms)")
        results.append({
            "scale": scale, "passed": passed, "total": len(QUERIES),
            "pct": pct, "recall_ms": elapsed_all,
        })
    RESULTS["recall_accuracy"] = results
    return results


async def _score_queries(hm, queries=None):
    queries = queries or QUERIES
    passed = 0
    hits = 0
    start = time.perf_counter()
    for query, keyword in queries:
        recall = await hm.recall(query)
        hit = any(keyword in m.content.lower() for m in recall.relevant)
        hits += int(hit)
        passed += int(hit)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, (passed, hits)


# ---------------------------------------------------------------------------
# 2. Recall with distractors
# ---------------------------------------------------------------------------

async def bench_recall_distractors():
    print("\n=== Benchmark 2: Recall with Distractors ===")

    cfg = make_config()
    hm = make_engine(cfg)

    for f in FACTS + DISTRACTORS:
        await hm.add_message("user", f)
    for i in range(500):
        await hm.add_message("user", FILLER[i % len(FILLER)])

    tests = [
        ("What's my name?", "eldrin", "cousin"),
        ("What's my bow called?", "moonwhisper", "starfall"),
        ("Where is the crown?", "dragon's maw", "sunken temple"),
        ("Who killed my father?", "shadow king", "ice king"),
    ]

    passed, wrong_hits = 0, 0
    for query, correct_kw, wrong_kw in tests:
        recall = await hm.recall(query)
        has_correct = any(correct_kw in m.content.lower() for m in recall.relevant)
        has_wrong = any(wrong_kw in m.content.lower() for m in recall.relevant)
        ok = has_correct and not has_wrong
        passed += int(ok)
        wrong_hits += int(has_wrong)
        print(f"  {'PASS' if ok else 'FAIL'} \"{query}\"")

    print(f"  Result: {passed}/{len(tests)} passed, {wrong_hits} distractor leaks")
    RESULTS["recall_distractors"] = {
        "passed": passed, "total": len(tests), "distractor_leaks": wrong_hits,
    }


# ---------------------------------------------------------------------------
# 3. Latency per call type
# ---------------------------------------------------------------------------

async def bench_latency():
    print("\n=== Benchmark 3: Latency per Call Type ===")

    cfg = make_config()
    hm = make_engine(cfg)
    llm = hm._llm

    for f in FACTS[:5]:
        await hm.add_message("user", f)

    async def sample(n, fn):
        times = []
        for _ in range(n):
            start = time.perf_counter()
            await fn()
            times.append((time.perf_counter() - start) * 1000)
        return sum(times) / len(times)

    avg_extract = await sample(5, lambda: hm.add_message("user", "I have a pet wolf named Shadow."))
    avg_recall = await sample(5, lambda: hm.recall("What's my name?"))
    avg_ida = await sample(
        5,
        lambda: update_world_ida(
            None, "Hello", "*She nods.*", "Elena, elven rogue.",
            llm_complete=llm.complete,
        ),
    )
    avg_filler = await sample(100, lambda: hm.add_message("user", "Ok."))

    print(f"  add_message (extraction): {avg_extract:.1f}ms avg")
    print(f"  recall:                   {avg_recall:.1f}ms avg")
    print(f"  worldIDA update:          {avg_ida:.1f}ms avg")
    print(f"  filler fast-path:         {avg_filler:.3f}ms avg")

    RESULTS["latency"] = {
        "extraction_ms": round(avg_extract, 1),
        "recall_ms": round(avg_recall, 1),
        "world_ida_ms": round(avg_ida, 1),
        "filler_ms": round(avg_filler, 3),
    }


# ---------------------------------------------------------------------------
# 4. Storage growth
# ---------------------------------------------------------------------------

async def bench_storage(scales: list[int]):
    print("\n=== Benchmark 4: Storage Growth (conversation length) ===")

    cfg = make_config()
    hm = make_engine(cfg)
    results = []

    for scale in scales:
        for i in range(scale - hm.state.total_messages):
            if hm.state.total_messages < len(FACTS):
                await hm.add_message("user", FACTS[hm.state.total_messages])
            else:
                await hm.add_message("user", FILLER[i % len(FILLER)])

        payload = json.dumps(hm.to_dict())
        kb = round(len(payload.encode()) / 1024, 1)
        print(f"  {scale:>6} messages: {kb} KB  (conversational filler → bounded)")
        results.append({"scale": scale, "size_kb": kb})

    RESULTS["storage_growth"] = results
    return results


async def bench_storage_per_memory(scales: list[int]):
    """Storage cost per stored memory — planted via the real remember() API."""
    print("\n=== Benchmark 4b: Storage per Memory ===")

    cfg = make_config()
    results = []

    for scale in scales:
        hm = make_engine(cfg)
        for i in range(scale):
            hm.remember(f"Memory #{i}: user prefers topic {i} in region {i % 7}")
        payload = json.dumps(hm.to_dict())
        kb = round(len(payload.encode()) / 1024, 2)
        print(f"  {scale:>6} memories: {kb} KB  ({round(kb * 1024 / scale, 1)} B/memory)")
        results.append({"scale": scale, "size_kb": kb,
                        "bytes_per_memory": round(kb * 1024 / scale, 1)})

    RESULTS["storage_per_memory"] = results
    return results


# ---------------------------------------------------------------------------
# 5. worldIDA stability
# ---------------------------------------------------------------------------

async def bench_worldida_stability(turns: int):
    print(f"\n=== Benchmark 5: worldIDA Stability ({turns} turns) ===")

    cfg = make_config()
    hm = make_engine(cfg)
    llm = hm._llm

    prev_ida = None
    drift_count = 0
    for turn in range(turns):
        user_msg = f"Message number {turn} in this conversation."
        ai_msg = f"*Responds naturally.* This is response {turn}."
        new_ida = await update_world_ida(
            prev_ida, user_msg, ai_msg,
            "Elena, elven rogue. Sharp-witted, guarded.",
            llm_complete=llm.complete,
        )
        if prev_ida is not None and new_ida.meta.scene_changed:
            if prev_ida.scene.location == new_ida.scene.location:
                drift_count += 1
        prev_ida = new_ida

    print(f"  Total turns: {turns}, false scene changes: {drift_count}")
    RESULTS["worldida_stability"] = {
        "turns": turns,
        "false_scene_changes": drift_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FULL_SCALES = [100, 500, 1000, 2000, 5000, 10000]
QUICK_SCALES = [100, 500, 1000]


async def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="small scales, faster runtime")
    parser.add_argument("--scales", type=str, default="",
                        help="comma-separated message scales (overrides defaults)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "benchmark_results.json"))
    parser.add_argument("--model", default=None,
                        help="model name (default: HyperMemConfig default)")
    parser.add_argument("--endpoint", default=None,
                        help="LLM endpoint (default: http://localhost:11434)")
    args = parser.parse_args()

    scales = [int(s) for s in args.scales.split(",") if s.strip()] or (
        QUICK_SCALES if args.quick else FULL_SCALES
    )
    turns = 25 if args.quick else 100

    overrides = {}
    if args.model:
        overrides["llm_model"] = args.model
    if args.endpoint:
        overrides["llm_endpoint"] = args.endpoint

    print("=" * 60)
    print("HyperMEM Benchmark Suite")
    print("=" * 60)
    print(f"Scales: {scales}   |   turns: {turns}")
    print(f"Model:  {overrides.get('llm_model', HyperMemConfig().llm_model)}")
    print(f"Python: {sys.version.split()[0]}")

    out_cfg = make_config(**overrides)
    hm = make_engine(out_cfg)

    await bench_recall_accuracy(scales)
    await bench_recall_distractors()
    await bench_latency()
    await bench_storage(scales)
    await bench_storage_per_memory([100, 500, 1000, 2000, 5000, 10000])
    await bench_worldida_stability(turns)

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