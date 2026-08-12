"""
HyperMEM — Full Benchmark Harness

Runs HyperMEM the way AI models get benchmarked: multiple capability
suites, repeated across seeds (mean ± std), across multiple LLM backends,
with a machine-readable JSON dump and a markdown leaderboard report.

Suites
------
recall_scaling       recall accuracy (pass@1) vs conversation distance
answer_scaling       end-to-end answer accuracy: normal vs hypermem vs
                     hypermem+worldIDA at every distance
distractors          resistance to competing facts (leak counting)
contradiction        supersession: updated facts must win over stale ones
paraphrase           recall robustness across rephrased questions
latency              p50/p95 latency: extraction, recall, worldIDA, filler
storage              state size growth (messages + per-memory)
worldida_stability   false scene changes + update latency over N turns

Usage
-----
    python benchmarks/full_suite.py --full            # everything, hours
    python benchmarks/full_suite.py --quick           # smoke, ~10-20 min
    python benchmarks/full_suite.py --models qwen2.5:7b,gemma3:12b --seeds 3
    python benchmarks/full_suite.py --suites recall,answer
    python benchmarks/full_suite.py --scales 100,1000,5000,10000 --seeds 5

Prerequisites: a running model backend (default: Ollama on localhost:11434).

Output: benchmark_results_full.json + benchmark_report_full.md
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypermem import HyperMEM, HyperMemConfig

from compare_modes import FACTS, QUERIES, FILLER, PERSONA, ANSWER_SYSTEM, answer_question
from run_benchmarks import DISTRACTORS

RESULTS: dict = {}
SUITE_ORDER = ["recall_scaling", "answer_scaling", "distractors", "contradiction",
               "paraphrase", "latency", "storage", "worldida_stability"]

DEFAULT_MODEL = "qwen2.5:7b"
ALL_MODELS = ["qwen2.5:7b", "gemma3:12b"]
FULL_SCALES = [100, 1000, 5000, 10000, 25000, 50000]
QUICK_SCALES = [100, 1000]

PARAPHRASES = [
    "What's my name?", "Who am I?",
    "What's my bow?", "What weapon do I carry?",
    "What are we searching for?", "What is our quest?",
    "Where is the crown hidden?", "Where do we find the crown?",
    "Who killed my father?", "Who murdered my dad?",
    "What does the crown do?", "What powers does the crown have?",
    "What's my sister's name?", "Who is Lyra?",
    "Where does my sister live?", "Where is Lyra's home?",
    "What's the vault password?", "What is the passphrase?",
    "What am I afraid of?", "What is my fear?",
    "Who blessed Moonwhisper?", "Who enchanted my bow?",
    "Shadow King's true name?", "What is Malachar's real identity?",
]


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def make_config() -> HyperMemConfig:
    return HyperMemConfig(auto_tag_threshold=0.3, max_active_memories=500)


async def plant_facts(hm: HyperMEM, rng: random.Random, scale: int,
                      with_ida: bool = False) -> None:
    facts = list(FACTS)
    rng.shuffle(facts)
    for f in facts:
        await hm.add_message("user", f)
        if with_ida:
            await hm.update_world_ida(f, "*Listens attentively.*", PERSONA)
    for i in range(scale - len(facts)):
        await hm.add_message("user", FILLER[i % len(FILLER)])


async def suite_recall_scaling(cfg: HyperMemConfig, rng: random.Random,
                               scales: list[int]) -> dict:
    rows = {}
    for scale in scales:
        hm = HyperMEM(cfg)
        await plant_facts(hm, rng, scale)
        passed, total, lat = 0, 0, []
        for query, keyword in QUERIES:
            start = time.perf_counter()
            recall = await hm.recall(query)
            lat.append((time.perf_counter() - start) * 1000)
            passed += int(any(keyword in m.content.lower() for m in recall.relevant))
            total += 1
        rows[f"recall@{scale}"] = round(passed / total, 3)
        rows[f"recall_latency_ms@{scale}"] = round(statistics.mean(lat), 1)
        await hm.close()
    return rows


async def suite_answer_scaling(cfg: HyperMemConfig, rng: random.Random,
                               scales: list[int]) -> dict:
    rows = {}
    for mode in ["normal", "hypermem", "hypermem_ida"]:
        for scale in scales:
            hm = HyperMEM(cfg if mode == "hypermem_ida" or mode == "hypermem"
                          else HyperMemConfig(auto_tagging=False, max_active_memories=500))
            if mode == "hypermem_ida":
                from hypermem.world_ida import WorldIDA
                hm.set_world_ida(WorldIDA())
            await plant_facts(hm, rng, scale, with_ida=(mode == "hypermem_ida"))
            passed, total, lats, tokens = 0, 0, [], []
            for query, keyword in QUERIES:
                ctx, answer, elapsed = await answer_question(hm, hm._llm, query, mode)
                passed += int(keyword in answer.lower())
                total += 1
                lats.append(elapsed)
                tokens.append(len(ctx) // 4)
            rows[f"answer_{mode}@{scale}"] = round(passed / total, 3)
            rows[f"answer_{mode}_latency_ms@{scale}"] = round(statistics.mean(lats), 1)
            rows[f"answer_{mode}_ctx_tokens@{scale}"] = round(statistics.mean(tokens), 1)
            await hm.close()
    return rows


async def suite_distractors(cfg: HyperMemConfig, rng: random.Random,
                            scales: list[int]) -> dict:
    rows = {}
    for scale in scales:
        hm = HyperMEM(cfg)
        for f in FACTS + DISTRACTORS:
            await hm.add_message("user", f)
        for i in range(scale):
            await hm.add_message("user", FILLER[i % len(FILLER)])
        tests = [
            ("What's my name?", "eldrin", "cousin"),
            ("What's my bow called?", "moonwhisper", "starfall"),
            ("Where is the crown?", "dragon's maw", "sunken temple"),
            ("Who killed my father?", "shadow king", "ice king"),
            ("What does the crown do?", "control the weather", "invisibility"),
        ]
        passed, leaks, total = 0, 0, len(tests)
        for query, correct, wrong in tests:
            recall = await hm.recall(query)
            has_correct = any(correct in m.content.lower() for m in recall.relevant)
            has_wrong = any(wrong in m.content.lower() for m in recall.relevant)
            passed += int(has_correct and not has_wrong)
            leaks += int(has_wrong)
        rows[f"distractor_accuracy@{scale}"] = round(passed / total, 3)
        rows[f"distractor_leaks@{scale}"] = leaks
        await hm.close()
    return rows


async def suite_contradiction(cfg: HyperMemConfig, rng: random.Random,
                              scales: list[int]) -> dict:
    rows = {}
    for scale in scales:
        hm = HyperMEM(cfg)
        await hm.add_message("user", "Vault password is 'Starlight through the darkness'.")
        await hm.add_message("user", "Actually, I changed the vault password to 'Midnight'.")
        for i in range(scale):
            await hm.add_message("user", FILLER[i % len(FILLER)])
        ctx, answer, _ = await answer_question(hm, hm._llm, "What's the vault password?", "hypermem")
        rows[f"contradiction_new_win@{scale}"] = int("midnight" in answer.lower())
        rows[f"contradiction_stale_leak@{scale}"] = int("starlight" in answer.lower())
        await hm.close()
    return rows


async def suite_paraphrase(cfg: HyperMemConfig, rng: random.Random,
                           scales: list[int]) -> dict:
    rows = {}
    for scale in scales:
        hm = HyperMEM(cfg)
        await plant_facts(hm, rng, scale)
        hits, total = 0, 0
        for i in range(0, len(PARAPHRASES), 2):
            for q in PARAPHRASES[i:i + 2]:
                keyword = QUERIES[i // 2][1]
                recall = await hm.recall(q)
                hits += int(any(keyword in m.content.lower() for m in recall.relevant))
                total += 1
        rows[f"paraphrase_recall@{scale}"] = round(hits / total, 3)
        await hm.close()
    return rows


async def suite_latency(cfg: HyperMemConfig, rng: random.Random,
                        scales: list[int]) -> dict:
    hm = HyperMEM(cfg)
    llm = hm._llm
    await plant_facts(hm, rng, 100)

    async def samples(n, fn):
        return [await _timed(fn) for _ in range(n)]

    async def _timed(fn):
        start = time.perf_counter()
        await fn()
        return (time.perf_counter() - start) * 1000

    extract = await samples(5, lambda: hm.add_message("user", "I have a pet wolf named Shadow."))
    recall = await samples(5, lambda: hm.recall("What's my name?"))
    filler = await samples(100, lambda: hm.add_message("user", "Ok."))
    ida = await samples(
        5,
        lambda: hm.update_world_ida(
            "I step into the tavern.", "*She nods.*", PERSONA,
        ),
    )
    await hm.close()

    rows = {}
    for name, xs in [("extraction", extract), ("recall", recall),
                     ("worldida", ida), ("filler", filler)]:
        rows[f"latency_{name}_p50_ms"] = round(statistics.median(xs), 1)
        rows[f"latency_{name}_p95_ms"] = round(sorted(xs)[int(len(xs) * 0.95) - 1], 1)
    return rows


async def suite_storage(cfg: HyperMemConfig, rng: random.Random,
                        scales: list[int]) -> dict:
    rows = {}
    hm = HyperMEM(cfg)
    for scale in scales:
        for i in range(scale - hm.state.total_messages):
            if hm.state.total_messages < len(FACTS):
                await hm.add_message("user", FACTS[hm.state.total_messages])
            else:
                await hm.add_message("user", FILLER[i % len(FILLER)])
        kb = len(json.dumps(hm.to_dict()).encode()) / 1024
        rows[f"state_kb@{scale}"] = round(kb, 1)
    await hm.close()

    for count in [100, 1000, 5000, 10000]:
        hm = HyperMEM(cfg)
        for i in range(count):
            hm.remember(f"Memory #{i}: user prefers topic {i} in region {i % 7}")
        kb = len(json.dumps(hm.to_dict()).encode()) / 1024
        rows[f"bytes_per_memory@{count}"] = round(kb * 1024 / count, 1)
        await hm.close()
    return rows


async def suite_worldida_stability(cfg: HyperMemConfig, rng: random.Random,
                                   scales: list[int], turns: int) -> dict:
    hm = HyperMEM(cfg)
    llm = hm._llm
    prev = None
    false_changes, lats = 0, []
    for turn in range(turns):
        user_msg = f"Message number {turn} in this conversation."
        ai_msg = f"*Responds naturally.* This is response {turn}."
        start = time.perf_counter()
        new = await hm.update_world_ida(user_msg, ai_msg, PERSONA)
        lats.append((time.perf_counter() - start) * 1000)
        if prev is not None and new.meta.scene_changed:
            if prev.scene.location == new.scene.location:
                false_changes += 1
        prev = new
    await hm.close()
    return {
        "worldida_false_scene_changes": false_changes,
        "worldida_update_p50_ms": round(statistics.median(lats), 1),
    }


SUITES = {
    "recall_scaling": suite_recall_scaling,
    "answer_scaling": suite_answer_scaling,
    "distractors": suite_distractors,
    "contradiction": suite_contradiction,
    "paraphrase": suite_paraphrase,
    "latency": suite_latency,
    "storage": suite_storage,
    "worldida_stability": suite_worldida_stability,
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

async def run_suite(name: str, model: str, endpoint: str, seed: int,
                    scales: list[int], turns: int) -> dict:
    rng = random.Random(seed)
    cfg = make_config()
    cfg.llm_model = model
    cfg.llm_endpoint = endpoint
    fn = SUITES[name]
    if name == "worldida_stability":
        return await fn(cfg, rng, scales, turns)
    return await fn(cfg, rng, scales)


def render_report() -> str:
    lines = [
        "# HyperMEM — Benchmark Report",
        "",
        f"Model(s): {', '.join(RESULTS['meta']['models'])}  |  "
        f"scales: {RESULTS['meta']['scales']}  |  "
        f"seeds: {RESULTS['meta']['seeds']}  |  "
        f"platform: {RESULTS['meta']['platform']}",
        "",
    ]
    for suite in SUITE_ORDER:
        if suite not in RESULTS["suites"]:
            continue
        lines.append(f"## {suite}")
        lines.append("")
        lines.append("| metric | " + " | ".join(RESULTS["meta"]["models"]) + " |")
        lines.append("|--------|" + "--------|" * len(RESULTS["meta"]["models"]))
        metrics = sorted(RESULTS["suites"][suite].keys())
        for m in metrics:
            cells = []
            for model in RESULTS["meta"]["models"]:
                agg = RESULTS["suites"][suite][m][model]
                cells.append(f"{agg['mean']} ± {agg['std']}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def build_results(runs: list[dict]) -> dict:
    """Aggregate per-(suite, metric, model) mean/std from raw runs."""
    suites: dict = {}
    for r in runs:
        for metric, val in r["result"].items():
            suites.setdefault(r["suite"], {}).setdefault(metric, {}).setdefault(r["model"], []).append(val)
    out = {}
    for suite, metrics in suites.items():
        out[suite] = {}
        for metric, by_model in metrics.items():
            out[suite][metric] = {
                model: {
                    "mean": round(statistics.mean(vals), 3),
                    "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                }
                for model, vals in by_model.items()
            }
    return out


async def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--suites", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--scales", type=str, default="")
    parser.add_argument("--turns", type=int, default=0)
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--resume", action="store_true",
                        help="skip runs already present in --out (crash recovery)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "benchmark_results_full.json"))
    parser.add_argument("--report", default=str(Path(__file__).parent / "benchmark_report_full.md"))
    args = parser.parse_args()

    if args.full:
        scales, models, seeds, turns = FULL_SCALES, ALL_MODELS, 3, 100
    elif args.quick:
        scales, models, seeds, turns = QUICK_SCALES, [DEFAULT_MODEL], 1, 25
    else:
        scales, models, seeds, turns = QUICK_SCALES, [DEFAULT_MODEL], 1, 25
    if args.scales:
        scales = [int(s) for s in args.scales.split(",") if s.strip()]
    if args.models:
        models = [m for m in args.models.split(",") if m.strip()]
    if args.seeds:
        seeds = args.seeds
    if args.turns:
        turns = args.turns
    suites = [s for s in args.suites.split(",") if s.strip()] or SUITE_ORDER
    invalid = set(suites) - set(SUITES)
    if invalid:
        parser.error(f"unknown suites: {', '.join(sorted(invalid))}")

    runs: list[dict] = []
    if args.resume and Path(args.out).exists():
        saved = json.loads(Path(args.out).read_text(encoding="utf-8"))
        runs = saved.get("runs", [])
        print(f"Resuming: {len(runs)} runs already completed")

    done = {(r["model"], r["suite"], r["seed"]) for r in runs}
    total_runs = len(models) * len(suites) * seeds
    print("=" * 64)
    print("HyperMEM — Full Benchmark")
    print("=" * 64)
    print(f"Models:  {models}")
    print(f"Suites:  {suites}")
    print(f"Seeds:   {seeds}   |   runs: {total_runs} ({total_runs - len(done)} remaining)")
    print(f"Scales:  {scales}   |   turns: {turns}")
    print(f"Endpoint:{args.endpoint}")

    start_all = time.perf_counter()
    for model in models:
        print(f"\n### model: {model}")
        for suite in suites:
            for seed in range(1, seeds + 1):
                if (model, suite, seed) in done:
                    print(f"  [skip] {suite} seed={seed} (already done)")
                    continue
                t0 = time.perf_counter()
                print(f"  {suite} seed={seed} ...", flush=True)
                r = await run_suite(suite, model, args.endpoint, seed, scales, turns)
                runs.append({"model": model, "suite": suite, "seed": seed, "result": r})
                print(f"      done in {time.perf_counter() - t0:.0f}s", flush=True)
                _checkpoint(args.out, args.report, runs, models, suites, seeds,
                            scales, turns, args.endpoint, start_all)

    _checkpoint(args.out, args.report, runs, models, suites, seeds,
                scales, turns, args.endpoint, start_all)
    n = len(runs)
    print(f"\n{'=' * 64}")
    print(f"Completed {n} runs in {round(time.perf_counter() - start_all)}s")
    print(f"JSON:   {args.out}")
    print(f"Report: {args.report}")
    print("=" * 64)


def _checkpoint(out_path, report_path, runs, models, suites, seeds, scales,
                turns, endpoint, start_all) -> None:
    data = {
        "runs": runs,
        "suites": build_results(runs),
        "meta": {
            "models": models,
            "suites": suites,
            "seeds": seeds,
            "scales": scales,
            "turns": turns,
            "endpoint": endpoint,
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "wall_seconds": round(time.perf_counter() - start_all, 1),
        },
    }
    RESULTS["runs"] = runs
    RESULTS["suites"] = data["suites"]
    RESULTS["meta"] = data["meta"]
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    Path(report_path).write_text(render_report(), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
