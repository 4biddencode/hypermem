"""Diagnostic: why does the distractor suite leak?

Plants FACTS + DISTRACTORS (+ filler) the way suite_distractors does, then
for each test query prints the full hybrid-score breakdown of every memory,
plus whether the distractors were stored as separate memories or near-dup
deduped at ingest. Tells us exactly which tuning lever fixes the leak.

Run AFTER the benchmark (contends for Ollama):
    python benchmarks/diag_distractors.py
"""

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypermem import HyperMEM, HyperMemConfig
from compare_modes import FACTS, FILLER
from run_benchmarks import DISTRACTORS

TESTS = [
    ("What's my name?", "eldrin", "cousin"),
    ("What's my bow called?", "moonwhisper", "starfall"),
    ("Where is the crown?", "dragon's maw", "sunken temple"),
    ("Who killed my father?", "shadow king", "ice king"),
    ("What does the crown do?", "control the weather", "invisibility"),
]


def cfg_for(model: str) -> HyperMemConfig:
    return HyperMemConfig(auto_tag_threshold=0.3, max_active_memories=500,
                          llm_model=model,
                          llm_endpoint="http://localhost:11434")


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    cfg = cfg_for(model)
    hm = HyperMEM(cfg)
    for f in FACTS + DISTRACTORS:
        res = await hm.add_message("user", f)
        flag = "NEW" if (res.tagged and res.tagged.content == f) else "dedup/refresh"
        print(f"  [{'stored' if flag=='NEW' else flag}] {f[:70]}")
    for i in range(100):
        await hm.add_message("user", FILLER[i % len(FILLER)])

    print(f"\n== {len(hm.state.active)} active memories ==")
    for m in hm.state.active:
        print(f"   {m.id[:8]} {m.content[:60]}")

    print("\n== recall breakdowns ==")
    for query, correct, wrong in TESTS:
        print(f"\n### query: {query!r}   (want {correct}, must not return {wrong})")
        pool = hm.state.active
        q_emb = await hm._embedder.embed(query)
        q_tokens = set(__import__("re").findall(r"[a-z0-9]{3,}", query.lower()))
        from hypermem.engine import _is_identity_query, _cosine
        ident = _is_identity_query(query)
        now = __import__("time").time()
        rows = []
        for m in pool:
            b = await hm.explain_recall(query, m.id)
            if b:
                rows.append((b["total"], m, b))
        rows.sort(key=lambda x: x[0], reverse=True)
        for score, m, b in rows[:6]:
            leak = "  <<< LEAK" if wrong in m.content.lower() else ""
            hit = "  <<< HIT" if correct in m.content.lower() else ""
            print(f"  {score:5.2f}  cos {b['cosine']:.2f}  lex {b['lexical']:.2f}  "
                  f"imp {b['importance']:.2f}  rec {b['recency']:.2f}  "
                  f"idb {b['identity_boost']:.1f}  {m.content[:50]}{hit}{leak}")
        # What the full pipeline actually returns (floor, evidence gate,
        # ambiguity tiebreak, budget all applied).
        r = await hm.recall(query)
        has_correct = any(correct in m.content.lower() for m in r.relevant)
        has_wrong = any(wrong in m.content.lower() for m in r.relevant)
        verdict = "PASS" if (has_correct and not has_wrong) else "FAIL"
        print(f"  -> recall returns {len(r.relevant)}: " +
              "; ".join(m.content[:44] for m in r.relevant))
        print(f"  -> {verdict}  (correct={has_correct}, wrong={has_wrong})")
    await hm.close()


if __name__ == "__main__":
    asyncio.run(main())
