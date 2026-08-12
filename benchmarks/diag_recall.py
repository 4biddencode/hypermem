"""Diagnose recall failures with the real LLM (qwen2.5:7b)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hypermem import HyperMEM, HyperMemConfig

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


async def main():
    hm = HyperMEM(HyperMemConfig(auto_tag_threshold=0.3))
    for f in FACTS:
        r = await hm.add_message("user", f)
        if r.tagged:
            print(f"PLANT -> stored: {r.tagged.content!r}  (imp={r.tagged.importance:.2f} kw={r.tagged.keywords})")
        else:
            print(f"PLANT -> NOT STORED: {f!r}")

    print(f"\n{len(hm.state.active)} active memories\n")

    for query, kw in QUERIES:
        recall = await hm.recall(query)
        contents = [m.content for m in recall.relevant]
        hit = any(kw in m.content.lower() for m in recall.relevant)
        print(f"{'HIT ' if hit else 'MISS'} {query!r}  (got {len(contents)} mem)")
        for c in contents:
            print(f"      -> {c!r}")


asyncio.run(main())