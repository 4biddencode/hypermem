"""HyperMEM 10K message stress test."""
import asyncio
import time
import pytest
from hypermem import HyperMEM, HyperMemConfig

pytestmark = pytest.mark.live

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


async def run_test(batch_size: int = 1000, num_batches: int = 10):
    """Run stress test with increasing message counts."""
    config = HyperMemConfig(
        llm_provider="ollama",
        llm_model="qwen2.5:7b",
        llm_endpoint="http://localhost:11434",
        auto_tag_threshold=0.3,
        max_active_memories=100,
    )
    hm = HyperMEM(config)

    print(f"\n{'=' * 55}")
    print(f"HyperMEM Python Stress Test")
    print(f"{'=' * 55}")
    print(f"\n{'Scale':<10} {'Passed':<10} {'Time':<10} {'Active':<10} {'Archived':<10}")
    print(f"{'-' * 55}")

    # Plant facts
    for fact in FACTS:
        await hm.add_message("user", fact)

    # Test in batches
    for batch in range(1, num_batches + 1):
        start = time.time()

        # Add filler messages
        for i in range(batch_size):
            await hm.add_message("user", FILLER[i % len(FILLER)],
                                 message_id=f"f_{batch}_{i}")

        # Test recall
        passed = 0
        for query, keyword in QUERIES:
            recall = await hm.recall(query)
            if any(keyword in m.content.lower() for m in recall.relevant):
                passed += 1

        total = len(FACTS) + batch * batch_size
        elapsed = f"{time.time() - start:.1f}s"
        print(f"{total:<10} {passed}/12      {elapsed:<10} {len(hm.state.active):<10} {len(hm.state.archive):<10}")

        if passed < 9:
            print(f"\nFAILED at {total} messages - memory degradation detected!")
            return

    print(f"\nAll tests passed! HyperMEM maintains memory across {len(FACTS) + num_batches * batch_size} messages.")


def main():
    asyncio.run(run_test())


if __name__ == "__main__":
    main()
