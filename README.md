# HyperMEM — AI Memory That Never Forgets

[![CI](https://github.com/4biddencode/hypermem/actions/workflows/ci.yml/badge.svg)](https://github.com/4biddencode/hypermem/actions/workflows/ci.yml)

HyperMEM is a memory layer for AI applications. It watches a conversation,
judges what is worth remembering, stores it, and later injects the relevant
memories back into the context window — so an AI can recall facts from
10,000+ messages ago as reliably as a chat from yesterday.

It works with any LLM (Ollama, OpenAI, Anthropic) and ships with:

- **Python engine** — `add_message` → `recall` → `get_context`
- **REST server** — HTTP API for any language/framework
- **worldIDA** — live "world state" tracking for roleplay/virtual agents
- **Persistence** — full state (memories + world state) as JSON

## Install

Requires Python 3.10+.

```bash
pip install -e .            # core (any LLM via Ollama/OpenAI/Anthropic)
pip install -e ".[server]"  # + REST server (fastapi, uvicorn)
pip install -e ".[test]"    # + test tooling
```

## Quick Start

```python
import asyncio
from hypermem import HyperMEM

hm = HyperMEM()  # defaults: Ollama, qwen2.5:7b on localhost:11434

async def main():
    # HyperMEM auto-tags important details as they arrive
    await hm.add_message("user", "My name is Emanuel, I live in Berlin")
    await hm.add_message("user", "I'm planning a hike in the Alps next week")

    # Later, relevant memories come back — even with different wording
    ctx = await hm.get_context("Where do I live?")
    print(ctx)

asyncio.run(main())
```

Output:

```
[RELEVANT MEMORIES]
- My name is Emanuel, I live in Berlin (importance: 100%)
[/RELEVANT MEMORIES]
```

Put `ctx` into your system prompt and your AI now answers with facts it was
never given in the visible chat history.

## REST Server

Expose the same engine over HTTP:

```bash
hypermem-server --port 8080 --llm-model qwen2.5:7b --llm-endpoint http://localhost:11434
```

```bash
# Create a session
curl -X POST localhost:8080/sessions -d '{"session_id": "rp1"}'

# Feed it messages (returns anything newly tagged/recalled)
curl -X POST localhost:8080/sessions/rp1/messages \
  -d '{"role": "user", "content": "My name is Emanuel"}' \
  -H "Content-Type: application/json"

# Ask for context — memories + recent chat + world state
curl "localhost:8080/sessions/rp1/context?message=What+is+my+name?"

# Inspect everything it remembers
curl localhost:8080/sessions/rp1/memories

# Export the full session state
curl localhost:8080/sessions/rp1/state
```

Endpoints: `GET /health`, `POST/GET/DELETE /sessions` (and `/sessions/{id}`),
`POST /sessions/{id}/messages`, `POST /sessions/{id}/remember`,
`GET /sessions/{id}/recall?query=`, `GET /sessions/{id}/context`,
`GET /sessions/{id}/memories`, `PUT /sessions/{id}/persona`,
`GET /sessions/{id}/world-ida`, `POST /sessions/{id}/world-ida/update`,
`GET /sessions/{id}/state`.

Sessions are auto-persisted to `--data-dir` (default `.hypermem_data/`) after
every change and survive restarts.

## worldIDA — live world state

For roleplay and virtual-agent use, HyperMEM maintains **one compact state
object per session** (scene, character mood, relationship, ongoing action).
Unlike long-term memory it is never searched — it is fully rewritten every
turn and injected in full, so the model always knows *where it is right now*
without a growing context.

```python
await hm.update_world_ida(user_msg, ai_msg)
ctx = await hm.get_context("...")  # now includes [WORLD STATE]
```

When a scene transitions (location/time shift), the old scene is summarized
into long-term memory automatically.

## How It Works

1. **Judge** — when you add a message, the LLM answers: "is there a fact here?"
   Every extracted fact is stored; its importance field ranks it (floored at
   the tagging threshold, so extraction never silently drops facts).
2. **Store** — facts go into `active` memories with keywords + importance.
   Contradictions supersede old static/temporal memories; episodic events
   coexist. Oversized history is archived by decay-adjusted importance.
3. **Recall** — for each message/question, the LLM ranks candidate memories
   (top-3, JSON output). If the model returns nothing, a keyword-overlap
   fallback still surfaces anything that lexically matches.
4. **Inject** — recalled memories + recent chat + worldIDA are assembled into
   the context prompt.

## Configuration

```python
from hypermem import HyperMEM, HyperMemConfig

config = HyperMemConfig(
    llm_provider="ollama",          # "ollama" | "openai" | "anthropic"
    llm_model="qwen2.5:7b",
    llm_endpoint="http://localhost:11434",
    llm_api_key=None,               # required for openai/anthropic
    auto_tag_threshold=0.4,         # minimum importance to rank a memory at
    max_active_memories=100,        # active mems before decay-archiving
    max_context_messages=20,        # recent chat lines kept for context
    auto_tagging=True,              # set False to only store explicit remembers
)
hm = HyperMEM(config)
```

## API

| Method | Description |
|--------|-------------|
| `await add_message(role, content)` | Add a message → auto-tag + recall + conflict resolution |
| `remember(content, memory_type)` | Explicitly store a memory (pinned) |
| `await recall(query)` | Search active + archived memories |
| `await get_context(message)` | Build prompt with world state + memories + chat |
| `await update_world_ida(user, ai)` | Update the live world state object |
| `set_persona(Persona)` | Define persona — protected from memory operations |
| `to_dict() / from_dict()` | Serialize / restore the full engine |
| `save(path) / load(path)` | Atomic JSON persistence (includes world state) |
| `memories()` | List all memories with effective importance |

The engine is an async context manager: `async with HyperMEM() as hm:` closes
the HTTP client on exit.

## Persona Isolation

`set_persona()` defines the character/assistant identity. Persona fields are
excluded from memory extraction prompts and worldIDA rules forbid modifying
persona-level traits — the system remembers what is *said to it*, never
mistakes its own identity for conversation facts.

## Benchmarks

The suite runs against a **real model** — no mock modes:

```bash
python benchmarks/run_benchmarks.py              # full: up to 10K messages
python benchmarks/run_benchmarks.py --quick      # ~minutes: 100/500/1K
python benchmarks/run_benchmarks.py --scales 100,1000,5000 --model gemma3:12b
```

Metrics: recall accuracy vs. distance, recall with distractors, per-call
latency, storage growth (conversation-bound and per-memory), worldIDA drift
over 100 turns. Results land in `benchmarks/benchmark_results.json`.

`benchmarks/diag_recall.py` prints every planted fact vs. what recall
returns — the first tool to reach for when troubleshooting.

## Repo Layout

```
hypermem/            Python package (engine, LLM client, worldIDA, server)
tests/               hermetic test suite (stubbed Ollama API — runs in CI)
benchmarks/          real-model benchmark suite
```

## License

MIT — see [LICENSE](LICENSE).