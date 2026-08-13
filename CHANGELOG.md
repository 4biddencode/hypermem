# Changelog

All notable changes to HyperMEM are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-08-13

### Full rework: a memory layer that actually works

The 0.x MVP stored judge-written summaries, never superseded changed facts,
recalled by ranking the whole list with an LLM, and failed completely on
some models. This release rebuilds the pipeline around honest memory
behavior. **The public API is unchanged** — `add_message`, `recall`,
`get_context`, `remember`, `set_persona`, `update_world_ida`, `save`/`load`,
`memories`, and every REST endpoint keep their shape. Internals are new.

**Ingestion**
- The judge now *classifies* (`has_fact`, `importance`, `memory_type`,
  `subject`, `keywords`) instead of writing memory text. The user's message
  is stored **verbatim** (capped at `max_memory_chars`), eliminating the
  paraphrase drift that destroyed substring recall.
- Robust JSON extraction (`extract_json_object`) fixed models like
  `gemma3:12b` that scored 0.0 on every suite because the judge's response
  was never parsed.
- Near-duplicate detection (embedding similarity + normalized text) refreshes
  a memory instead of storing a second copy. First-person identity
  statements ("My name is…") are deterministically tagged.

**Lifecycle**
- Type-aware **supersession**: a newer static/temporal fact on the same
  subject supersedes an old one when a correction cue ("actually", "changed",
  "instead") is present; episodic events coexist. Superseded memories are
  excluded from recall — the stale-leak fix.
- **Consolidation** (MemGPT-style): once a subject accumulates enough
  episodic events, the oldest are fused into one static knowledge memory and
  the originals archived (throttled by `consolidation_interval`).
- Decay-archived memories are excluded from recall unless `search_archive` is
  enabled.

**Recall**
- New optional **semantic recall** via `EmbeddingClient` (Ollama
  `nomic-embed-text` or OpenAI `text-embedding-3-small`), auto-detected from
  the LLM provider, with graceful fallback to the proven LLM+lexical path
  when embeddings are unavailable.
- **Hybrid scoring**: `2.0·cosine + 1.2·lexical + 0.5·importance +
  0.3·recency`, plus a deterministic identity boost for name queries. A
  relative relevance floor drops off-topic noise; a token budget caps the
  injected context.
- Embedding-first recall skips the per-recall LLM call by default
  (`recall_use_llm=False`) — recall latency drops to tens of milliseconds.
- **Distractor resistance**: near-duplicate facts that share keywords with a
  true answer no longer leak into recall. A deterministic echo-marker demotion
  ("also", "another", "the other") keeps the primary fact on top; an
  ambiguity-triggered LLM tiebreak adjudicates near-tied candidates and
  excludes the decoys it rejects; an evidence gate (shared token or cosine
  ≥ 0.5) drops no-evidence lookalikes. Plus a tightened identity detector so
  "What's my bow called?" no longer triggers the name boost.

**worldIDA**
- Partial-output updates: the model returns only changed fields, merged over
  the previous state — fewer tokens, lower latency. Scene transitions still
  emit a long-term-memory summary.

**Server**
- Per-session `asyncio.Lock` serializes the mutate+persist sequence (fixes a
  read-modify-write race).
- Auto-generated sessions persist on create, not just on first mutation.
- New `GET /sessions/{id}/memories/{id}` **provenance** endpoint: why a
  memory exists (source message + classification) and, with `?query=`, why it
  would surface (live score breakdown).
- API responses no longer leak raw embedding vectors.

## [0.1.0] — 2026-07

- Initial MVP: judge → store → recall → inject pipeline, REST server,
  worldIDA, JSON persistence.
