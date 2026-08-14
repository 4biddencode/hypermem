# HyperMEM — Full Logic Review

A logic/correctness audit of the HyperMEM codebase (judge → store → recall → inject).
Reviewed `hypermem/{engine,llm,embeddings,world_ida,server,types}.py`, `examples/demo.py`,
and the benchmarks. Findings below are ranked by severity. Each was verified against the
actual code path (several reproduced with live runs against a stubbed LLM).

**Status: all confirmed bugs are FIXED and verified (191 tests green).** Fixes noted inline.

---

## Round 2 — second independent review (9 confirmed findings, all FIXED)

A second multi-agent workflow (8 module reviewers + adversarial verifiers + completeness
critic) ran after the round-1 fixes. It confirmed 9 new bugs not in round 1 — all in the
load/coercion/merge/fallback paths — and refuted 2 others as not-reachable. All 9 are fixed
and locked in with regression tests.

### R2-1. MEDIUM — Auto-detected OpenAI drops the gateway endpoint (`hypermem/embeddings.py:40`) — ✅ FIXED
`resolve_embedding_provider` returns `"openai"` for an OpenAI-compatible gateway at
`llm_endpoint=http://gw:8000/v1`, but `_embedding_url("openai", endpoint)` used only the
*embedding* endpoint (default `None`), producing `https://api.openai.com/v1/embeddings` — so
text went to OpenAI instead of the local gateway (401 permanent disable without a key).
**Fix (applied):** when `provider="auto"` resolves to openai and no explicit embedding
endpoint is given, the LLM endpoint is inherited as the embed base URL.

### R2-2. MEDIUM — `from_dict` loads uncoerced field values that crash later (`hypermem/types.py:135`) — ✅ FIXED
`_build_from_fields` passed raw values through. Stale/hand-edited JSON with `content: null`
loaded `content=None`, then `_tokens_of(None).lower()` crashed every recall; string
`access_count`/`importance` crashed `+=1` and `max()` in `add_message`/`_apply_decay`.
**Fix (applied):** `_mem` coerces text fields to str, numeric fields to float/int, and
`keywords` to a list, treating `None` as "use the default".

### R2-3. LOW — 429 rate limit treated as permanent (`hypermem/embeddings.py:180`) — ✅ FIXED
`_handle_error` routed all 4xx (incl. 429) to `_fail`, latching `_available=False` forever —
a transient rate limit killed semantic recall for the process lifetime. **Fix (applied):**
429 is now transient (backoff + retry), only config errors stay permanent.

### R2-4. LOW — Success path marks client available before validating the vector (`hypermem/embeddings.py:155`) — ✅ FIXED
A 200 with a null/missing embedding set `_available=True` before the `isinstance` guard, so
the engine's available gate kept calling `embed` every recall with no backoff. **Fix
(applied):** only a valid list marks the client available.

### R2-5. LOW — worldIDA merge never resets/increments `turn_count_in_scene` (`hypermem/world_ida.py:267`) — ✅ FIXED
When the LLM omitted `turn_count_in_scene` (a natural "only what changed" response), the
merge carried the stale count forward — a new scene started at 8 and only grew. **Fix
(applied):** the merge now maintains rule 3 — reset to 0 on `scene_changed`, otherwise
increment the prior count — while still respecting an explicitly-provided count.

### R2-6. LOW — `_as_int("6.0")` raises and resets a count to 0 (`hypermem/world_ida.py:149`) — ✅ FIXED
`int("6.0")` raises ValueError → default 0, silently losing a turn count a small model
emitted as a float string. **Fix (applied):** `_as_int` parses via `int(float(value))`.

### R2-7. LOW — `_ida_from_dict` crashes on a present-but-None section (`hypermem/world_ida.py:115`) — ✅ FIXED
`data.get("meta", {}).get(k)` raised AttributeError when a corrupt/foreign save contained
`"meta": null` (the load path bypasses validation). **Fix (applied):** each section is
isinstance-checked before `.get`.

### R2-8. LOW — Explicit `null` confidence resets it to max (`hypermem/world_ida.py:154`) — ✅ FIXED
`_as_float(None)` → 1.0, so `{"confidence": null}` clobbered a previously-lowered confidence
to max while an omitted one carried over — asymmetric. **Fix (applied):** the merge drops
explicit nulls so they read as "unchanged" and the prior value carries forward.

### R2-9. LOW — `total_messages` not coerced to int (`hypermem/types.py:183`) — ✅ FIXED
A string `total_messages` in stale JSON made `+= 1` and the consolidation throttle
subtraction raise TypeError. **Fix (applied):** coerced to int at load.

### Refuted (not real, no change)
- `world_ida.py:193` "numeric/bool scalars in text fields crash the context join" — `_as_str`
  already coerces all text fields; no crash path.
- `types.py:151` "stale memories share one module-level keywords list via setdefault" — real
  shared object, but no code path mutates `keywords` in place; no reachable wrong behavior.

---

## Round 3 — third review (11 confirmed findings, all FIXED)

A third multi-agent workflow ran per-function reviewers with tiny schema-constrained output
(small outputs defeat the model's 128k output-token cap that killed the engine/llm/server
reviewers in rounds 1-2). It confirmed 11 new bugs — the first deep review of the never-reviewed
`engine.py` persistence/ingestion paths and `llm.py`. All 11 are fixed and locked in with
regression tests (191 total, up from 180). 14 of 17 reviewers and the completeness critic again
died to the output cap, so the findings come from the 3 reviewers that completed plus adversarial
verifiers, cross-checked against my own line-by-line read of the same modules.

### R3-1. MEDIUM — `from_dict` leaves stale world state when the loaded dict omits it (`hypermem/engine.py:967`) — ✅ FIXED
`from_dict` restored `_world_ida`/`_world_ida_store` only under `if "world_ida" in data` with no
else-reset. Loading a save without world_ida keys (a session that never had one) kept the
*previous* session's scene injected into context and used as the base for the next update.
Reproduced: engine kept `'tavern'` after loading a keyless dict. **Fix (applied):** both are
reset to `None` before the conditional restore.

### R3-2. MEDIUM — `update_world_ida` re-remembers a scene transition on every failed turn (`hypermem/engine.py:656`) — ✅ FIXED
On a failed update (timeout/429/parse/validate) `world_ida.update_world_ida` returns
`previous_ida` unchanged but with `scene_changed=True`, so the engine stored a duplicate
"Scene ended" pinned memory every failed turn until a success reset the flag. **Fix (applied):**
the scene summary is recorded only when the update actually produced a *new* object
(`new_ida is not old_ida`) with `scene_changed` — a failed update is an identity match and is
skipped.

### R3-3. MEDIUM — Scene-transition summaries stored pinned, importance 1.0, never pruned (`hypermem/engine.py:674`) — ✅ FIXED
Auto scene summaries went through `remember()`, which pins (STATIC, importance 1.0) — pinned
survives the archive cap and STATIC never decays, so N scene changes created N permanent
full-score stale memories. **Fix (applied):** `remember()` gained `pinned`/`source` params; the
scene summary is stored `pinned=False, source="auto"` so decay/archive/consolidation can prune
it. The public `remember()` default stays pinned.

### R3-4. MEDIUM — Identity boost applies to all `name`-tagged memories, letting a lookalike outrank the user (`hypermem/llm.py:508`) — ✅ FIXED
`_keyword_fallback` added +5 to any memory with the "name" keyword, so a higher-importance
lookalike ("My dog's name is Rex", imp 0.9) outranked the user identity ("My name is Emanuel",
imp 0.6) for "What's my name?". **Fix (applied):** the boost now applies only to memories whose
content is a first-person identity statement (`_is_identity_statement`, mirrored from the
engine); lookalikes carrying the "name" keyword are sunk below it — consistent with the engine's
`_score_memory` identity-query lookalike penalty.

### R3-5. LOW — Identity boost never fires for "Who am I?" (zero overlap) (`hypermem/llm.py:504`) — ✅ FIXED
The boost was inside `if overlap > 0:`, so a "Who am I?" query (tokens `{'who'}`) shared no token
with the stored name and the identity memory was missed entirely. **Fix (applied):** the identity
branch now runs regardless of overlap, giving even a zero-overlap identity memory a chance.

### R3-6. LOW — `_is_filler` misses common filler variants (`hypermem/engine.py:392`) — ✅ FIXED
The exact-match set mixed dotted/bare forms ("Yes." filtered but "Yes"/"Ok"/"Sure"/"Hi."/lowercase
missed), so a filler could fire a judge call and get stored. **Fix (applied):** case and trailing
punctuation are normalized to a canonical token set, expanded with common variants.

### R3-7. LOW — Embedder auto-resolution reads config, not the injected LLM (`hypermem/engine.py:344`) — ✅ FIXED
`__init__` passed `config.llm_provider/endpoint` to the EmbeddingClient for `auto`, ignoring the
injected `self._llm`'s actual provider — an injected OpenAI-compatible LLM (config still defaulting
to ollama) silently misrouted embeddings. **Fix (applied):** auto-resolution now reads
`self._llm.provider/endpoint`.

### R3-8. LOW — Consolidation fallback truncates event text, tail lost (`hypermem/engine.py:617`) — ✅ FIXED
The fallback joined events and cut at `max_memory_chars`, then archived the originals with
`superseded_by` — the truncated tail was unrecoverable even with `search_archive`. **Fix
(applied):** a failed summary now skips the round and keeps the originals intact and searchable.

### R3-9. LOW — `from_dict` worldIDA restore lacks an isinstance guard (`hypermem/engine.py:975`) — ✅ FIXED
A corrupt list value for `world_ida` crashed the whole load (`_ida_from_dict` calls `.get` on a
list). **Fix (applied):** the restore is guarded with `isinstance(dict)`.

### R3-10. LOW — `_last_consolidation` restored via bare `int()` crashes on float-string (`hypermem/engine.py:973`) — ✅ FIXED
`int("50.5")` raised ValueError, aborting load, unlike the tolerant `_as_int` used elsewhere.
**Fix (applied):** coerced via `_as_int`.

### R3-11. LOW — `save()` never fsyncs before the atomic rename (`hypermem/engine.py:992`) — ✅ FIXED
A power loss between `os.replace` and the page-cache flush could leave the target empty with the
old file gone, defeating the atomic-write durability claim. **Fix (applied):** `f.flush()` +
`os.fsync(f.fileno())` before `os.replace`.

### Refuted (not real, no change)
- `llm.py:422` `judge()` parses with strict `json.loads` — dead/backcompat code with no callers;
  production uses `extract_json_object` with a generous verbatim fallback, so no fact is lost.

---

## Confirmed logic bugs (fix first)

### 1. HIGH — Access-count decay is inverted (`hypermem/engine.py:235`) — ✅ FIXED
`_apply_decay`:
```python
access_decay = 1.0 / (1.0 + 0.1 * mem.access_count)
```
This is monotonically **decreasing** in `access_count`. The docstring and the test
`test_high_access_count_still_remembered` ("Frequently accessed memories should decay
slower") state the opposite intent. Verified numerically:
- `access_count=100` → decay score **0.081**
- `access_count=1`   → decay score **0.81**

Because `add_message` (line 389) and `recall` (line 851) increment `access_count` every
time a memory is recalled, **the more useful a memory proves to be, the faster it decays**
toward archival. When the `max_active_memories` cap is hit, the archive sort
(`scored.sort(key=(not pinned, score))`, line 482) picks the lowest scores — i.e. the
most-frequently-recalled memories — and archives them, making them unreachable via
`_recall_candidates` (which excludes archived by default). A memory can then be archived
and re-created as a duplicate.

**Fix (applied):** access count now slows decay — `effective_hours = hours_since_access /
(1.0 + 0.1 * access_count)` — so frequent access keeps a memory live, never raising the
score above stored importance. Verified: access=100 → 0.899, access=1 → 0.892.

### 2. MEDIUM — Supersession fails when a correction changes the *value* (`hypermem/engine.py:114-117`) — ✅ FIXED
`_resolve_conflict` requires the new fact's keywords to overlap the old memory's content:
```python
shared = [kw for kw in new_keywords
          if kw and len(kw) >= 4 and kw in existing.content.lower()]
if not shared:
    return False, None
```
When a user corrects a fact by changing its value — "I live in Vienna" → "Actually, I
moved to Berlin" — the new keywords ("moved", "berlin") do **not** appear in the old
content ("I live in Vienna"), so `shared` is empty and no supersession happens. Reproduced
with a live stub: both memories coexist, and the stale "I live in Vienna" stays active and
recallable. The demo's vault-password case works only because "vault"/"password" are
*unchanged* keywords present in both.

This is the exact scenario the "new fact supersedes old" contract is meant to handle, and
it silently fails. **Fix (applied):** `_resolve_conflict` and `_find_conflicts` now also
match on a shared non-empty `subject` (lowercased) as an alternative to keyword overlap,
so a value-change correction supersedes. The correction cue is still required, and a
correction about a *different* subject still does not supersede. 3 new regression tests.

### 3. MEDIUM — worldIDA `scene_changed` is never coerced to bool (`hypermem/world_ida.py:217`) — ✅ FIXED
`_ida_from_dict` passes the LLM's raw value through: `Meta(scene_changed="false")`. A
model emitting the string `"false"` (plausible for small models) yields a **truthy** value.
Verified: `bool("false") is True`. Then `engine.py:608`
`if new_ida.meta.scene_changed:` fires **every turn**, storing a bogus
`scene_transition_summary` memory and persisting `"false"` as a string in saved state.

**Fix (applied):** `_ida_from_dict` coerces `scene_changed` via `_as_bool` (accepts bools
and the string forms "true"/"false"/"1"/"0"), and coerces the numeric meta fields too.

### 4. MEDIUM — worldIDA `_validate_ida` accepts non-scalar field values → crash (`hypermem/world_ida.py:135`) — ✅ FIXED
`_validate_ida` checks only that each top-level section is a dict; it does not validate the
field *values*. A malformed LLM response `{"scene":{"location":{"coords":[1,2]}}}` passes,
and `Scene.location` becomes a dict/list. Then `world_ida_to_context_string` (line 258)
does `', '.join(scene_parts)` with a non-str element → **`TypeError`**. Verified. Neither
`get_context` (engine.py:862) nor the server endpoint (server.py:251) catches it → 500 /
crashed context injection.

**Fix (applied):** `_validate_ida` rejects any section field whose value is a
dict/list/set/tuple.

### 5. MEDIUM — Server `create` has a TOCTOU race on session id (`hypermem/server.py:86-95`) — ✅ FIXED
```python
if session_id and session_id in self._sessions:   # NOT under self._lock
    raise HTTPException(409, ...)
...
self._write(sid, hm)                              # NOT under self._lock
with self._lock:
    self._sessions[sid] = hm
```
The existence check and the file write happen **outside** `self._lock`. Two concurrent
`create("foo")` calls (threadpool workers / multiple uvicorn workers) can both pass the
409 check, both write `foo.json`, and the second silently replaces the first in memory.
The 409 contract is violated and a session can be lost.

**Fix (applied):** `self._lock` now held across the check + write + insert.

---

## Edge cases / design issues

### 6. LOW — Temporal memories never decay (`hypermem/engine.py:239`, `add_message:442-456`) — ✅ FIXED
`_apply_decay` returns `mem.importance` unchanged for `TEMPORAL` (line 239), yet temporal
memories ARE stored in `active` by `add_message`. Transient state (mood, current location)
thus accumulates as permanent, never-decaying clutter — contradicting the docstring
"Temporal memories are handled by worldIDA." They're only removed by the
`max_active_memories` cap. **Fix (applied):** temporal memories now decay with time (no
access-count boost — transient state shouldn't be kept alive by recall), so they age out of
`active` via the archive cap.

### 7. LOW — worldIDA merge leaves stale `meta` (`hypermem/world_ida.py:213-220`) — ✅ FIXED
The partial-output merge carries forward whatever `meta` the previous state had. If a
scene-change turn sets `scene_changed=true` and the next turn's LLM output omits `meta`,
`scene_changed` stays true → a duplicate scene-transition memory is recorded. The prompt's
own rule 3 (increment/reset `turn_count_in_scene`) is never enforced by the merge.
**Fix (applied):** `scene_changed` is reset to `False` on merge unless this turn explicitly
set it.

### 8. LOW — `cosine()` clamps negative similarity to 0 (`hypermem/embeddings.py:200-201`) — ⏸️ INTENTIONAL
`if dot <= 0: return 0.0`. An anti-correlated vector (true cosine −0.6) scores `sim=0`
instead of −0.6, so the `2.0*cosine` term never contributes negative signal. Such a memory
can pass the `0.5*floor` filter and rank above genuinely-matching memories. Verified:
`cosine([1,0],[-1,0])` returns `0.0`. Deliberate: clamping avoids surprising negative
contributions and is consistent with `_has_evidence` using `>= 0.5` as the floor. Left as-is.

### 9. LOW — `diff_versions` guard off-by-one (`hypermem/world_ida.py:354`) — ✅ FIXED
`if len(history) < max(abs(version_a), abs(version_b))` allows a positive index equal to
`len(history)`, then `history[index]` raises `IndexError` instead of returning `None`.
**Fix (applied):** each index validated against the actual range `[-n, n)` before indexing.

### 10. LOW — Recency vs decay use different time bases (`hypermem/engine.py:774` vs `_apply_decay`) — ✅ FIXED
The score's `recency = 1 − (now − created_at)/(30·86400)` uses `created_at`, while decay
uses `last_accessed_at`. A memory recalled recently but created long ago gets low recency
even though it's active — inconsistent treatment of "recent". **Fix (applied):** recency
now uses `last_accessed_at`, consistent with decay.

### 11. LOW — OpenAI embed path indexes `data[0]` blindly (`hypermem/embeddings.py:162`) — ✅ FIXED
`data.get("data",[{}])[0].get("embedding")` raises `IndexError` on an empty `data` array,
which the broad `except` reclassifies as a transient network failure → the client parks 30s
and retries the same deterministic malformed response forever. **Fix (applied):** guarded
against an empty `data` array.

### 12. LOW — Embedding auto-detect misroutes OpenAI-compatible endpoints (`hypermem/embeddings.py:61`) — ✅ FIXED
`resolve_embedding_provider` maps any non-"openai" `http` endpoint to the `ollama` provider.
An OpenAI-compatible gateway at `http://gw:8000/v1` (no "openai" in the URL) gets the
Ollama request schema sent to `http://localhost:11434/api/embeddings` → permanent disable.
**Fix (applied):** a `/v1` endpoint path (the OpenAI-compatible convention) now routes to
`openai`, consistent with the LLM provider inference.

---

## What's correct (verified, no action needed)

- **Verbatim storage + judge-classify separation** — sound; the judge never rewrites memory
  text, exact tokens survive for recall.
- **Identity keyword tagging** (`_is_identity_statement` / `_is_identity_query`) — the
  first-person-only regexes correctly avoid tagging another character's name; the two copies
  (engine + llm `_keyword_fallback`) are identical.
- **Stale-leak exclusion** — `_recall_candidates` excludes superseded/consolidated originals
  by default; `search_archive` is a correct opt-in.
- **Robust JSON extraction** (`extract_json_object` / `_balanced_blocks`) — handles fences,
  single quotes, trailing commas; index access in `_rank_memories` is guarded against
  out-of-range.
- **Atomic save** (`engine.py:923-940`) — temp file + `os.replace`, cleans up on error.
- **Per-session asyncio lock** around mutate+persist in the server is correct for the
  single-event-loop case (the create-race is the exception, see #5).

---

## Process note

The multi-agent workflow (11 parallel reviewers + adversarial verification) was run twice.
The lens reviewers (concurrency/persistence/lifecycle/score-math) and most module reviewers
repeatedly hit the model's 128k output-token cap — a systemic issue with many
large-output agents — so their structured findings were lost. The round-1 findings come from
my own line-by-line read of every module, cross-checked against the two module reviewers
that did complete (world_ida, embeddings), and each was reproduced/verified with a live run
against a stubbed LLM. The engine findings (#1, #2) were independently confirmed with
executable reproductions.

**Round 2** used a tighter workflow (max 8 short findings per reviewer, adversarial verify,
completeness critic). The embeddings/world_ida/types reviewers completed and produced the 9
confirmed findings above; the engine/llm/server reviewers and the critic again hit the output
cap. The round-2 fixes were cross-checked against my own independent read of the same
modules. Tests grew from 169 to 180 with regression coverage for every round-2 fix.
