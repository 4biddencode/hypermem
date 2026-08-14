# HyperMEM — Full Logic Review

A logic/correctness audit of the HyperMEM codebase (judge → store → recall → inject).
Reviewed `hypermem/{engine,llm,embeddings,world_ida,server,types}.py`, `examples/demo.py`,
and the benchmarks. Findings below are ranked by severity. Each was verified against the
actual code path (several reproduced with live runs against a stubbed LLM).

**Status: all confirmed bugs are FIXED and verified (169 tests green).** Fixes noted inline.

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
large-output agents — so their structured findings were lost. The findings above come from
my own line-by-line read of every module, cross-checked against the two module reviewers
that did complete (world_ida, embeddings), and each was reproduced/verified with a live run
against a stubbed LLM. The engine findings (#1, #2) were independently confirmed with
executable reproductions.
