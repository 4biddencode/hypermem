"""HyperMEM - Core engine."""
import time
import re
import logging
from typing import Optional
from .types import (
    Message, HyperMem, HyperMemState, HyperMemConfig, MemoryType, Persona,
    RecallResult, AddMessageResult, state_to_dict, state_from_dict,
)
from .llm import LLMClient, extract_json_object
from .embeddings import cosine as _cosine

logger = logging.getLogger("hypermem")

_mem_id_counter = 0


def _next_id() -> str:
    global _mem_id_counter
    _mem_id_counter += 1
    return f"hm_{int(time.time() * 1000)}_{_mem_id_counter}"


def _extract_keywords(text: str) -> list[str]:
    """Extract keywords from text for fallback matching."""
    if not text:
        return []
    words = re.findall(r"[a-z0-9]{4,}", text.lower())
    stopwords = {
        "the", "this", "that", "with", "from", "have", "been", "were",
        "what", "when", "where", "which", "their", "there", "about",
        "would", "could", "should", "after", "before", "into", "over",
        "only", "other", "than", "then", "very", "just", "also", "more",
        "some", "such", "like",
    }
    return list(dict.fromkeys(w for w in words if w not in stopwords))


# ---- Contradiction resolution ----

# Phrases that mark a message as a *correction* of an earlier fact, as opposed
# to a new fact that happens to share a topic. Only these allow a memory to be
# superseded — so "the crown controls the weather" never swallows "we seek the
# crown" just because both mention the crown.
_CORRECTION_CUES = (
    "actually", "no wait", "wait,", "changed", "instead", "correction",
    "scratch that", "not anymore", "no longer", "on second thought",
)


def _has_correction_cue(text: str) -> bool:
    low = text.lower()
    return any(c in low for c in _CORRECTION_CUES)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _resolve_conflict(existing: HyperMem, new_content: str,
                      new_importance: float,
                      new_type: MemoryType = MemoryType.EPISODIC,
                      subject: str = "",
                      new_keywords: Optional[list] = None) -> tuple[bool, Optional[HyperMem]]:
    """Decide whether a new fact supersedes an existing memory.

    Supersession is deliberately conservative — only a *correction* of an
    ongoing attribute replaces the old fact. New episodic events always
    coexist, and a shared keyword alone never drops a memory (that was the
    "silently lost facts" bug). A correction only wins when the message
    carries an explicit correction cue ("actually", "changed", "instead", …)
    AND the new fact's keywords genuinely overlap the old memory.

    Args:
        existing: The currently stored memory.
        new_content: The new information (verbatim user message).
        new_importance: Importance of the new information.
        new_type: memory_type the judge assigned to the new fact.
        subject: entity the new fact is about (judge-assigned).
        new_keywords: keywords the judge assigned to the new fact.

    Returns:
        (should_replace, replacement_memory):
        - (True, new_mem) → existing should be superseded
        - (False, None) → append both (they coexist)
    """
    new_keywords = new_keywords or []
    # Only an attribute change (static/temporal) can supersede; an old
    # episodic event is never retroactively deleted by a later fact.
    if new_type not in (MemoryType.STATIC, MemoryType.TEMPORAL):
        return False, None
    if existing.memory_type not in (MemoryType.STATIC, MemoryType.TEMPORAL):
        return False, None
    if not _has_correction_cue(new_content):
        return False, None

    # The new fact must actually touch the old memory, not just be a
    # correction about some other topic that happens to share a stopword.
    shared = [kw for kw in new_keywords
              if kw and len(kw) >= 4 and kw in existing.content.lower()]
    if not shared:
        return False, None

    return True, HyperMem(
        id=_next_id(),
        content=new_content,
        created_at=time.time(),
        last_accessed_at=time.time(),
        access_count=0,
        keywords=new_keywords,
        importance=new_importance,
        source="auto",
        memory_type=new_type,
        superseded_by=None,
        pinned=False,
        subject=subject,
    )


def _find_conflicts(memories: list[HyperMem], new_keywords: list[str],
                    new_content: str) -> list[int]:
    """Find memories that might conflict with new information.

    Looks for memories sharing keywords with the new content.
    """
    conflicts = []
    new_lower = new_content.lower()
    for i, mem in enumerate(memories):
        # Check keyword overlap
        if any(kw in new_lower for kw in mem.keywords):
            # Check if they're about the same topic (name, location, etc.)
            shared = set(kw for kw in mem.keywords if kw in new_lower)
            if len(shared) >= 1:
                conflicts.append(i)
    return conflicts


# ---- Persona isolation ----

def _build_judge_prompt(content: str, persona: Optional[Persona] = None) -> str:
    """Build the judge prompt with persona isolation baked in.

    The judge no longer writes the memory text — rewriting facts destroyed
    the exact tokens that recall needs (the core recall bug). It now only
    *classifies* the message; the engine stores the message verbatim.
    """
    persona_block = ""
    if persona:
        persona_block = f"""
Persona context (DO NOT store these as memories — they are already known):
- Name: {persona.name}
- Description: {persona.description}
- Traits: {', '.join(persona.traits)}
- Backstory: {persona.backstory}
"""

    return f"""Message: "{content}"{persona_block}

Decide whether this message contains a fact worth remembering, and classify it.

Return JSON ONLY:
{{"has_fact": bool, "importance": number, "memory_type": "static"|"episodic"|"temporal", "subject": string, "keywords": [string]}}

Rules:
- has_fact: true if the message reveals identity, relationships, preferences,
  plans, goals, world lore, cause-and-effect, or a notable event. false ONLY
  for pure greetings, filler, or questions that carry no new information.
  Be generous — when in doubt, has_fact=true.
- importance 0.8-1.0: critical to identity, plot, or world logic.
  0.5-0.8: meaningful character, relationship, or world detail.
  0-0.4: minor detail.
- memory_type: "static" for an ongoing attribute that could change (name,
  preference, password, plan, location); "episodic" for a one-time event at
  a specific moment; "temporal" for transient state (mood, current location).
- subject: the short entity this fact is about (e.g. "user", "lyra",
  "vault password", "the crown"). Use "user" for facts about the speaker.
- keywords: lowercase words or short phrases that describe the fact.
  If the fact is about the USER's own name or identity, include the exact
  keyword "name". (Do NOT add "name" for another character's name.)
- NEVER store persona information (it is already known).

Example: {{"has_fact":true,"importance":0.8,"memory_type":"static","subject":"vault password","keywords":["vault","password"]}}

ENGLISH ONLY."""


# ---- Consolidation (episodic → semantic, MemGPT-style) ----

def _build_consolidation_prompt(subject: str, memories: list[HyperMem]) -> str:
    """Prompt the LLM to fuse several episodic events about one subject into
    a single durable knowledge statement."""
    mems = "\n".join(f"{i + 1}. {m.content}" for i, m in enumerate(memories))
    return f"""Summarize these episodic memories about "{subject}" into ONE concise, timeless knowledge statement.

Memories:
{mems}

Rules:
- The summary must read like a durable fact about {subject} — what they do, where they are, what they value — not a list of events.
- Keep it under 200 words.
- Return JSON ONLY: {{"summary": "..."}}

ENGLISH ONLY."""


# ---- Decay ----

def _apply_decay(mem: HyperMem) -> float:
    """Calculate the effective importance after decay.

    Static memories never decay.
    Episodic memories decay with time and access count.
    Temporal memories are handled by worldIDA.
    """
    if mem.memory_type == MemoryType.STATIC:
        return mem.importance

    if mem.memory_type == MemoryType.EPISODIC:
        hours_since_access = (time.time() - mem.last_accessed_at) / 3600
        access_decay = 1.0 / (1.0 + 0.1 * mem.access_count)
        time_decay = max(0.5, 1.0 - 0.01 * hours_since_access)
        return mem.importance * access_decay * time_decay

    return mem.importance


# Echo markers: a memory that frames its subject as secondary or additional
# ("another crown", "the crown can *also* grant invisibility", "the other
# key") describes a non-primary instance. When a primary fact about the same
# thing matches the query better, the echo shouldn't crowd it out of context;
# it stays recallable only when it is itself the best match.
_ECHO_MARKER = re.compile(
    r"\b(also|another|the other|separate|distinct)\b", re.I
)


def _is_identity_statement(text: str) -> bool:
    """True if a message reveals the USER's own identity.

    First-person patterns only ("my name is X", "I am X", "I'm called X"),
    so another character's name ("The king was called Eldrin", "Shadow
    King's true name is Malachar") never gets the identity tag. A possessive
    lookalike ("My name is Eldrin's cousin") is not naming yourself — it
    names someone who belongs to you — so it never gets the tag either.
    """
    if re.search(r"\bmy name is\s+\w+'s\b", text, re.I):
        return False
    return bool(re.search(
        r"\bmy name\b|\bi (?:am|'m|was)\b|\bi'?m (?:called|known as)\b",
        text, re.I,
    ))


def _is_identity_query(text: str) -> bool:
    """True when the query asks about the USER's own identity.

    The user must be the *subject* of the name/called question, not just a
    word in it: "What's my name?" and "Who am I?" qualify; "What's my bow
    called?", "my sister's name", and "the king's name" do not — they ask
    about a thing or another person, never about the user.
    """
    return bool(re.search(
        r"\bmy name\b|\bwho am i\b|\bwhat (?:am i|'m i|are we) called\b"
        r"|\bwhat('s| is) my name\b",
        text, re.I,
    ))


# ---- Main engine ----

class HyperMEM:
    """HyperMEM - AI memory system that never forgets what matters."""

    def __init__(self, config: Optional[HyperMemConfig] = None,
                 llm: Optional[LLMClient] = None):
        self.config = config or HyperMemConfig()
        self.state = HyperMemState(conversation_id=f"hm_{int(time.time())}")
        self._llm = llm or LLMClient(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            endpoint=self.config.llm_endpoint,
            api_key=self.config.llm_api_key,
        )
        from .embeddings import EmbeddingClient
        # Reuse the LLM's injected transport so tests can serve embeddings
        # through the same stub (production passes transport=None → real HTTP).
        self._embedder = EmbeddingClient(
            provider=self.config.embedding_provider,
            model=self.config.embedding_model,
            endpoint=self.config.embedding_endpoint,
            api_key=self.config.embedding_api_key,
            llm_provider=self.config.llm_provider,
            llm_endpoint=self.config.llm_endpoint,
            transport=getattr(self._llm, "_transport", None),
        )
        self._last_consolidation = 0  # message-count throttle for consolidation
        self._world_ida = None
        self._world_ida_store = None

    async def __aenter__(self) -> "HyperMEM":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self):
        """Release the underlying HTTP clients."""
        close = getattr(self._llm, "close", None)
        if close is not None:
            await close()
        close = getattr(self._embedder, "close", None)
        if close is not None:
            await close()

    def _get_ida_store(self):
        if self._world_ida_store is None:
            from .world_ida import WorldIDAStore
            self._world_ida_store = WorldIDAStore()
        return self._world_ida_store

    def set_persona(self, persona: Persona):
        """Set the persona definition. Protected from memory operations."""
        self.state.persona = persona

    def set_world_ida(self, ida) -> None:
        self._world_ida = ida
        store = self._get_ida_store()
        store.set(self.state.conversation_id, ida)

    def get_world_ida(self):
        return self._world_ida

    # ---- Public API ----

    def _is_filler(self, content: str) -> bool:
        return content.strip() in {
            "Ok.", "Hmm.", "I see.", "Yes.", "No.", "Maybe.", "Sure.",
            "Wait.", "Oh.", "Right.", "Let's go.", "Alright.", "Cool.",
            "Nice.", "Good.", "Hey", "Hi", "Hello", "Bye", "Thanks",
        }

    async def add_message(self, role: str, content: str,
                          message_id: Optional[str] = None,
                          memory_type: MemoryType = MemoryType.EPISODIC) -> AddMessageResult:
        """Add a message. Auto-tags, recalls, resolves conflicts."""
        msg = Message(
            id=message_id or _next_id(),
            role=role, content=content, timestamp=time.time(),
        )

        self.state.recent_messages.append(msg)
        if len(self.state.recent_messages) > self.config.max_context_messages:
            if self.config.max_context_messages > 0:
                self.state.recent_messages = self.state.recent_messages[-self.config.max_context_messages:]
            else:
                self.state.recent_messages = []
        self.state.total_messages += 1

        if self._is_filler(content):
            return AddMessageResult(self.state, None, None)

        tagged = None
        recalled = None

        # 1. Recall check (hybrid-ranked; surfaced alongside this message)
        if self.state.active:
            ranked = await self._rank_memories(content)
            if ranked:
                relevant_mems = ranked[:3]
                for mem in relevant_mems:
                    for active_mem in self.state.active:
                        if active_mem.id == mem.id:
                            active_mem.last_accessed_at = time.time()
                            active_mem.access_count += 1
                            break
                recalled = RecallResult(relevant_mems, f"Recalled {len(relevant_mems)} memories")

        # 2. Classification + verbatim storage (user messages only by default)
        if self.config.auto_tagging and role in self.config.auto_tag_roles:
            prompt = _build_judge_prompt(content, self.state.persona)
            result = await self._llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=120,
            )

            if result:
                parsed = extract_json_object(result) or {}
                # has_fact gates storage; an unparseable judge response is
                # treated as "generous" (store verbatim) rather than silently
                # dropping the user's statement.
                if parsed.get("has_fact", True):
                    importance = self._to_float(parsed.get("importance"), 0.0)
                    mtype = self._to_memory_type(parsed.get("memory_type"), memory_type)
                    keywords = [str(k).strip().lower() for k in parsed.get("keywords", [])
                                if str(k).strip()]
                    subject = str(parsed.get("subject", "")).strip()

                    # Deterministic identity tag: "My name is X", "I am X",
                    # "I'm called X" in the raw message → the memory holds the
                    # user's identity, so identity questions can find it.
                    if _is_identity_statement(content) and "name" not in keywords:
                        keywords.append("name")
                    if not keywords:
                        keywords = _extract_keywords(content)

                    # Verbatim storage: the message itself is the memory — no
                    # judge paraphrase that would destroy the exact tokens
                    # recall needs. Capped so a pathological message can't
                    # bloat the store.
                    mem_content = content[:self.config.max_memory_chars]
                    importance = max(importance, self.config.auto_tag_threshold)

                    # Near-duplicate? Refresh its recency instead of storing a
                    # second copy of the same fact.
                    embedding = None
                    if self._embedder.enabled:
                        embedding = await self._embedder.embed(mem_content)
                    dup = self._find_duplicate(mem_content, embedding)
                    if dup is not None:
                        dup.importance = max(dup.importance, importance)
                        dup.last_accessed_at = time.time()
                        dup.access_count += 1
                        if dup.embedding is None:
                            dup.embedding = embedding
                        tagged = dup
                    else:
                        mem = HyperMem(
                            id=_next_id(),
                            content=mem_content,
                            created_at=time.time(),
                            last_accessed_at=time.time(),
                            access_count=0,
                            keywords=keywords,
                            importance=importance,
                            source="auto",
                            memory_type=mtype,
                            pinned=False,
                            subject=subject,
                            embedding=embedding,
                            source_message_id=msg.id,
                        )
                        # Type-aware supersession (conservative: only a clear
                        # correction of an ongoing attribute replaces the old
                        # fact; episodic events and topic overlaps coexist).
                        replaced = False
                        for ci in _find_conflicts(self.state.active, keywords, mem_content):
                            existing = self.state.active[ci]
                            should_replace, replacement = _resolve_conflict(
                                existing, mem_content, importance, mtype, subject, keywords)
                            if should_replace and replacement:
                                if replacement.embedding is None:
                                    replacement.embedding = embedding
                                existing.superseded_by = replacement.id
                                self.state.archive.append(existing)
                                self.state.active[ci] = replacement
                                tagged = replacement
                                replaced = True
                                break
                        if not replaced:
                            # No supersession — new memory (coexisting facts)
                            self.state.active.append(mem)
                            tagged = mem

        # 3. Archive if over limit (respecting decay)
        if len(self.state.active) > self.config.max_active_memories:
            scored = [(m, _apply_decay(m)) for m in self.state.active]
            scored.sort(key=lambda x: (not x[0].pinned, x[1]))
            to_archive = scored[:len(scored) - self.config.max_active_memories]
            for mem, _ in to_archive:
                self.state.archive.append(mem)
                self.state.active.remove(mem)

        # 4. Episodic → semantic consolidation (throttled, off by threshold 0)
        await self._maybe_consolidate()

        return AddMessageResult(self.state, tagged, recalled)

    def _to_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_memory_type(self, value, default: MemoryType) -> MemoryType:
        try:
            return MemoryType(str(value).strip().lower())
        except ValueError:
            return default

    def _find_duplicate(self, content: str,
                        embedding: Optional[list]) -> Optional[HyperMem]:
        """Find an active memory that is effectively the same fact.

        Matches on exact verbatim content first, then on embedding
        similarity when vectors are available (same fact, different wording).
        Returns None when nothing matches, so the caller stores a new memory.
        """
        norm = _normalize(content)
        for m in self.state.active:
            if norm and _normalize(m.content) == norm:
                return m
        if embedding:
            for m in self.state.active:
                if m.embedding and _cosine(embedding, m.embedding) > 0.88:
                    return m
        return None

    async def _maybe_consolidate(self) -> None:
        """Fuse a subject's oldest episodic memories into one knowledge memory.

        MemGPT-style episodic → semantic: once a subject accumulates at least
        ``consolidation_threshold`` events, the oldest are LLM-summarized into
        a single STATIC memory (``consolidated_from`` lists the originals) and
        the originals are archived — excluded from recall, exactly like a
        superseded fact. Runs synchronously on add_message, throttled to at
        most once per ``consolidation_interval`` messages. Disabled when the
        threshold is 0.
        """
        cfg = self.config
        if cfg.consolidation_threshold <= 0:
            return
        if (cfg.consolidation_interval > 0
                and self.state.total_messages - self._last_consolidation
                < cfg.consolidation_interval):
            return

        # Group episodic memories by subject (a subject is required — fusing
        # a jumble of unlabeled events would produce garbage).
        by_subject: dict[str, list[HyperMem]] = {}
        for m in self.state.active:
            if m.memory_type == MemoryType.EPISODIC and m.subject:
                by_subject.setdefault(m.subject.lower(), []).append(m)
        if not by_subject:
            return

        subject = max(by_subject, key=lambda s: len(by_subject[s]))
        group = by_subject[subject]
        if len(group) < cfg.consolidation_threshold:
            return
        # Oldest first — the events most worth folding into durable knowledge.
        group.sort(key=lambda m: m.created_at)
        oldest = group[:cfg.consolidation_threshold]

        result = await self._llm.complete(
            [{"role": "user", "content": _build_consolidation_prompt(subject, oldest)}],
            temperature=0.1, max_tokens=300,
        )
        summary = ""
        if result:
            parsed = extract_json_object(result) or {}
            summary = str(parsed.get("summary", "")).strip()
        if not summary:
            # Fallback: keep the events verbatim rather than losing them.
            summary = " ".join(m.content for m in oldest)[:self.config.max_memory_chars]

        consolidation = HyperMem(
            id=_next_id(),
            content=summary,
            created_at=time.time(),
            last_accessed_at=time.time(),
            access_count=0,
            keywords=list(dict.fromkeys(
                k for m in oldest for k in (m.keywords or []))),
            importance=min(1.0, max(m.importance for m in oldest)),
            source="auto",
            memory_type=MemoryType.STATIC,
            pinned=False,
            subject=subject,
            consolidated_from=[m.id for m in oldest],
        )
        for m in oldest:
            m.superseded_by = consolidation.id
            self.state.archive.append(m)
            self.state.active.remove(m)
        self.state.active.append(consolidation)
        self._last_consolidation = self.state.total_messages

    async def update_world_ida(self, user_msg: str, ai_msg: str,
                                persona_context: Optional[str] = None) -> None:
        from .world_ida import update_world_ida as _update, scene_transition_summary

        old_ida = self._world_ida
        new_ida = await _update(
            previous_ida=old_ida,
            last_user_message=user_msg,
            last_ai_response=ai_msg,
            persona_context=persona_context or (
                self.state.persona.description if self.state.persona else None
            ),
            llm_complete=self._llm.complete,
        )

        if old_ida is not None and new_ida.meta.scene_changed:
            summary = scene_transition_summary(old_ida)
            self.remember(summary)

        self.set_world_ida(new_ida)

    def remember(self, content: str, memory_type: MemoryType = MemoryType.STATIC) -> HyperMem:
        """Explicitly tell HyperMEM to remember something."""
        mem = HyperMem(
            id=_next_id(),
            content=content,
            created_at=time.time(),
            last_accessed_at=time.time(),
            access_count=0,
            keywords=_extract_keywords(content),
            importance=1.0,
            source="user",
            memory_type=memory_type,
            pinned=True,
        )
        self.state.active.append(mem)
        return mem

    def _recall_candidates(self) -> list[HyperMem]:
        """The searchable memory pool.

        Active memories by default. Superseded and consolidated originals are
        always excluded (the stale-leak fix — recall must never surface a fact
        the user corrected). Decay-archived memories are only searched when
        ``search_archive`` is enabled (opt-in forensic recall).
        """
        pool = list(self.state.active)
        if self.config.search_archive:
            pool += [m for m in self.state.archive if m.superseded_by is None]
        return pool

    async def _rank_memories(self, query: str) -> list[HyperMem]:
        """Rank the recall pool for a query, best first.

        With embeddings available → hybrid score (cosine + lexical overlap +
        importance + recency) plus a deterministic identity boost — no
        per-recall LLM rank call unless ``recall_use_llm`` asks for it.
        Without embeddings → the proven LLM+lexical path (unchanged).
        """
        pool = self._recall_candidates()
        if not pool:
            return []

        q_emb = None
        if self._embedder.enabled and self._embedder.available:
            q_emb = await self._embedder.embed(query)

        if q_emb is None:
            indices = await self._llm.find_relevant(query, pool)
            return [pool[i] for i in indices if i < len(pool)]

        q_tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        identity_query = _is_identity_query(query)
        now = time.time()
        scored = []
        for m in pool:
            score, _ = self._score_memory(query, m, q_emb, q_tokens,
                                          identity_query, now)
            scored.append((score, m))

        if self.config.recall_use_llm:
            # Second opinion: memories the LLM explicitly chose get a boost.
            indices = await self._llm.find_relevant(query, pool)
            boosted = {pool[i].id for i in indices if i < len(pool)}
            for i, (score, m) in enumerate(scored):
                if m.id in boosted:
                    scored[i] = (score + 0.5, m)

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []

        # Ambiguity tiebreak: when the leader doesn't clearly beat a close
        # runner-up, the hybrid scorer can't tell a fact from a
        # similar-sounding decoy ("There is a different bow called Starfall"
        # vs the real bow). Ask the LLM to adjudicate the near-tied band:
        # its picks get a boost, and the near-tied candidates it rejected are
        # excluded from this recall. Fires only when the scores are genuinely
        # close (``recall_ambiguity_gap``), so the embedding-fast path is
        # untouched for unambiguous queries.
        gap = self.config.recall_ambiguity_gap
        if (len(scored) >= 2 and gap > 0
                and scored[0][0] - scored[1][0] < gap):
            # The near-tied band: the leader and anything within 0.2 of the
            # runner-up, capped at 5 so the adjudication stays comparable but
            # still contains the true fact when several distractors outrank it.
            band = [m for score, m in scored
                    if score >= scored[1][0] - 0.2][:5]
            try:
                picks = await self._llm.find_relevant(query, band, raw=True)
                picked = {band[i].id for i in picks if i < len(band)}
            except Exception:
                picked = set()
            if picked:
                # The model compared the band and rejected the rest — they
                # are decoys, keep them out of the context window.
                decoys = {m.id for m in band if m.id not in picked}
                scored = [(s, m) for s, m in scored if m.id not in decoys]
                for i, (s, m) in enumerate(scored):
                    if m.id in picked:
                        scored[i] = (s + 1.5, m)
                scored.sort(key=lambda x: x[0], reverse=True)

        # Relative relevance floor: the base importance + recency terms give
        # every memory a floor score, so without a cutoff an off-topic query
        # would surface the whole store. Drop memories scoring below half the
        # best (absolute floor 0.3) — a clearly-relevant memory still pulls
        # close neighbors with it, but unrelated noise falls out.
        floor = max(0.3, 0.5 * scored[0][0])
        above_floor = []
        for score, m in scored:
            if score < floor:
                continue
            # Evidence gate: a memory with zero lexical overlap and only weak
            # semantic similarity is scoring on its importance+recency
            # baseline alone — that is not relevance ("The Ice King also
            # wanted the crown" vs "Who killed my father?"), so drop it.
            if self._has_evidence(m, q_tokens, q_emb):
                above_floor.append(m)

        # Diversity pass: never return two memories that say the same thing.
        # A candidate that is a near-duplicate (embedding cosine above the
        # ingest-dedup threshold) of one already chosen is dropped — the
        # context window shouldn't carry both "My name is Eldrin" and its
        # near-identical echo, and it keeps competing distractors from
        # surfacing alongside the real fact.
        kept: list[HyperMem] = []
        for m in above_floor:
            if any(_cosine(m.embedding, k.embedding) > 0.88
                   for k in kept if m.embedding and k.embedding):
                continue
            kept.append(m)
        return kept

    @staticmethod
    def _has_evidence(mem: HyperMem, q_tokens: set, q_emb) -> bool:
        """True if a memory shares real evidence with the query.

        Lexical keyword overlap, or strong semantic similarity, both count.
        A memory with neither is scoring on its importance+recency baseline
        alone — a standing score that no question earned it, so it never
        surfaces. With no embedding signal available we don't over-filter.
        """
        m_tokens = set(re.findall(r"[a-z0-9]{3,}", mem.content.lower()))
        m_tokens |= {kw.lower() for kw in (mem.keywords or [])}
        if len(q_tokens & m_tokens) > 0:
            return True
        if q_emb is not None and mem.embedding:
            return _cosine(q_emb, mem.embedding) >= 0.5
        return True

    def _score_memory(self, query: str, mem: HyperMem, q_emb, q_tokens: set,
                      identity_query: bool, now: float) -> tuple[float, dict]:
        """Hybrid score for one memory against a query, with the component
        breakdown (shared by ranking and by the provenance endpoint)."""
        sim = _cosine(q_emb, mem.embedding) if mem.embedding else 0.0
        m_tokens = set(re.findall(r"[a-z0-9]{3,}", mem.content.lower()))
        kwset = {kw.lower() for kw in (mem.keywords or [])}
        m_tokens |= kwset
        lex = len(q_tokens & m_tokens) / max(1, len(q_tokens))
        imp = _apply_decay(mem)
        rec = max(0.0, 1.0 - (now - mem.created_at) / (30 * 86400))
        identity_boost = 0.0
        echo_penalty = 0.0
        score = 2.0 * sim + 1.2 * lex + 0.5 * imp + 0.3 * rec
        if identity_query:
            if "name" in kwset or "identity" in kwset:
                identity_boost = 1.5  # the user-identity memory surfaces
            elif sim > 0.35:
                # Any other memory close to an identity question is a
                # lookalike ("My name is Eldrin's cousin…", "Shadow King's
                # true name is Malachar") — it competes with the real identity
                # but is never the answer, so sink it firmly.
                identity_boost = -1.5
            score += identity_boost
        elif _ECHO_MARKER.search(mem.content):
            # A secondary-instance echo loses ground to the primary fact about
            # the same subject — but only when it isn't itself the best match
            # (a top-ranked echo still clears the floor it defines).
            echo_penalty = -1.5
            score += echo_penalty
        breakdown = {
            "cosine": round(sim, 4),
            "lexical": round(lex, 4),
            "importance": round(imp, 4),
            "recency": round(rec, 4),
            "identity_boost": identity_boost,
            "echo_penalty": echo_penalty,
            "total": round(score, 4),
        }
        return score, breakdown

    async def explain_recall(self, query: str, memory_id: str) -> Optional[dict]:
        """Provenance: the score breakdown for one memory against a query.

        Returns the live hybrid-scoring components that decide whether this
        memory surfaces, or None if the memory isn't in the recall pool.
        """
        pool = self._recall_candidates()
        mem = next((m for m in pool if m.id == memory_id), None)
        if mem is None:
            return None
        q_emb = None
        if self._embedder.enabled and self._embedder.available:
            q_emb = await self._embedder.embed(query)
        q_tokens = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        _, breakdown = self._score_memory(
            query, mem, q_emb, q_tokens, _is_identity_query(query), time.time())
        return breakdown

    async def recall(self, query: str) -> RecallResult:
        """Recall the memories most relevant to a query.

        Ranked by the hybrid scorer, then capped to a token budget
        (``max_recall_tokens``) so the injected block stays small. Memories
        whose content duplicates a recent chat line are *not* skipped here:
        verbatim storage means a just-stored memory always matches the chat
        it came from, and ``recall()`` is the standalone API contract — it
        must return what it found. Callers building an injection can drop
        redundant memories themselves if they need to.
        """
        pool = self._recall_candidates()
        if not pool:
            return RecallResult([], "No memories stored")

        ranked = await self._rank_memories(query)
        budget = self.config.max_recall_tokens
        relevant: list[HyperMem] = []
        used = 0
        for mem in ranked:
            est = max(1, len(mem.content) // 4)
            if used + est > budget and relevant:
                break
            relevant.append(mem)
            used += est
            for active_mem in self.state.active:
                if active_mem.id == mem.id:
                    active_mem.last_accessed_at = time.time()
                    active_mem.access_count += 1
                    break

        return RecallResult(relevant, f"Found {len(relevant)} relevant memories")

    async def get_context(self, current_message: str) -> str:
        """Build context prompt with world state + relevant memories."""
        parts = []

        if hasattr(self, '_world_ida') and self._world_ida is not None:
            from .world_ida import world_ida_to_context_string
            ctx_str = world_ida_to_context_string(self._world_ida)
            if ctx_str:
                parts.append(f"[WORLD STATE]\n{ctx_str}\n[/WORLD STATE]")

        recall = await self.recall(current_message)
        if recall.relevant:
            mem_block = "[RELEVANT MEMORIES]\n" + "\n".join(
                f"- {m.content} (importance: {round(m.importance * 100)}%)"
                for m in recall.relevant
            ) + "\n[/RELEVANT MEMORIES]"
            parts.append(mem_block)

        if self.state.recent_messages:
            chat = "\n".join(
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
                for m in self.state.recent_messages[-10:]
            )
            parts.append(chat)

        return "\n\n".join(parts)

    def memories(self) -> list[dict]:
        return [
            {
                "id": m.id,
                "content": m.content,
                "keywords": m.keywords,
                "importance": _apply_decay(m),
                "memory_type": m.memory_type.value,
                "source": m.source,
                "pinned": m.pinned,
                "access_count": m.access_count,
                "superseded_by": m.superseded_by,
                "created_at": m.created_at,
            }
            for m in self.state.active + self.state.archive
        ]

    # ---- Persistence ----

    def to_dict(self) -> dict:
        """Serialize the full engine state (memories + worldIDA) to a dict."""
        from .world_ida import _ida_to_dict
        data = state_to_dict(self.state)
        if self._world_ida is not None:
            data["world_ida"] = _ida_to_dict(self._world_ida)
        if self._world_ida_store is not None:
            data["world_ida_store"] = self._world_ida_store.to_dict()
        return data

    def from_dict(self, data: dict) -> None:
        """Restore the full engine state from a dict."""
        from .world_ida import _ida_from_dict, WorldIDAStore
        self.state = state_from_dict(data)
        if "world_ida" in data:
            self._world_ida = _ida_from_dict(data["world_ida"])
        if "world_ida_store" in data:
            store = WorldIDAStore()
            store.from_dict(data["world_ida_store"])
            self._world_ida_store = store

    def save(self, path: str):
        """Persist full engine state to JSON (atomic write)."""
        import json
        import os
        import tempfile

        target = os.path.abspath(path)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp, target)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self, path: str):
        """Restore full engine state from a JSON file."""
        import json
        with open(path, encoding="utf-8") as f:
            self.from_dict(json.load(f))
