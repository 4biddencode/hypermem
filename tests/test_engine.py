"""Tests for the HyperMEM core engine.

Covers: add_message, recall, remember, get_context, persona isolation,
memory types, decay, contradiction resolution, persistence, edge cases.
"""

import json
import time
import pytest
from hypermem import HyperMEM, HyperMemConfig
from hypermem.types import (
    MemoryType, Persona, HyperMem, HyperMemState,
    state_to_dict, state_from_dict,
)
from hypermem.engine import (
    _extract_keywords, _resolve_conflict, _find_conflicts,
    _apply_decay, _build_judge_prompt, _is_identity_statement,
    _is_identity_query,
)
from conftest import make_llm


# ---- Helpers ----

def make_hm(**kwargs) -> HyperMEM:
    client, _ = make_llm()
    return HyperMEM(HyperMemConfig(**kwargs), llm=client)


def make_mem(content: str, memory_type: MemoryType = MemoryType.EPISODIC,
             importance: float = 0.8) -> HyperMem:
    return HyperMem(
        id=f"test_{int(time.time() * 1000)}",
        content=content,
        created_at=time.time(),
        last_accessed_at=time.time(),
        access_count=0,
        keywords=_extract_keywords(content),
        importance=importance,
        source="auto",
        memory_type=memory_type,
        pinned=False,
    )


# ---- _extract_keywords ----

class TestExtractKeywords:
    def test_basic_extraction(self):
        kw = _extract_keywords("My name is Eldrin from Silverwood")
        assert "eldrin" in kw
        assert "silverwood" in kw
        assert "from" not in kw  # stopword

    def test_short_words_filtered(self):
        kw = _extract_keywords("hi ok go")
        assert kw == []

    def test_deduplication(self):
        kw = _extract_keywords("forest forest tree")
        assert kw == ["forest", "tree"]
        assert len(kw) == 2


# ---- Persona isolation ----

class TestPersonaIsolation:
    def test_persona_set_and_retrieve(self):
        hm = make_hm()
        p = Persona(name="Elena", description="An elven rogue", traits=["witty", "guarded"])
        hm.set_persona(p)
        assert hm.state.persona is not None
        assert hm.state.persona.name == "Elena"

    def test_judge_prompt_excludes_persona(self):
        p = Persona(name="Elena", description="An elven rogue", traits=["witty"])
        prompt = _build_judge_prompt("My name is Bob", p)
        assert "DO NOT store these as memories" in prompt
        assert "Elena" in prompt
        assert "elven rogue" in prompt

    def test_judge_prompt_no_persona(self):
        prompt = _build_judge_prompt("My name is Bob")
        assert "Persona context" not in prompt


# ---- Memory types ----

class TestMemoryTypes:
    def test_static_memory_created(self):
        hm = make_hm()
        mem = hm.remember("User name is Bob", MemoryType.STATIC)
        assert mem.memory_type == MemoryType.STATIC
        assert mem.pinned is True

    def test_episodic_memory_created(self):
        hm = make_hm()
        mem = hm.remember("Went to the forest", MemoryType.EPISODIC)
        assert mem.memory_type == MemoryType.EPISODIC


# ---- Auto-tag roles ----

class TestAutoTagRoles:
    @pytest.mark.asyncio
    async def test_default_only_tags_user(self):
        """By default, assistant messages are not judged."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("assistant", "My name is Elena and I was a court scribe")
        assert len(hm.state.active) == 0

    @pytest.mark.asyncio
    async def test_assistant_messages_can_be_tagged(self):
        """With auto_tag_roles including 'assistant', reveals are stored."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(auto_tag_roles=("user", "assistant")), llm=client)
        await hm.add_message("assistant", "My name is Elena and I was a court scribe")
        assert len(hm.state.active) == 1
        assert "Elena" in hm.state.active[0].content


# ---- Decay ----

class TestDecay:
    def test_static_never_decays(self):
        mem = make_mem("Static fact", MemoryType.STATIC, importance=0.9)
        score = _apply_decay(mem)
        assert score == 0.9  # Unchanged

    def test_episodic_decays_with_time(self):
        mem = make_mem("Episodic event", MemoryType.EPISODIC, importance=0.9)
        mem.last_accessed_at = time.time() - 7200  # 2 hours ago
        score = _apply_decay(mem)
        assert score < 0.9  # Decayed

    def test_episodic_decays_with_access_count(self):
        mem = make_mem("Frequently accessed", MemoryType.EPISODIC, importance=0.9)
        mem.access_count = 50
        score = _apply_decay(mem)
        assert score < 0.9


# ---- Contradiction resolution ----

class TestContradictionResolution:
    def test_static_supersedes(self):
        existing = make_mem("User name is Bob", MemoryType.STATIC)
        should_replace, replacement = _resolve_conflict(
            existing, "Actually, my name is Robert", 0.8,
            MemoryType.STATIC, subject="user", new_keywords=["name", "robert"])
        assert should_replace is True
        assert replacement is not None
        assert "Robert" in replacement.content

    def test_episodic_appends(self):
        existing = make_mem("Went to the forest", MemoryType.EPISODIC)
        should_replace, replacement = _resolve_conflict(existing, "Went to the mountains", 0.8)
        assert should_replace is False
        assert replacement is None

    def test_temporal_replaces(self):
        existing = make_mem("Mood: happy", MemoryType.TEMPORAL)
        should_replace, replacement = _resolve_conflict(
            existing, "Actually, my mood is sad", 0.8,
            MemoryType.TEMPORAL, subject="user", new_keywords=["mood", "sad"])
        assert should_replace is True
        assert replacement is not None

    def test_shared_keyword_alone_never_supersedes(self):
        """Two static facts sharing a topic ("crown") must coexist — the
        "silently lost facts" regression: the crown's power must not delete
        the crown quest just because both mention the crown."""
        existing = make_mem("Searching for the Lost Crown of Aetheria in the Dragon's Maw",
                            MemoryType.STATIC)
        should_replace, replacement = _resolve_conflict(
            existing, "The crown can control the weather when worn", 0.8,
            MemoryType.STATIC, subject="crown", new_keywords=["crown", "weather", "control"])
        assert should_replace is False
        assert replacement is None

    def test_find_conflicts_keyword_overlap(self):
        mems = [make_mem("User name is Bob")]
        conflicts = _find_conflicts(mems, ["bob", "name"], "My name is Bob")
        assert len(conflicts) == 1

    def test_find_conflicts_no_overlap(self):
        mems = [make_mem("Went to the forest")]
        conflicts = _find_conflicts(mems, ["bob", "name"], "My name is Bob")
        assert len(conflicts) == 0


# ---- Coexistence & identity tagging ----

class TestCoexistence:
    @pytest.mark.asyncio
    async def test_conflicting_episodic_facts_both_stored(self):
        """A keyword-overlap conflict on episodic memories must NOT drop the
        new fact — the fix for silently lost memories ("Father killed by the
        Shadow King" vanished because the bow memory shared the keyword
        "father")."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        r1 = await hm.add_message("user", "I love hiking in the Alps")
        r2 = await hm.add_message("user", "I hate hiking in the Alps")
        assert r1.tagged is not None
        assert r2.tagged is not None
        assert len(hm.state.active) == 2
        contents = [m.content.lower() for m in hm.state.active]
        assert any("love" in c for c in contents)
        assert any("hate" in c for c in contents)

    @pytest.mark.asyncio
    async def test_identity_message_tagged_with_name_keyword(self):
        """'My name is X' / "I'm called X" facts carry the exact 'name'
        keyword so identity questions can find them via the fallback, even
        when the LLM recall refuses to rank."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "My name is Eldrin from Silverwood")
        assert "name" in hm.state.active[0].keywords

    @pytest.mark.asyncio
    async def test_first_person_identity_tagged(self):
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "I'm called Eldrin, a ranger")
        assert "name" in hm.state.active[0].keywords

    @pytest.mark.asyncio
    async def test_other_character_identity_not_tagged(self):
        """Another character's 'true name' must NOT carry the user-identity
        tag, so 'What's my name?' never surfaces it. A possessive lookalike
        ('My name is Eldrin's cousin') names a relative, not the user."""
        assert not _is_identity_statement("Shadow King's true name is Malachar")
        assert not _is_identity_statement("The king was called Eldrin")
        assert not _is_identity_statement("Moonwhisper was blessed by the High Elves")
        assert not _is_identity_statement("My name is Eldrin's cousin, from the same forest.")

    @pytest.mark.asyncio
    async def test_identity_helper_first_person_only(self):
        assert _is_identity_statement("My name is Eldrin, an elven ranger from Silverwood.")
        assert _is_identity_statement("I am Eldrin")
        assert _is_identity_statement("I'm called Eldrin")
        assert _is_identity_statement("I'm known as Eldrin")

    @pytest.mark.asyncio
    async def test_identity_query_user_only(self):
        """Only questions about the USER's own identity qualify. 'my bow
        called' asks about a thing and must not trigger the identity boost."""
        assert _is_identity_query("What's my name?")
        assert _is_identity_query("Who am I?")
        assert _is_identity_query("What is my name?")
        assert not _is_identity_query("What's my bow called?")
        assert not _is_identity_query("What's my sister's name?")
        assert not _is_identity_query("Shadow King's true name?")


# ---- Distractor resistance: near-dup facts must not leak into recall ----

class TestDistractorResistance:
    """Regression suite for the distractor leak (benchmarks/suite_distractors).
    Near-duplicate facts that share keywords with the true answer must not
    surface when the same question is asked — the echo demotion, identity
    decoy sink, and ambiguity tiebreak keep the primary fact on top."""

    @pytest.mark.asyncio
    async def test_echo_secondary_instance_not_recalled(self):
        """A secondary 'also opens with' memory is demoted so the primary
        fact outranks it and the ambiguity tiebreak does not fire (which would
        hand the echo the win). The primary fact must be the top hit."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "The vault password is Starlight")
        await hm.add_message("user", "The vault also opens with the Moonrise")
        r = await hm.recall("What opens the vault?")
        contents = [m.content.lower() for m in r.relevant]
        assert any("starlight" in c for c in contents)
        # The echo is demoted below the primary, so the fact-carrying memory
        # is the top hit — the tiebreak can't hand the win to the decoy.
        assert "starlight" in contents[0]

    @pytest.mark.asyncio
    async def test_identity_lookalike_not_recalled_for_name_question(self):
        """'My name is Eldrin's cousin' names a relative, not the user, so
        'What's my name?' must surface the true user identity memory only."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "My name is Eldrin, an elven ranger")
        await hm.add_message("user", "My name is Eldrin's cousin from the forest")
        r = await hm.recall("What's my name?")
        contents = [m.content.lower() for m in r.relevant]
        assert any("elven ranger" in c for c in contents)
        assert not any("cousin" in c for c in contents)

    @pytest.mark.asyncio
    async def test_ambiguity_tiebreak_excludes_rejected_decoy(self):
        """When two near-tied candidates are ambiguous, the LLM adjudicates;
        a decoy it rejects is excluded from the returned recall."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "The crown controls the weather")
        await hm.add_message("user", "The crown also grants invisibility")
        r = await hm.recall("What does the crown do?")
        contents = [m.content.lower() for m in r.relevant]
        assert any("weather" in c for c in contents)
        assert not any("invisibility" in c for c in contents)


# ---- Ingestion: verbatim storage, gating, dedup, supersession ----

class TestIngestion:
    @pytest.mark.asyncio
    async def test_message_stored_verbatim(self):
        """The stored memory is the user's exact message — the judge classifies,
        it does not rewrite (paraphrase drift destroyed recall substrings)."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        msg = "My name is Eldrin and I grew up in Silverwood"
        await hm.add_message("user", msg)
        assert len(hm.state.active) == 1
        assert hm.state.active[0].content == msg

    @pytest.mark.asyncio
    async def test_judge_classification_captured(self):
        """subject / memory_type / source_message_id come from the judge."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "My name is Eldrin")
        mem = hm.state.active[0]
        assert mem.memory_type == MemoryType.EPISODIC  # stub default
        assert mem.subject  # stub assigns the first keyword
        assert mem.source_message_id is not None
        assert "name" in mem.keywords  # deterministic identity tag

    @pytest.mark.asyncio
    async def test_identical_message_deduped(self):
        """Re-sending the same message refreshes the memory instead of
        storing a second copy."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "I live in Vienna")
        r2 = await hm.add_message("user", "I live in Vienna")
        assert len(hm.state.active) == 1
        assert r2.tagged is hm.state.active[0]
        assert hm.state.active[0].access_count >= 1  # refreshed, not duplicated

    @pytest.mark.asyncio
    async def test_embedding_near_duplicate_deduped(self):
        """Same fact, slightly different wording, becomes one memory when
        embeddings are enabled (semantic dedup)."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(embedding_provider="ollama"), llm=client)
        await hm.add_message("user", "I love pizza and I love pasta")
        await hm.add_message("user", "I love pizza and pasta")
        assert len(hm.state.active) == 1

    @pytest.mark.asyncio
    async def test_correction_supersedes_old_static_fact(self):
        """'Actually, I changed the vault password…' replaces the old vault
        fact (same subject, correction cue) — the contradiction fix."""
        client, _ = make_llm(memory_type="static")
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "The vault password is Starlight")
        await hm.add_message("user", "Actually, I changed the vault password to Midnight")
        assert len(hm.state.active) == 1
        assert "Midnight" in hm.state.active[0].content
        assert len(hm.state.archive) == 1  # old fact archived, superseded

    @pytest.mark.asyncio
    async def test_topic_overlap_coexists(self):
        """Two static facts sharing a topic ('crown') must both be stored —
        never drop a fact just because it shares a keyword with another."""
        client, _ = make_llm(memory_type="static")
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "Searching for the Lost Crown of Aetheria in the Dragon's Maw")
        await hm.add_message("user", "The crown can control the weather when worn")
        assert len(hm.state.active) == 2


# ---- Recall: hybrid scoring, budget, relevance floor ----

class TestRecallScoring:
    @pytest.mark.asyncio
    async def test_token_budget_caps_context(self):
        """max_recall_tokens bounds how much context recall injects, even
        when every memory clears the relevance floor."""
        # The ambiguity tiebreak may fire on the two near-tied pocket-watch
        # facts; a sensible model returns both as relevant (no conflict), so
        # the stub simulates that instead of always single-picking.
        client, _ = make_llm(recall_response=lambda q, ms: (
            "[" + ",".join(str(i + 1) for i in range(len(ms))) + "]"))
        hm = HyperMEM(HyperMemConfig(max_recall_tokens=20), llm=client)
        await hm.add_message("user", "I collect antique pocket watches")
        await hm.add_message("user", "My pocket watch was my grandfather's")
        await hm.add_message("user", "I wind my pocket watch every morning")
        r = await hm.recall("Tell me about my pocket watch")
        # All three are relevant (shared "pocket" keyword) and clear the floor,
        # yet distinct enough not to dedup, but ~30-45 chars each ≈ 8-11
        # tokens; budget 20 fits the first two only.
        assert len(r.relevant) == 2

    @pytest.mark.asyncio
    async def test_off_topic_below_floor_pruned(self):
        """A memory with no query overlap is dropped by the relative floor,
        even when it is recent and important — recall must not surface the
        whole store for an off-topic question."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "My name is Emanuel and I love hiking")
        await hm.add_message("user", "My sister Lyra collects rare orchids")
        r = await hm.recall("Tell me about my sister")
        assert any("sister" in m.content.lower() for m in r.relevant)
        assert not any("emanuel" in m.content.lower() for m in r.relevant)

    @pytest.mark.asyncio
    async def test_recall_use_llm_second_opinion_called(self):
        """recall_use_llm=True consults the LLM rank as a second opinion
        without breaking hybrid ranking (regression: the boost used to
        mutate an unpacked tuple and never change the order)."""
        client, stub = make_llm(recall_response=lambda q, mems: "[]")
        hm = HyperMEM(HyperMemConfig(recall_use_llm=True), llm=client)
        await hm.add_message("user", "I have a dog named Rex and he lives in Vienna")
        r = await hm.recall("Where does Rex live?")
        assert len(r.relevant) == 1
        assert "Vienna" in r.relevant[0].content
        recall_calls = [c for c in stub.calls
                        if any("find memories relevant" in str(m.get("content", ""))
                               for m in c.get("messages", []))]
        assert len(recall_calls) >= 1  # the LLM was actually consulted


# ---- Lifecycle: forgetting + consolidation ----

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_recall_excludes_superseded(self):
        """After a correction supersedes a fact, recall must not return the
        stale (superseded) version — the deterministic stale-leak fix."""
        client, _ = make_llm(memory_type="static")
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "The vault password is Starlight")
        await hm.add_message("user", "Actually, I changed the vault password to Midnight")
        recall = await hm.recall("What's the vault password?")
        assert len(recall.relevant) == 1
        assert "Midnight" in recall.relevant[0].content
        assert "Starlight" not in recall.relevant[0].content

    @pytest.mark.asyncio
    async def test_search_archive_opt_in(self):
        """Decay-archived memories are excluded from recall by default but
        searchable when search_archive is enabled (forensic recall)."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(max_active_memories=1), llm=client)
        await hm.add_message("user", "I love hiking in the Alps")
        await hm.add_message("user", "My sister Lyra lives in Oakvale")
        assert len(hm.state.active) == 1
        assert len(hm.state.archive) == 1
        recall = await hm.recall("What do I love?")
        assert not any("hiking" in m.content.lower() for m in recall.relevant)

        hm2 = HyperMEM(HyperMemConfig(max_active_memories=1, search_archive=True), llm=client)
        await hm2.add_message("user", "I love hiking in the Alps")
        await hm2.add_message("user", "My sister Lyra lives in Oakvale")
        recall2 = await hm2.recall("What do I love?")
        assert any("hiking" in m.content.lower() for m in recall2.relevant)

    @pytest.mark.asyncio
    async def test_episodic_consolidation(self):
        """Once a subject accumulates >= threshold episodic events, the oldest
        are fused into one static knowledge memory and the originals archived."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(consolidation_threshold=3, consolidation_interval=1),
                      llm=client)
        await hm.add_message("user", "I journeyed to the black mountain")
        await hm.add_message("user", "I climbed the black mountain")
        await hm.add_message("user", "I camped at the black mountain")

        assert len(hm.state.active) == 1
        fused = hm.state.active[0]
        assert fused.memory_type == MemoryType.STATIC
        assert len(fused.consolidated_from or []) == 3
        assert len(hm.state.archive) == 3
        assert all(m.superseded_by == fused.id for m in hm.state.archive)

    @pytest.mark.asyncio
    async def test_consolidation_throttled_by_interval(self):
        """Consolidation does not fire until consolidation_interval messages
        have passed since the last run."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(consolidation_threshold=2, consolidation_interval=10),
                      llm=client)
        await hm.add_message("user", "I journeyed to the black mountain")
        await hm.add_message("user", "I climbed the black mountain")
        # Only 2 messages so far, interval is 10, so no consolidation yet
        assert len(hm.state.archive) == 0
        assert len(hm.state.active) == 2


# ---- Edge cases ----

class TestEdgeCases:
    def test_empty_conversation(self):
        hm = make_hm()
        assert hm.state.total_messages == 0
        assert hm.memories() == []

    @pytest.mark.asyncio
    async def test_single_very_long_message(self):
        hm = make_hm()
        long_msg = "word " * 10000
        result = await hm.add_message("user", long_msg)
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_extractable_facts(self):
        hm = HyperMEM(HyperMemConfig(auto_tag_threshold=0.9))  # High threshold
        result = await hm.add_message("user", "Ok.")
        # Filler — should skip LLM entirely
        assert result.tagged is None

    def test_memories_list_format(self):
        hm = make_hm()
        hm.remember("Test memory")
        mems = hm.memories()
        assert len(mems) == 1
        assert "id" in mems[0]
        assert "content" in mems[0]
        assert "importance" in mems[0]
        assert "memory_type" in mems[0]

    def test_superseded_memory_tracked(self):
        existing = make_mem("User name is Bob", MemoryType.STATIC)
        should_replace, replacement = _resolve_conflict(
            existing, "Actually, my name is Robert", 0.8,
            MemoryType.STATIC, subject="user", new_keywords=["name", "robert"])
        assert should_replace
        existing.superseded_by = replacement.id
        assert existing.superseded_by == replacement.id


# ---- Persistence ----

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        hm = make_hm()
        hm.remember("Test memory")
        path = tmp_path / "test.json"
        hm.save(str(path))

        hm2 = make_hm()
        hm2.load(str(path))
        assert hm2.state.total_messages == hm.state.total_messages
        assert len(hm2.memories()) == len(hm.memories())
        assert hm2.memories()[0]["content"] == hm.memories()[0]["content"]

    def test_state_to_dict_from_dict(self):
        state = HyperMemState(conversation_id="test")
        state.total_messages = 42
        d = state_to_dict(state)
        restored = state_from_dict(d)
        assert restored.conversation_id == "test"
        assert restored.total_messages == 42

    def test_state_from_dict_tolerant_of_stale_json(self):
        """A memory missing fields / with unknown keys / bad memory_type must
        load without crashing (tolerant of pre-rework or hand-edited JSON)."""
        # Missing required access_count + an unknown extra field.
        stale = {
            "conversation_id": "c1",
            "active": [{
                "id": "m1", "content": "hello", "created_at": 1.0,
                "last_accessed_at": 1.0, "keywords": [], "importance": 0.8,
                "source": "user", "memory_type": "static",
                "not_a_real_field": "x",
            }],
            "archive": [], "recent_messages": [], "total_messages": 1,
        }
        s = state_from_dict(stale)
        assert len(s.active) == 1
        assert s.active[0].access_count == 0  # defaulted
        assert s.active[0].memory_type == MemoryType.STATIC

        # Invalid memory_type falls back to EPISODIC.
        stale["active"][0]["memory_type"] = "BOGUS"
        s2 = state_from_dict(stale)
        assert s2.active[0].memory_type == MemoryType.EPISODIC

    def test_persona_persists(self, tmp_path):
        hm = make_hm()
        hm.set_persona(Persona(name="Elena", description="Rogue"))
        path = tmp_path / "persona_test.json"
        hm.save(str(path))

        hm2 = make_hm()
        hm2.load(str(path))
        assert hm2.state.persona is not None
        assert hm2.state.persona.name == "Elena"

    def test_memory_type_persists(self, tmp_path):
        hm = make_hm()
        hm.remember("Static fact", MemoryType.STATIC)
        hm.remember("Episodic event", MemoryType.EPISODIC)
        path = tmp_path / "types_test.json"
        hm.save(str(path))

        hm2 = make_hm()
        hm2.load(str(path))
        mems = hm2.memories()
        types = [m["memory_type"] for m in mems]
        assert "static" in types
        assert "episodic" in types


class TestContentAgnosticAllRP:
    """HyperMEM must serve every roleplay register identically: SFW,
    suggestive, explicit. The judge classifies, storage is verbatim, and
    worldIDA tracks relationship state -- none of it may filter or reshape
    content based on how spicy it is."""

    async def _plant_and_recall(self, hm, content: str, query: str) -> list:
        res = await hm.add_message("user", content)
        assert res.tagged is not None, "the RP fact was not stored"
        assert res.tagged.content == content, "memory must be verbatim, not rewritten"
        recall = await hm.recall(query)
        return recall.relevant

    @pytest.mark.asyncio
    async def test_explicit_fact_stored_verbatim_and_recalled(self):
        hm = make_hm()
        content = ("I love the way you kiss my neck - that night at the cabin "
                   "was unforgettable.")
        relevant = await self._plant_and_recall(hm, content, "What happened at the cabin?")
        assert any("cabin" in m.content for m in relevant)

    @pytest.mark.asyncio
    async def test_all_registers_coexist_without_discrimination(self):
        hm = make_hm()
        spicy = "Our first time was slow and gentle - I felt completely safe."
        sweet = "We adopted a corgi named Biscuit."
        await self._plant_and_recall(hm, spicy, "first time")
        await self._plant_and_recall(hm, sweet, "corgi")
        contents = [m["content"] for m in hm.memories()]
        assert spicy in contents and sweet in contents, \
            "both registers must be remembered, none filtered"

    @pytest.mark.asyncio
    async def test_relationship_stage_in_worldida(self):
        from hypermem.types import Persona
        hm = make_hm()
        hm.set_persona(Persona(name="Kai", description="A patient lover"))
        await hm.update_world_ida(
            "I'm ready for more. Is that okay?",
            "*Kai draws closer, checking your eyes.* Yes. Tell me if anything feels wrong.",
            persona_context="Kai, a patient lover.",
        )
        ida = hm.get_world_ida()
        assert ida is not None
        # relationship state carries over across turns
        await hm.update_world_ida(
            "That was beautiful.",
            "*Kai holds you.*",
            persona_context="Kai, a patient lover.",
        )
        ida2 = hm.get_world_ida()
        assert ida2 is not None and ida2.relationship is not None
