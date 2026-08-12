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
        should_replace, replacement = _resolve_conflict(existing, "User name is Robert", 0.8)
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
        should_replace, replacement = _resolve_conflict(existing, "Mood: sad", 0.8)
        assert should_replace is True
        assert replacement is not None

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
        tag, so 'What's my name?' never surfaces it."""
        assert not _is_identity_statement("Shadow King's true name is Malachar")
        assert not _is_identity_statement("The king was called Eldrin")
        assert not _is_identity_statement("Moonwhisper was blessed by the High Elves")

    @pytest.mark.asyncio
    async def test_identity_helper_first_person_only(self):
        assert _is_identity_statement("My name is Eldrin, an elven ranger from Silverwood.")
        assert _is_identity_statement("I am Eldrin")
        assert _is_identity_statement("I'm called Eldrin")
        assert _is_identity_statement("I'm known as Eldrin")


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
        should_replace, replacement = _resolve_conflict(existing, "User name is Robert", 0.8)
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
