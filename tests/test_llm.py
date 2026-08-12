"""Tests for the LLM client: parsing, protocol, judge floor and recall fallback."""

import json
import time
import pytest

from hypermem import HyperMEM, HyperMemConfig
from hypermem.llm import _extract_indices, LLMClient
from hypermem.types import HyperMem
from conftest import make_llm


# ---- _extract_indices parsing ----

class TestExtractIndices:
    def test_json_array(self):
        assert _extract_indices("[3, 1]") == [2, 0]

    def test_fenced_json(self):
        assert _extract_indices("```json\n[2]\n```") == [1]

    def test_json_object(self):
        assert _extract_indices('{"indices": [4, 2]}') == [3, 1]

    def test_comma_list(self):
        assert _extract_indices("3, 1") == [2, 0]

    def test_bare_number(self):
        assert _extract_indices("5") == [4]

    def test_empty_array(self):
        assert _extract_indices("[]") == []

    def test_garbage(self):
        assert _extract_indices("no matching memory") == []

    def test_zero_and_shuffled(self):
        assert _extract_indices("[0, 2]") == [1]


# ---- Judge floor: extracted fact always stored ----

class TestJudgeFloor:
    @pytest.mark.asyncio
    async def test_low_importance_still_stored(self):
        """Judge returns importance below threshold but a real fact → stored."""
        client, _ = make_llm()
        hm = HyperMEM(HyperMemConfig(auto_tag_threshold=0.7), llm=client)

        result = await hm.add_message("user", "My brother owns a tavern in Duskport")
        assert result.tagged is not None
        assert result.tagged.importance >= 0.7  # floored, not dropped

    @pytest.mark.asyncio
    async def test_importance_field_is_used(self):
        """Judge importance flows through and ranks the memory."""
        client, _ = make_llm(importance=0.9)
        hm = HyperMEM(HyperMemConfig(auto_tag_threshold=0.3), llm=client)

        result = await hm.add_message("user", "I study astronomy at night")
        assert result.tagged is not None
        assert result.tagged.importance == 0.9


# ---- Recall via the real prompt pipeline ----

class TestRecallPipeline:
    @pytest.mark.asyncio
    async def test_recall_returns_ranked_list(self):
        """LLM may return several indices; recall honors the order."""
        client, _ = make_llm(recall_response=lambda q, mems: "[2, 1]")
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "fact one about mountains")
        await hm.add_message("user", "fact two about oceans")
        result = await hm.recall("tell me something")
        assert len(result.relevant) == 2
        assert result.relevant[0].content.startswith("fact two")

    @pytest.mark.asyncio
    async def test_empty_list_uses_keyword_fallback(self):
        """LLM returns [] but a memory shares keywords → still recalled."""
        client, _ = make_llm(recall_response=lambda q, mems: "[]")
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "I have a dog named Rex and he lives in Berlin")
        result = await hm.recall("Where does Rex live?")
        assert len(result.relevant) == 1
        assert "Berlin" in result.relevant[0].content

    @pytest.mark.asyncio
    async def test_no_overlap_no_recall(self):
        client, _ = make_llm(recall_response=lambda q, mems: "[]")
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "My favorite color is teal")
        result = await hm.recall("What is the weather like today?")
        assert result.relevant == []

    @pytest.mark.asyncio
    async def test_out_of_range_indices_ignored(self):
        client, _ = make_llm(recall_response=lambda q, mems: "[7, 2]")
        hm = HyperMEM(HyperMemConfig(), llm=client)
        await hm.add_message("user", "only one fact here")
        result = await hm.recall("some query")
        assert result.relevant == []  # both indices exceed the memory list


# ---- Keyword fallback implementation ----

class TestKeywordFallback:
    def test_overlap_ranking(self):
        mems = [
            HyperMem(id="1", content="I love hiking near the Alps", created_at=time.time(),
                     last_accessed_at=time.time(), access_count=0, keywords=["alps"],
                     importance=0.5, source="auto"),
            HyperMem(id="2", content="My cat is named Pixel", created_at=time.time(),
                     last_accessed_at=time.time(), access_count=0, keywords=[],
                     importance=0.9, source="auto"),
        ]
        idx = LLMClient._keyword_fallback("tell me about the Alps", mems)
        assert idx == [0]

    def test_no_match(self):
        mems = [
            HyperMem(id="1", content="totally unrelated fact here", created_at=time.time(),
                     last_accessed_at=time.time(), access_count=0, keywords=[],
                     importance=0.5, source="auto"),
        ]
        assert LLMClient._keyword_fallback("quantum physics trivia", mems) == []

    def test_identity_boost_picks_user_identity(self):
        """'What's my name?' must surface the identity-tagged memory (exact
        'name' keyword), not the lookalike 'true name is Malachar' memory."""
        mems = [
            HyperMem(id="1", content="Eldrin is an elven ranger from Silverwood.",
                     created_at=time.time(), last_accessed_at=time.time(), access_count=0,
                     keywords=["eldrin", "name"], importance=0.8, source="auto"),
            HyperMem(id="2", content="Shadow King's true name is Malachar.",
                     created_at=time.time(), last_accessed_at=time.time(), access_count=0,
                     keywords=["shadow", "king", "true name", "malachar"],
                     importance=0.8, source="auto"),
        ]
        idx = LLMClient._keyword_fallback("What's my name?", mems)
        assert idx[0] == 0

    def test_identity_boost_not_needed_for_non_identity_query(self):
        mems = [
            HyperMem(id="1", content="Eldrin is an elven ranger from Silverwood.",
                     created_at=time.time(), last_accessed_at=time.time(), access_count=0,
                     keywords=["eldrin", "name"], importance=0.8, source="auto"),
            HyperMem(id="2", content="Shadow King's true name is Malachar.",
                     created_at=time.time(), last_accessed_at=time.time(), access_count=0,
                     keywords=["shadow", "king", "true name", "malachar"],
                     importance=0.8, source="auto"),
        ]
        idx = LLMClient._keyword_fallback("What's the Shadow King's name?", mems)
        assert idx[0] == 1


# ---- HTTP protocol level ----

class TestTransportProtocol:
    @pytest.mark.asyncio
    async def test_ollama_request_shape(self):
        """The request sent to the model has the right Ollama wire format."""
        client, stub = make_llm()
        await client.complete(
            [{"role": "user", "content": "Extract the key factual information now"}],
            temperature=0.1, max_tokens=100,
        )
        assert len(stub.calls) == 1
        payload = stub.calls[0]
        assert payload["model"] == "qwen2.5:7b"
        assert payload["stream"] is False
        assert payload["options"]["num_predict"] == 100
        assert "temperature" in payload["options"]

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """5xx responses are retried, then None is returned."""
        import httpx
        attempts = {"n": 0}

        def flaky(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(500, json={"error": "boom"})

        client = LLMClient(
            provider="ollama", model="m",
            endpoint="http://stub.local:11434",
            timeout=2,
            transport=httpx.MockTransport(flaky),
        )
        result = await client.complete([{"role": "user", "content": "hi"}])
        assert result is None
        assert attempts["n"] == 1 + 2  # initial + MAX_RETRIES

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        import httpx
        import asyncio

        async def slow(transport, request):
            await asyncio.sleep(5)
            return httpx.Response(200, json={"message": {"content": "late"}})

        client = LLMClient(
            provider="ollama", model="m",
            endpoint="http://stub.local:11434",
            timeout=0.1,
            transport=httpx.MockTransport(slow),
        )
        result = await client.complete([{"role": "user", "content": "hi"}])
        assert result is None  # timed out after retries

    @pytest.mark.asyncio
    async def test_llm_error_classes_exist(self):
        from hypermem.llm import (
            LLMError, LLMTimeoutError, LLMRateLimitError,
        )
        assert issubclass(LLMTimeoutError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)