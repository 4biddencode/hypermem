"""Tests for multi-provider integration (OpenAI / Anthropic / compatible)."""

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

from hypermem.llm import (
    LLMClient, infer_provider, _chat_url,
    DEFAULT_ENDPOINTS,
)


def capture_handler(requests_log: list):
    """Return a transport handler that records requests and answers OpenAI-style."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests_log.append(request)
        if request.url.host == "api.anthropic.com":
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "hello from claude"}],
            })
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hello from gpt"}}],
        })

    return handler


# ---- Provider inference ----

class TestInferProvider:
    def test_explicit_wins(self):
        assert infer_provider("ollama", "claude-3-5-sonnet", "https://api.anthropic.com") == "ollama"

    def test_auto_from_model(self):
        assert infer_provider("auto", "claude-3-5-sonnet-20241022", None) == "anthropic"
        assert infer_provider("auto", "gpt-4o-mini", None) == "openai"
        assert infer_provider("auto", "qwen2.5:7b", None) == "ollama"

    def test_auto_from_endpoint(self):
        assert infer_provider("auto", "random-model", "https://api.anthropic.com") == "anthropic"
        assert infer_provider("auto", "random-model", "http://localhost:1234/v1") == "openai"
        assert infer_provider("auto", "random-model", "https://api.openai.com") == "openai"

    def test_auto_empty(self):
        assert infer_provider("", None, None) == "ollama"
        assert infer_provider(None, None, None) == "ollama"


# ---- Chat URL construction ----

class TestChatUrl:
    def test_ollama(self):
        assert _chat_url("ollama", "http://localhost:11434") == "http://localhost:11434/api/chat"

    def test_openai_default(self):
        assert _chat_url("openai", "https://api.openai.com") == "https://api.openai.com/v1/chat/completions"

    def test_openai_endpoint_with_v1(self):
        assert _chat_url("openai", "http://localhost:1234/v1") == "http://localhost:1234/v1/chat/completions"

    def test_openai_bare_full_path(self):
        assert _chat_url("openai", "https://openrouter.ai/api/v1/chat/completions") == \
            "https://openrouter.ai/api/v1/chat/completions"

    def test_anthropic_default(self):
        assert _chat_url("anthropic", "https://api.anthropic.com") == "https://api.anthropic.com/v1/messages"

    def test_anthropic_with_v1(self):
        assert _chat_url("anthropic", "https://x.example.com/v1") == "https://x.example.com/v1/messages"


# ---- Constructor behavior ----

class TestConstructor:
    def test_endpoint_swapped_when_provider_changes(self):
        client = LLMClient(provider="openai", model="gpt-4o-mini",
                           endpoint="http://localhost:11434")
        assert client.endpoint == DEFAULT_ENDPOINTS["openai"]

    def test_explicit_endpoint_kept(self):
        client = LLMClient(provider="openai", model="gpt-4o-mini",
                           endpoint="http://localhost:1234/v1")
        assert client.endpoint == "http://localhost:1234/v1"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            LLMClient(provider="make-believe")

    def test_auto_detects_anthropic(self):
        client = LLMClient(provider="auto", model="claude-3-5-sonnet")
        assert client.provider == "anthropic"
        assert client.endpoint == DEFAULT_ENDPOINTS["anthropic"]

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        client = LLMClient(provider="openai", model="gpt-4o-mini",
                           endpoint="https://api.openai.com", api_key=None)
        assert client.api_key == "sk-env-test"

    def test_api_key_env_depends_on_provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-env-test")
        client = LLMClient(provider="anthropic", model="claude-3-5-sonnet")
        assert client.api_key == "ak-env-test"


# ---- OpenAI request shape ----

class TestOpenAI:
    @pytest.mark.asyncio
    async def test_request_wire_format(self):
        log: list[httpx.Request] = []
        client = LLMClient(
            provider="openai", model="gpt-4o-mini",
            endpoint="https://api.openai.com", api_key="sk-test",
            transport=httpx.MockTransport(capture_handler(log)),
        )
        out = await client.complete(
            [{"role": "user", "content": "What's my name?"}],
            temperature=0.0, max_tokens=64,
        )
        assert out == "hello from gpt"

        req = log[0]
        assert req.url == "https://api.openai.com/v1/chat/completions"
        assert req.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(req.content)
        assert body["model"] == "gpt-4o-mini"
        assert body["max_tokens"] == 64
        assert body["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_openai_compatible_local_server(self):
        """LM Studio / vLLM style base URL works with provider=openai."""
        log: list[httpx.Request] = []
        client = LLMClient(
            provider="openai", model="qwen2.5-7b-instruct",
            endpoint="http://localhost:1234/v1", api_key="lm-studio",
            transport=httpx.MockTransport(capture_handler(log)),
        )
        out = await client.complete([{"role": "user", "content": "hi"}])
        assert out is not None
        assert str(log[0].url) == "http://localhost:1234/v1/chat/completions"


# ---- Anthropic request shape ----

class TestAnthropic:
    @pytest.mark.asyncio
    async def test_request_wire_format(self):
        log: list[httpx.Request] = []
        client = LLMClient(
            provider="anthropic", model="claude-3-5-sonnet",
            endpoint="https://api.anthropic.com", api_key="sk-ant-test",
            transport=httpx.MockTransport(capture_handler(log)),
        )
        out = await client.complete(
            [{"role": "user", "content": "What's my name?"}],
            temperature=0.2, max_tokens=128,
        )
        assert out == "hello from claude"

        req = log[0]
        assert req.url == "https://api.anthropic.com/v1/messages"
        assert req.headers["x-api-key"] == "sk-ant-test"
        assert req.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(req.content)
        assert body["model"] == "claude-3-5-sonnet"
        assert body["max_tokens"] == 128


# ---- Engine wiring ----

class TestEngineWiring:
    @pytest.mark.asyncio
    async def test_engine_works_with_openai_client(self):
        from hypermem import HyperMEM, HyperMemConfig
        from hypermem.engine import LLMClient as _  # noqa: F401

        log: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            log.append(request)
            prompt = json.loads(request.content)["messages"][-1]["content"]
            if "Decide whether this message contains a fact worth remembering" in prompt:
                return httpx.Response(200, json={"choices": [{
                    "message": {"content": json.dumps(
                        {"has_fact": True, "importance": 0.9,
                         "memory_type": "static", "subject": "user",
                         "keywords": ["berlin", "lives"]})}}]})
            if "find memories relevant" in prompt:
                return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = LLMClient(
            provider="openai", model="gpt-4o-mini", endpoint="https://api.openai.com",
            api_key="sk-test", transport=httpx.MockTransport(handler),
        )
        hm = HyperMEM(HyperMemConfig(llm_provider="openai", llm_model="gpt-4o-mini",
                                     llm_endpoint="https://api.openai.com",
                                     llm_api_key="sk-test"), llm=client)
        result = await hm.add_message("user", "I live in Vienna")
        assert result.tagged is not None
        assert "Vienna" in result.tagged.content

    @pytest.mark.asyncio
    async def test_anthropic_world_ida_via_engine(self):
        import asyncio
        from hypermem import HyperMEM, HyperMemConfig

        log: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            log.append(request)
            prompt = json.loads(request.content)["messages"][-1]["content"]
            if "roleplay scene" in prompt:
                return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps({
                    "scene": {"location": "tavern", "ongoing_action": "pouring ale"},
                    "user": {"physical_state": "standing in the doorway"},
                    "character": {"physical_state": "seated in the corner booth",
                                   "position": "far end of the room"},
                    "meta": {"scene_changed": False, "turn_count_in_scene": 1},
                })}]})
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

        client = LLMClient(
            provider="anthropic", model="claude-3-5-sonnet",
            endpoint="https://api.anthropic.com", api_key="sk-ant-test",
            transport=httpx.MockTransport(handler),
        )
        hm = HyperMEM(HyperMemConfig(llm_provider="anthropic", llm_model="claude-3-5-sonnet",
                                     llm_endpoint="https://api.anthropic.com",
                                     llm_api_key="sk-ant-test"), llm=client)
        await hm.update_world_ida("I push open the tavern door", "*She nods.*")
        ida = hm.get_world_ida()
        assert ida is not None
        assert ida.scene.location == "tavern"
        assert all("anthropic.com" in str(r.url) for r in log)