"""Shared test fixture: a stubbed Ollama HTTP API.

Responds to ``POST /api/chat`` exactly like a well-behaved Ollama instance,
so the *real* LLMClient (prompt building, JSON parsing, retries) and the
whole engine pipeline run against it — no live model needed in CI.

Limited to the prompt shapes HyperMEM itself generates:
- judge prompts        -> {"has_fact", "importance", "memory_type", "subject", "keywords"}
- recall prompts       -> "[n, ...]" index list (or "[]")
- worldIDA prompts     -> scene/user/character/relationship/meta JSON
- /api/embeddings      -> deterministic bag-of-words vector
"""

import json
import re
import zlib
import httpx
import pytest

from hypermem.llm import LLMClient

JUDGE_MARKERS = [
    "Extract the key factual information",
    "Is there a specific fact to remember",
    "Decide whether this message contains a fact worth remembering",
]
RECALL_MARKERS = [
    "find memories relevant to the question",
]
CONSOLIDATE_MARKERS = [
    "Summarize these episodic memories",
]
IDA_MARKERS = [
    "tracking the current state of a roleplay scene",
]

_STOPWORDS = {
    "the", "this", "that", "with", "from", "have", "been", "were",
    "what", "when", "where", "which", "their", "there", "about",
    "would", "could", "should", "after", "before", "into", "over",
    "only", "other", "than", "then", "very", "just", "also", "more",
    "some", "such", "like", "and", "are", "for", "not", "you", "your",
    "find", "number", "memory", "memories", "question", "answers",
    "contains", "because", "please", "key", "facts", "fact",
}


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


class OllamaStub:
    """Drop-in stand-in for the Ollama /api/chat endpoint."""

    def __init__(self, importance: float = 0.8,
                 memory_type: str = "episodic",
                 recall_response=None,
                 ida_response=None):
        """
        Args:
            importance: importance the "model" assigns to judged memories.
            memory_type: memory_type the "model" assigns ("episodic" default
                keeps event-like facts coexisting; pass "static" to exercise
                supersession).
            recall_response: optional callable(question, memories) -> str
                overriding the default keyword-overlap recall.
            ida_response: optional callable(previous_state_json) -> dict
                overriding the default pass-through worldIDA response.
        """
        self.importance = importance
        self.memory_type = memory_type
        self.recall_response = recall_response
        self.ida_response = ida_response
        self.calls: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.calls.append(payload)
        # Embeddings endpoint (POST /api/embeddings with {"prompt": ...}).
        if request.url.path == "/api/embeddings":
            return self._embed(payload)
        prompt = payload["messages"][-1]["content"]
        content = self._respond(prompt)
        return httpx.Response(200, json={
            "model": payload.get("model", "qwen2.5:7b"),
            "message": {"role": "assistant", "content": content},
        })

    # ---- dispatch ----

    def _respond(self, prompt: str) -> str:
        if any(m in prompt for m in JUDGE_MARKERS):
            return self._judge(prompt)
        if any(m in prompt for m in RECALL_MARKERS):
            return self._recall(prompt)
        if any(m in prompt for m in CONSOLIDATE_MARKERS):
            return self._consolidate(prompt)
        if any(m in prompt for m in IDA_MARKERS):
            return self._ida(prompt)
        return "ok"

    # ---- behaviors ----

    def _judge(self, prompt: str) -> str:
        match = re.search(r'Message:\s*"((?:[^"\\]|\\.)*)"', prompt, re.S)
        memory = (match.group(1) if match else "").replace('\\"', '"').strip()
        if not memory:
            return json.dumps({"has_fact": False, "importance": 0,
                               "memory_type": self.memory_type, "subject": "",
                               "keywords": []})
        keywords = sorted(_tokens(memory) - _STOPWORDS)[:8]
        subject = keywords[0] if keywords else "user"
        return json.dumps({
            "has_fact": True,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "subject": subject,
            "keywords": keywords,
        })

    def _embed(self, payload: dict) -> httpx.Response:
        prompt = payload.get("prompt", "")
        return httpx.Response(200, json={"embedding": self._hash_vec(prompt)})

    @staticmethod
    def _hash_vec(text: str) -> list:
        """Deterministic bag-of-words vector: the same token set yields the
        same vector, so cosine similarity behaves like a rough semantic
        overlap."""
        vec = [0.0] * 64
        for tok in _tokens(text):
            vec[zlib.crc32(tok.encode("utf-8")) % 64] += 1.0
        return vec

    def _recall(self, prompt: str) -> str:
        q_match = re.search(r'Question:\s*"((?:[^"\\]|\\.)*)"', prompt, re.S)
        question = (q_match.group(1) if q_match else "").replace('\\"', '"')
        mems = [m.replace('\\"', '"')
                for m in re.findall(r'\d+\.\s*"((?:[^"\\]|\\.)*)"', prompt, re.S)]

        if self.recall_response:
            return self.recall_response(question, mems)

        q = _tokens(question) - _STOPWORDS
        if not q or not mems:
            return "[]"
        best_idx, best_score = -1, 0
        for i, m in enumerate(mems):
            score = len(q & _tokens(m))
            if score > best_score:
                best_score, best_idx = score, i
        return f"[{best_idx + 1}]" if best_idx >= 0 else "[]"

    def _consolidate(self, prompt: str) -> str:
        """Return a 'summary' that preserves the source memories verbatim
        (joined), so hermetic tests can assert on the fused knowledge memory."""
        block = prompt.split("Memories:")[1].split("Rules:")[0].strip()
        lines = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if ". " in line:
                line = line.split(". ", 1)[1]
            lines.append(line)
        return json.dumps({"summary": " ".join(lines)})

    def _ida(self, prompt: str) -> str:
        prev_match = re.search(r"Previous state: (.+?)\nPersona:", prompt, re.S)
        prev = None
        if prev_match:
            try:
                prev = json.loads(prev_match.group(1).strip())
            except json.JSONDecodeError:
                prev = None

        if self.ida_response:
            return json.dumps(self.ida_response(prev))

        if prev is None:
            prev = {}
        meta = prev.get("meta", {})
        return json.dumps({
            "scene": prev.get("scene", {}),
            "user": prev.get("user", {}),
            "character": prev.get("character", {}),
            "relationship": prev.get("relationship", {}),
            "meta": {
                "scene_changed": False,
                "turn_count_in_scene": int(meta.get("turn_count_in_scene", 0)) + 1,
                "last_updated_turn_index": 0,
                "confidence": 1.0,
            },
        })


def make_llm(**stub_kwargs) -> tuple[LLMClient, OllamaStub]:
    """Build a real LLMClient wired to an OllamaStub transport."""
    stub = OllamaStub(**stub_kwargs)
    client = LLMClient(
        provider="ollama",
        model="qwen2.5:7b",
        endpoint="http://stub.local:11434",
        transport=httpx.MockTransport(stub),
    )
    return client, stub


@pytest.fixture
def ollama_stub():
    """Factory fixture: call ollama_stub(**kwargs) to get (client, stub)."""
    return make_llm