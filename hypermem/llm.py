"""HyperMEM - LLM client (Ollama, OpenAI, Anthropic, OpenAI-compatible).

Provides a unified async interface for multiple LLM providers with
provider auto-detection, environment-variable API keys, timeout, retry,
and shared HTTP client lifecycle.
"""

import json
import asyncio
import os
import re
import logging
from typing import Optional
import httpx

logger = logging.getLogger("hypermem")


DEFAULT_TIMEOUT = 30
MAX_RETRIES = 2
RETRY_DELAY = 1.0

DEFAULT_ENDPOINTS = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Models that reveal the provider when no explicit provider is set.
_MODEL_PROVIDERS = [
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
]


def infer_provider(provider: Optional[str], model: Optional[str],
                   endpoint: Optional[str]) -> str:
    """Determine the provider from an explicit value, then model, then endpoint.

    provider="auto" (or empty) enables detection.
    """
    if provider and provider not in ("", "auto"):
        return provider

    m = (model or "").lower()
    for prefix, prov in _MODEL_PROVIDERS:
        if m.startswith(prefix):
            return prov

    e = (endpoint or "").lower()
    if "anthropic" in e:
        return "anthropic"
    if "openai" in e or "/v1" in e:
        return "openai"
    return "ollama"


def _chat_url(provider: str, endpoint: str) -> str:
    """Full chat-completion URL for a provider and base endpoint.

    Handles base URLs with or without the version suffix, and bare
    ``/chat/completions`` / ``/messages`` endpoints (some gateways expose
    the full path directly).
    """
    e = endpoint.rstrip("/")
    if provider == "openai":
        if e.endswith("/chat/completions"):
            return e
        if e.endswith("/v1"):
            return f"{e}/chat/completions"
        return f"{e}/v1/chat/completions"
    if provider == "anthropic":
        if e.endswith("/messages"):
            return e
        if e.endswith("/v1"):
            return f"{e}/messages"
        return f"{e}/v1/messages"
    return f"{e}/api/chat"


class LLMError(Exception):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` (or ``` ... ```) markdown fences."""
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    return cleaned


def _balanced_blocks(text: str) -> list[str]:
    """Return the text of each top-level {...} block found in ``text``."""
    blocks = []
    i = 0
    while True:
        start = text.find("{", i)
        if start < 0:
            break
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blocks.append(text[start:j + 1])
                        i = j + 1
                        break
        else:
            i = start + 1
    return blocks


def _lenient_load(text: str):
    """Parse an LLM JSON-ish object, tolerating Python-style quirks.

    Tries, in order: strict json, then trailing-comma/True/None fixups,
    then ast.literal_eval (handles single quotes, None/True/False, etc.).
    """
    import ast
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    fixed = _fix_common_json_quirks(text)
    if fixed is not None:
        try:
            return json.loads(fixed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        return None


def _fix_common_json_quirks(text: str) -> Optional[str]:
    """Patch trailing commas and Python True/False/None into strict JSON."""
    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    return fixed


def extract_json_object(raw: str) -> Optional[dict]:
    """Best-effort parse of a JSON *object* from an LLM response.

    Handles markdown fences, prose wrapped around the JSON, single-quoted
    keys/values, trailing commas and Python-style literals — the failure
    modes that trip up smaller / non-instruction-tuned models.
    """
    cleaned = _strip_fences(raw or "")
    for candidate in [cleaned] + _balanced_blocks(cleaned):
        if not candidate.strip():
            continue
        parsed = _lenient_load(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_json_list(raw: str) -> Optional[list]:
    """Best-effort parse of a JSON *array* from an LLM response."""
    cleaned = _strip_fences(raw or "")
    for candidate in [cleaned] + _balanced_blocks(cleaned):
        if not candidate.strip():
            continue
        parsed = _lenient_load(candidate)
        if isinstance(parsed, list):
            return parsed
    return None


def _extract_indices(raw: str) -> list[int]:
    """Parse 1-based memory indices out of an LLM recall response.

    Accepts JSON arrays ([3, 1]), JSON objects ({"indices": [3]}), fenced
    JSON, bare numbers, and comma-separated lists ("3, 1").
    """
    cleaned = raw.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            nums = [int(i) for i in parsed if str(i).strip().lstrip("-").isdigit()]
        elif isinstance(parsed, dict):
            nums = [int(i) for i in parsed.get("indices", [])]
        else:
            nums = []
        if nums:
            return [n - 1 for n in nums if n > 0]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    nums = [int(n) for n in re.findall(r"\d+", cleaned)]
    return [n - 1 for n in nums if n > 0]


class LLMClient:
    """Unified async client for multiple LLM providers.

    Args:
        provider: "ollama", "openai", "anthropic", or "auto" (detect from the
            model name/endpoint). Also accepts OpenAI-compatible base URLs
            (LM Studio, vLLM, Groq, OpenRouter, ...) via provider="openai".
        model: Model name (e.g. "qwen2.5:7b", "gpt-4o-mini", "claude-3-5-sonnet"). 
        endpoint: API base URL. Sensible per-provider defaults are used when
            the default Ollama endpoint was left untouched.
        api_key: API key; falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY.
        timeout: Request timeout in seconds.
        transport: Optional injected HTTP transport (proxies/tests).
    """

    def __init__(self, provider: str = "auto", model: str = "qwen2.5:7b",
                 endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 transport=None):
        self.provider = infer_provider(provider, model, endpoint)
        if self.provider not in ("ollama", "openai", "anthropic"):
            raise ValueError(
                f"Unsupported LLM provider {self.provider!r}. Use one of "
                "'ollama', 'openai' (also covers OpenAI-compatible endpoints "
                "like LM Studio, vLLM, Groq, OpenRouter), or 'anthropic'. "
                "Set provider='auto' to detect it from the model/endpoint."
            )
        # Use a provider-default endpoint when none was given or the caller
        # still has the Ollama default while switching providers.
        if not endpoint:
            endpoint = DEFAULT_ENDPOINTS[self.provider]
        elif (self.provider != "ollama"
                and endpoint.rstrip("/") == DEFAULT_ENDPOINTS["ollama"]):
            endpoint = DEFAULT_ENDPOINTS[self.provider]
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get(API_KEY_ENV.get(self.provider, ""))
        self.timeout = timeout
        self._transport = transport  # optional injected HTTP transport (proxy/tests)
        self._client: Optional[httpx.AsyncClient] = None
        self._chat_url = _chat_url(self.provider, self.endpoint)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout, transport=self._transport)
        return self._client

    async def close(self):
        """Close the shared HTTP client."""
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def complete(self, messages: list[dict], temperature: float = 0.1,
                       max_tokens: int = 100,
                       model: Optional[str] = None) -> Optional[str]:
        """Send a chat completion request.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            model: Override the model name for this call (optional).

        Returns:
            Response text, or None on failure after retries.
        """
        use_model = model or self.model
        last_error = None

        for attempt in range(1 + MAX_RETRIES):
            try:
                client = await self._get_client()

                if self.provider == "ollama":
                    resp = await client.post(self._chat_url, json={
                        "model": use_model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "num_batch": 2048,
                        },
                    })
                    if resp.status_code == 429:
                        raise LLMRateLimitError("Rate limited")
                    if resp.is_error:
                        last_error = f"HTTP {resp.status_code}"
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    data = resp.json()
                    return data.get("message", {}).get("content")

                elif self.provider == "openai":
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    resp = await client.post(
                        self._chat_url,
                        headers=headers,
                        json={
                            "model": use_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    if resp.status_code == 429:
                        raise LLMRateLimitError("Rate limited")
                    if resp.is_error:
                        last_error = f"HTTP {resp.status_code}"
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    data = resp.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content")

                elif self.provider == "anthropic":
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key or "",
                        "anthropic-version": "2023-06-01",
                    }
                    resp = await client.post(
                        self._chat_url,
                        headers=headers,
                        json={
                            "model": use_model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )
                    if resp.status_code == 429:
                        raise LLMRateLimitError("Rate limited")
                    if resp.is_error:
                        last_error = f"HTTP {resp.status_code}"
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    data = resp.json()
                    return data.get("content", [{}])[0].get("text")

            except httpx.TimeoutException:
                last_error = "timeout"
                await asyncio.sleep(RETRY_DELAY)
                continue
            except LLMRateLimitError:
                last_error = "rate_limit"
                await asyncio.sleep(RETRY_DELAY * 2)
                continue
            except Exception as e:
                last_error = str(e)
                await asyncio.sleep(RETRY_DELAY)
                continue

        return None

    # ---- Convenience methods (prompt-level, not transport-level) ----
    # These live on the engine instead — they're about WHAT to ask,
    # not HOW to ask it. Kept here for backward compatibility.

    async def judge(self, message_content: str) -> Optional[dict]:
        """Ask LLM to judge importance and extract memory."""
        prompt = f"""Message: "{message_content}"

Is there a specific fact to remember? If yes, quote the exact detail. If no, importance=0.

Return JSON. Example: {{"memory":"User name is Emanuel","keywords":["emanuel","name"],"importance":0.7}}

ENGLISH ONLY."""

        result = await self.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=100,
        )
        if not result:
            return None

        try:
            cleaned = result.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return None

    async def find_relevant(self, message: str, memories: list,
                            max_results: int = 3, raw: bool = False) -> list[int]:
        """Ask LLM which memories are relevant to the current message.

        Returns 0-based indices into ``memories``, best first. Falls back to
        keyword overlap when the LLM returns no usable answer. With ``raw``,
        only the LLM's own picks are returned (no keyword-overlap ordering)
        — used when the caller needs the model's semantic judgment alone.
        """
        if not memories:
            return []

        memories_list = "\n".join(
            f"{i + 1}. \"{m.content}\""
            for i, m in enumerate(memories)
        )
        prompt = f"""Given these memories and a question, find memories relevant to the question.

Memories:
{memories_list}

Question: "{message}"

Rules:
- A memory is relevant if it contains a fact the question asks about, even if the question uses different wording.
- Identity questions ("What is my name?", "Who am I?") are answered by the memory containing the user's name.
- Questions about the user's world, plans, or history are answered by the memory holding that fact.
- Prefer the fact-carrying memory over a similar-sounding one.
- Return the up to {max_results} most relevant memory indices, most relevant first, as a JSON array of numbers.

Return ONLY the JSON array. Example: [3, 1]
If no memory is relevant, return []."""

        result = await self.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=50,
        )
        if result:
            logger.debug("recall raw response: %r", result)
            indices = _extract_indices(result)
            if indices:
                indices = indices[:max_results]
                if raw:
                    return indices
                # Lexical evidence first (deterministic), then LLM picks
                # (deduplicated). A small model that ranks the wrong memory
                # cannot hide a fact the question literally names.
                ordered = list(LLMClient._keyword_fallback(message, memories))
                for i in indices:
                    if i not in ordered:
                        ordered.append(i)
                return ordered[:max_results]

        if raw:
            return []
        return LLMClient._keyword_fallback(message, memories)

    @staticmethod
    def _keyword_fallback(message: str, memories: list) -> list[int]:
        """Deterministic overlap ranking when the LLM gives no answer."""
        q = set(re.findall(r"[a-z0-9]{3,}", message.lower()))
        if not q:
            return []
        # "What's my name?" / "Who am I?" — the user is the subject of the
        # name question. "What's my bow called?", "my sister's name", and
        # "the king's name" are about a thing or another person, never the
        # user, so they are not identity queries.
        identity_query = bool(re.search(
            r"\bmy name\b|\bwho am i\b|\bwhat (?:am i|'m i|are we) called\b"
            r"|\bwhat('s| is) my name\b",
            message, re.I,
        ))
        scored = []
        for i, mem in enumerate(memories):
            haystack = set(re.findall(r"[a-z0-9]{3,}", mem.content.lower()))
            kwset = {kw.lower() for kw in (mem.keywords or [])}
            haystack |= kwset
            overlap = len(q & haystack)
            if overlap > 0:
                # Identity questions ("What's my name?") must surface the
                # user-identity memory (tagged with the exact "name" keyword),
                # never a lookalike ("true name of ...").
                if identity_query and ("name" in kwset or "identity" in kwset):
                    overlap += 5
                scored.append((overlap, -mem.importance, -i, i))
        scored.sort(reverse=True)
        return [i for *_, i in scored]
