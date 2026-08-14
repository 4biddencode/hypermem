"""EmbeddingClient: optional semantic recall.

One async interface for embedding text via Ollama's local embedding models
(``nomic-embed-text``) or OpenAI's embedding API. The provider is resolved
from config (or the LLM provider when set to "auto"). Every call degrades
gracefully: a failure marks the client unavailable and returns ``None``, so
a missing or misconfigured embedding model never breaks the engine — recall
falls back to the LLM+lexical path.
"""

import logging
import time
from typing import Optional
import httpx

logger = logging.getLogger("hypermem")

DEFAULT_EMBED_MODELS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
}

EMBED_TIMEOUT = 30.0  # generous: a cold Ollama model load can exceed 10s
EMBED_RETRY_SECONDS = 30.0  # backoff before retrying a transient embed failure


def _embedding_url(provider: str, endpoint: Optional[str]) -> Optional[str]:
    """Full embedding endpoint URL for a provider and base URL."""
    if provider not in ("ollama", "openai"):
        return None
    e = (endpoint or "").rstrip("/")
    if provider == "ollama":
        if e.endswith("/api/embeddings"):
            return e
        return f"{e or 'http://localhost:11434'}/api/embeddings"
    if e.endswith("/embeddings"):
        return e
    if e.endswith("/v1"):
        return f"{e}/embeddings"
    return f"{e or 'https://api.openai.com'}/v1/embeddings"


def resolve_embedding_provider(provider: str, llm_provider: Optional[str],
                               llm_endpoint: Optional[str]) -> str:
    """Map an explicit/auto embedding provider to a concrete one.

    ``auto`` follows the LLM provider: ollama and openai map to their own
    embedding APIs; anything else (anthropic has no embedding API) maps to
    "none".
    """
    if provider and provider != "auto":
        return provider
    if llm_provider == "ollama":
        return "ollama"
    if llm_provider == "openai":
        return "openai"
    e = (llm_endpoint or "").lower()
    if "openai" in e:
        return "openai"
    # A /v1 path is the OpenAI-compatible convention (vLLM, LM Studio, custom
    # gateways). Treat it as OpenAI so an OpenAI-compatible gateway at
    # http://gw:8000/v1 gets the OpenAI request schema instead of being
    # misrouted to Ollama's /api/embeddings.
    if e.startswith("http") and e.rstrip("/").endswith("/v1"):
        return "openai"
    if e.startswith("http"):
        return "ollama"
    return "none"


class EmbeddingClient:
    """Async client for embedding text via Ollama or OpenAI.

    Args:
        provider: "auto", "ollama", "openai", or "none".
        model: Embedding model name (defaults per provider).
        endpoint: Base URL override.
        api_key: OpenAI API key (falls back to OPENAI_API_KEY).
        llm_provider/llm_endpoint: used only when provider="auto" to infer.
        transport: optional injected HTTP transport (proxies/tests).
    """

    def __init__(self, provider: str = "auto", model: Optional[str] = None,
                 endpoint: Optional[str] = None, api_key: Optional[str] = None,
                 llm_provider: Optional[str] = None,
                 llm_endpoint: Optional[str] = None,
                 transport=None):
        self.provider = resolve_embedding_provider(provider, llm_provider, llm_endpoint)
        if self.provider not in ("ollama", "openai", "none"):
            self.provider = "none"
        self.model = model or DEFAULT_EMBED_MODELS.get(self.provider)
        self.endpoint = endpoint
        self.api_key = api_key
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None
        self._url = _embedding_url(self.provider, endpoint)
        self._available: Optional[bool] = None  # lazily probed on first call
        self._retry_at: float = 0.0  # monotonic ts after which transient errors retry

    @property
    def enabled(self) -> bool:
        return self.provider != "none" and self._url is not None

    @property
    def available(self) -> bool:
        """True when enabled and worth calling right now.

        Unprobed counts as available (the first call decides). A transient
        failure (timeout / 5xx / cold model load) parks the client for a
        short backoff window instead of disabling it forever — only a hard
        client error (e.g. the model isn't installed) latches it off.
        """
        if not self.enabled:
            return False
        if self._available is False:
            return False
        if self._retry_at and time.monotonic() < self._retry_at:
            return False
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=EMBED_TIMEOUT,
                                             transport=self._transport)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def embed(self, text: str) -> Optional[list]:
        """Embed text. Returns a list[float], or None on any failure.

        Once a call fails (e.g. the model isn't installed), the client is
        marked unavailable so subsequent calls short-circuit to None
        instead of hammering a dead endpoint.
        """
        if not self.enabled or not text:
            return None
        if self._available is False:
            return None
        if self._retry_at and time.monotonic() < self._retry_at:
            return None
        try:
            client = await self._get_client()
            if self.provider == "ollama":
                resp = await client.post(self._url, json={
                    "model": self.model, "prompt": text,
                })
                if resp.is_error:
                    return self._handle_error(resp.status_code)
                data = resp.json()
                vec = data.get("embedding")
                self._available = True
                self._retry_at = 0.0
                return vec if isinstance(vec, list) else None
            if self.provider == "openai":
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                resp = await client.post(self._url, headers=headers, json={
                    "model": self.model, "input": text,
                })
                if resp.is_error:
                    return self._handle_error(resp.status_code)
                data = resp.json()
                items = data.get("data", []) or []
                vec = items[0].get("embedding") if items else None
                self._available = True
                self._retry_at = 0.0
                return vec if isinstance(vec, list) else None
        except Exception as e:  # timeout, connection error, bad payload
            logger.debug("embedding failure (%s): %s", self.provider, e)
            return self._transient()
        return None

    def _handle_error(self, status: int) -> None:
        """4xx is permanent (bad config/missing model); 5xx is transient."""
        if 400 <= status < 500:
            return self._fail(status)
        return self._transient()

    def _transient(self) -> None:
        """A failure that may clear up (server restart, cold model load, 5xx).
        Park the client for a short backoff, then retry on the next call —
        a cold Ollama model load can take longer than the HTTP timeout."""
        logger.warning("embedding provider temporarily unavailable — semantic "
                       "recall paused, will retry shortly (LLM+lexical for now)")
        self._retry_at = time.monotonic() + EMBED_RETRY_SECONDS
        return None

    def _fail(self, status: int) -> None:
        logger.warning("embedding provider unavailable (HTTP %s) — semantic recall "
                       "disabled, falling back to LLM+lexical", status)
        self._available = False
        return None


# ---- Cosine similarity (pure Python — no numpy dependency) ----

def cosine(a: list, b: list) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    if dot <= 0:
        return 0.0
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
