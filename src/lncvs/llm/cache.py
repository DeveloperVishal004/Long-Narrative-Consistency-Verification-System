"""LLM completion cache: the LLMCache protocol and a caching LLMClient decorator.

Structurally mirrors lncvs.indexing.cache (EmbeddingCache / InMemoryEmbeddingCache /
CachingEmbedder). Per CLAUDE.md's determinism mandate, model calls must be
cached by input hash. CachingLLMClient satisfies the existing LLMClient
protocol, so it drops in by injection with no change to its consumers.

JsonlLLMCache (Phase H1) is the LLMClient-side counterpart to
lncvs.llm.structured_cache.JsonlStructuredLLMCache -- until now, only the
StructuredLLMClient protocol (used by graph extraction) had a file-backed
cache; LLMClient (used by claim decomposition) had only InMemoryLLMCache,
which cannot satisfy "never recompute cached work" across process runs.
Mirrors JsonlStructuredLLMCache's exact persistence discipline (one JSON
line per entry, loaded eagerly, appended immediately on every miss) with
no schema_version field, since LLMClient has no structured-output schema
to version.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from lncvs.llm.base import LLMClient, LLMCompletion
from lncvs.llm.config import LLMConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMCache(Protocol):
    """Dependency-injection point for completion storage.

    In-memory now; swappable for a persistent store later without changing
    CachingLLMClient or any of its callers.
    """

    def get(self, key: str) -> LLMCompletion | None:
        """Return the cached completion for key, or None if not present."""
        ...

    def put(self, key: str, completion: LLMCompletion) -> None:
        """Store completion under key."""
        ...


class InMemoryLLMCache:
    """A simple dict-backed LLMCache. No persistence across process restarts."""

    def __init__(self) -> None:
        self._store: dict[str, LLMCompletion] = {}

    def get(self, key: str) -> LLMCompletion | None:
        return self._store.get(key)

    def put(self, key: str, completion: LLMCompletion) -> None:
        self._store[key] = completion


class JsonlLLMCache:
    """File-backed LLMCache: one JSON line per entry.

    Existing entries are loaded fully into memory at construction; every
    cache miss is appended to the file immediately (not batched), so a
    crash partway through a decomposition run never loses already-paid-for
    completions. Identical persistence discipline to
    lncvs.llm.structured_cache.JsonlStructuredLLMCache.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._store: dict[str, LLMCompletion] = {}

        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._store[record["key"]] = LLMCompletion.model_validate(record["completion"])
            logger.info("Loaded %d cached LLM completions from %s", len(self._store), path)

    def get(self, key: str) -> LLMCompletion | None:
        return self._store.get(key)

    def put(self, key: str, completion: LLMCompletion) -> None:
        self._store[key] = completion
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "completion": completion.model_dump()}) + "\n")


class CachingLLMClient:
    """An LLMClient that caches completions by (config fingerprint, prompt).

    The config fingerprint namespaces every cache key: a different model,
    temperature, or max_tokens can never read back a completion produced
    under a different configuration, even when sharing the same cache
    instance. A prompt-template edit changes the rendered prompt text and
    therefore the cache key automatically — no separate prompt-version
    plumbing is needed for cache correctness.
    """

    def __init__(self, client: LLMClient, cache: LLMCache, config: LLMConfig) -> None:
        self._client = client
        self._cache = cache
        self._fingerprint = config.fingerprint()

    def complete(self, prompt: str) -> LLMCompletion:
        key = self._cache_key(prompt)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("LLM cache hit for prompt of length %d", len(prompt))
            return cached

        completion = self._client.complete(prompt)
        self._cache.put(key, completion)
        return completion

    def _cache_key(self, prompt: str) -> str:
        digest_input = f"{self._fingerprint}:{prompt}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()
