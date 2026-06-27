"""Structured-completion cache: the StructuredLLMCache protocol and a caching
StructuredLLMClient decorator, plus a file-backed implementation.

Structurally mirrors lncvs.llm.cache exactly, with one necessary
extension: the cache key folds in schema_version as well as the config
fingerprint, because the same prompt against two different extraction
schema versions must never collide (frozen G2 interface §9).

JsonlStructuredLLMCache is the concrete mechanism behind the frozen
spec's "committed frozen extraction artifacts" (§6): one JSON line per
cached completion, loaded fully into memory at construction and appended
to on every cache miss. The file is meant to be committed to version
control, so the graph can be rebuilt forever from these artifacts with
zero further API calls -- the Tier-1 reproducibility guarantee.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from lncvs.llm.config import LLMConfig
from lncvs.llm.structured import StructuredCompletion, StructuredLLMClient

logger = logging.getLogger(__name__)


@runtime_checkable
class StructuredLLMCache(Protocol):
    """Dependency-injection point for structured-completion storage."""

    def get(self, key: str) -> StructuredCompletion | None:
        """Return the cached completion for key, or None if not present."""
        ...

    def put(self, key: str, completion: StructuredCompletion) -> None:
        """Store completion under key."""
        ...


class InMemoryStructuredLLMCache:
    """A simple dict-backed StructuredLLMCache. No persistence across process restarts."""

    def __init__(self) -> None:
        self._store: dict[str, StructuredCompletion] = {}

    def get(self, key: str) -> StructuredCompletion | None:
        return self._store.get(key)

    def put(self, key: str, completion: StructuredCompletion) -> None:
        self._store[key] = completion


class JsonlStructuredLLMCache:
    """File-backed StructuredLLMCache: one JSON line per entry.

    Existing entries are loaded fully into memory at construction (the
    artifact files this build produces are small -- a few hundred
    extraction windows -- so this is never a concern at this project's
    scale). Every cache miss is appended to the file immediately, not
    batched, so a crash partway through a build never loses already-paid-
    for completions.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._store: dict[str, StructuredCompletion] = {}

        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._store[record["key"]] = StructuredCompletion.model_validate(record["completion"])
            logger.info("Loaded %d cached structured completions from %s", len(self._store), path)

    def get(self, key: str) -> StructuredCompletion | None:
        return self._store.get(key)

    def put(self, key: str, completion: StructuredCompletion) -> None:
        self._store[key] = completion
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "completion": completion.model_dump()}) + "\n")


class CachingStructuredLLMClient:
    """A StructuredLLMClient that caches completions by
    (config fingerprint, schema_version, prompt).

    The config fingerprint and schema_version together namespace every
    cache key: a different model, sampling parameter, or schema version
    can never read back a completion produced under a different one, even
    when sharing the same cache instance.
    """

    def __init__(
        self, client: StructuredLLMClient, cache: StructuredLLMCache, config: LLMConfig, schema_version: str
    ) -> None:
        self._client = client
        self._cache = cache
        self._fingerprint = config.fingerprint()
        self._schema_version = schema_version

    def complete_structured(self, prompt: str, response_schema: dict) -> StructuredCompletion:
        key = self._cache_key(prompt)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("Structured LLM cache hit for prompt of length %d", len(prompt))
            return cached

        completion = self._client.complete_structured(prompt, response_schema)
        self._cache.put(key, completion)
        return completion

    def _cache_key(self, prompt: str) -> str:
        digest_input = f"{self._fingerprint}:{self._schema_version}:{prompt}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()
