"""Embedding cache: the EmbeddingCache protocol and a caching Embedder decorator.

Per CLAUDE.md's determinism mandate, model calls must be cached by input
hash. CachingEmbedder satisfies the existing Embedder protocol, so it drops
into ChromaIndex (or any Embedder consumer) by injection with no change to
that consumer's code or contract.
"""

import hashlib
import logging
from typing import Protocol, runtime_checkable

from lncvs.indexing.config import EmbeddingConfig
from lncvs.indexing.embedder import Embedder

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingCache(Protocol):
    """Dependency-injection point for embedding vector storage.

    In-memory now; swappable for a persistent store later without changing
    CachingEmbedder or any of its callers.
    """

    def get(self, key: str) -> list[float] | None:
        """Return the cached vector for key, or None if not present."""
        ...

    def put(self, key: str, vector: list[float]) -> None:
        """Store vector under key."""
        ...


class InMemoryEmbeddingCache:
    """A simple dict-backed EmbeddingCache. No persistence across process restarts."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def get(self, key: str) -> list[float] | None:
        return self._store.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self._store[key] = vector


class CachingEmbedder:
    """An Embedder that caches vectors by (config fingerprint, text) so identical
    text under an identical EmbeddingConfig is never re-embedded.

    The config fingerprint namespaces every cache key: a different model,
    device, or normalization setting can never read back a vector computed
    under a different configuration, even when sharing the same cache
    instance.
    """

    def __init__(self, embedder: Embedder, cache: EmbeddingCache, config: EmbeddingConfig) -> None:
        self._embedder = embedder
        self._cache = cache
        self._fingerprint = config.fingerprint()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")

        keys = [self._cache_key(text) for text in texts]
        cached: list[list[float] | None] = [self._cache.get(key) for key in keys]

        miss_indices = [i for i, vector in enumerate(cached) if vector is None]
        if miss_indices:
            miss_vectors = self._embedder.embed_texts([texts[i] for i in miss_indices])
            for i, vector in zip(miss_indices, miss_vectors):
                self._cache.put(keys[i], vector)
                cached[i] = vector

        logger.debug(
            "Embedding cache: %d hit(s), %d miss(es) out of %d text(s)",
            len(texts) - len(miss_indices),
            len(miss_indices),
            len(texts),
        )

        result: list[list[float]] = []
        for vector in cached:
            assert vector is not None  # every entry is filled by this point
            result.append(vector)
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _cache_key(self, text: str) -> str:
        digest_input = f"{self._fingerprint}:{text}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()
