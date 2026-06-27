"""NLI prediction cache: the NLICache protocol and a caching NLIModel decorator.

Structurally mirrors lncvs.llm.cache (LLMCache / InMemoryLLMCache /
CachingLLMClient) and lncvs.indexing.cache (EmbeddingCache / CachingEmbedder).

Unlike an LLM call, a cross-encoder in eval mode with no sampling is already
deterministic on its own -- this cache's value is performance (avoiding
redundant inference on repeated (premise, hypothesis) pairs across runs),
not determinism. The determinism guarantee for NLI rests on the model
itself; this cache is the same house pattern applied for consistency and
throughput, not a correctness requirement.
"""

import hashlib
import logging
from typing import Protocol, runtime_checkable

from lncvs.reasoning.nli.config import NLIConfig
from lncvs.reasoning.nli.model import NLIModel, NLIPrediction

logger = logging.getLogger(__name__)


@runtime_checkable
class NLICache(Protocol):
    """Dependency-injection point for prediction storage.

    In-memory now; swappable for a persistent store later without changing
    CachingNLIModel or any of its callers.
    """

    def get(self, key: str) -> NLIPrediction | None:
        """Return the cached prediction for key, or None if not present."""
        ...

    def put(self, key: str, prediction: NLIPrediction) -> None:
        """Store prediction under key."""
        ...


class InMemoryNLICache:
    """A simple dict-backed NLICache. No persistence across process restarts."""

    def __init__(self) -> None:
        self._store: dict[str, NLIPrediction] = {}

    def get(self, key: str) -> NLIPrediction | None:
        return self._store.get(key)

    def put(self, key: str, prediction: NLIPrediction) -> None:
        self._store[key] = prediction


class CachingNLIModel:
    """An NLIModel that caches predictions by (config fingerprint, premise, hypothesis).

    The config fingerprint namespaces every cache key: a different model,
    max_length, or device can never read back a prediction produced under a
    different configuration, even when sharing the same cache instance.
    """

    def __init__(self, model: NLIModel, cache: NLICache, config: NLIConfig) -> None:
        self._model = model
        self._cache = cache
        self._fingerprint = config.fingerprint()

    def predict(self, premise: str, hypothesis: str) -> NLIPrediction:
        key = self._cache_key(premise, hypothesis)
        cached = self._cache.get(key)
        if cached is not None:
            logger.debug("NLI cache hit for premise of length %d", len(premise))
            return cached

        prediction = self._model.predict(premise, hypothesis)
        self._cache.put(key, prediction)
        return prediction

    def _cache_key(self, premise: str, hypothesis: str) -> str:
        digest_input = f"{self._fingerprint}:{premise}:{hypothesis}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()
