"""EmbeddingCache / CachingEmbedder tests."""

import pytest

from lncvs.indexing import CachingEmbedder, Embedder, EmbeddingCache, EmbeddingConfig, InMemoryEmbeddingCache
from tests.indexing.fakes import FakeEmbedder


class _CountingEmbedder:
    """Wraps FakeEmbedder and counts how many times embed_texts is actually invoked."""

    def __init__(self) -> None:
        self._inner = FakeEmbedder()
        self.embed_texts_calls = 0
        self.embedded_texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embed_texts_calls += 1
        self.embedded_texts.extend(texts)
        return self._inner.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def _config(model_name: str = "fake-model-a") -> EmbeddingConfig:
    return EmbeddingConfig(model_name=model_name)


def test_in_memory_cache_round_trip() -> None:
    cache = InMemoryEmbeddingCache()
    assert cache.get("missing-key") is None

    cache.put("key", [1.0, 2.0, 3.0])
    assert cache.get("key") == [1.0, 2.0, 3.0]


def test_caching_embedder_satisfies_embedder_protocol() -> None:
    embedder = CachingEmbedder(_CountingEmbedder(), InMemoryEmbeddingCache(), _config())
    assert isinstance(embedder, Embedder)


def test_in_memory_cache_satisfies_embedding_cache_protocol() -> None:
    assert isinstance(InMemoryEmbeddingCache(), EmbeddingCache)


def test_caching_embedder_returns_identical_vectors_to_wrapped_embedder() -> None:
    inner = _CountingEmbedder()
    caching = CachingEmbedder(inner, InMemoryEmbeddingCache(), _config())

    direct = inner.embed_query("John lost his left arm in an accident.")
    cached = caching.embed_query("John lost his left arm in an accident.")

    assert direct == cached


def test_second_identical_call_is_served_from_cache() -> None:
    inner = _CountingEmbedder()
    caching = CachingEmbedder(inner, InMemoryEmbeddingCache(), _config())

    caching.embed_query("John lost his left arm in an accident.")
    caching.embed_query("John lost his left arm in an accident.")

    assert inner.embed_texts_calls == 1


def test_mixed_batch_recomputes_only_misses_and_preserves_order() -> None:
    inner = _CountingEmbedder()
    cache = InMemoryEmbeddingCache()
    caching = CachingEmbedder(inner, cache, _config())

    caching.embed_texts(["alpha", "beta"])
    inner.embed_texts_calls = 0
    inner.embedded_texts.clear()

    result = caching.embed_texts(["alpha", "gamma", "beta"])

    # Only the miss ("gamma") should have been sent to the wrapped embedder.
    assert inner.embedded_texts == ["gamma"]
    assert inner.embed_texts_calls == 1

    expected_alpha = caching.embed_query("alpha")
    expected_beta = caching.embed_query("beta")
    expected_gamma = caching.embed_query("gamma")
    assert result == [expected_alpha, expected_gamma, expected_beta]


def test_different_config_fingerprints_do_not_cross_serve_vectors() -> None:
    shared_cache = InMemoryEmbeddingCache()
    inner_a = _CountingEmbedder()
    inner_b = _CountingEmbedder()

    caching_a = CachingEmbedder(inner_a, shared_cache, _config("model-a"))
    caching_b = CachingEmbedder(inner_b, shared_cache, _config("model-b"))

    caching_a.embed_query("shared text")
    caching_b.embed_query("shared text")

    # Both embedders had to compute the vector themselves; neither served the
    # other's cached value, because their config fingerprints differ.
    assert inner_a.embed_texts_calls == 1
    assert inner_b.embed_texts_calls == 1


def test_caching_embedder_rejects_empty_batch() -> None:
    caching = CachingEmbedder(_CountingEmbedder(), InMemoryEmbeddingCache(), _config())
    with pytest.raises(ValueError):
        caching.embed_texts([])
