"""Deterministic, offline test double for the Embedder protocol.

Uses SHA-256-hashed bag-of-words buckets rather than a real embedding
model, so unit tests are fast, deterministic across processes, and require
no network access or model download. It is "semantic-ish" only in that
texts sharing more words produce vectors with higher cosine similarity —
sufficient to exercise indexing/retrieval plumbing, not a substitute for
the real-model acceptance test.
"""

import hashlib
import math
import re

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _bucket(word: str, dim: int) -> int:
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % dim


def _tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


class FakeEmbedder:
    """A small, deterministic bag-of-words embedder satisfying the Embedder protocol."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in _tokenize(text):
            vector[_bucket(word, self._dim)] += 1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm > 0:
            vector = [component / norm for component in vector]
        return vector
