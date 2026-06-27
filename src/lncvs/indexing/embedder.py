"""Embedding generation: the Embedder protocol and its sentence-transformers implementation."""

import logging
from typing import Protocol, runtime_checkable

from lncvs.indexing.config import EmbeddingConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Dependency-injection point for embedding generation.

    Any implementation (real model, fake/deterministic test double) must
    satisfy this shape. ChromaIndex depends on this protocol, never on a
    concrete embedding library.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input in the same order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    The model is loaded once, eagerly, at construction time — there is no
    lazy-loading or caching beyond what sentence-transformers itself does.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self._config = config
        logger.info("Loading embedding model %r on device %r", config.model_name, config.device)
        self._model = SentenceTransformer(config.model_name, device=config.device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts")
        vectors = self._model.encode(
            texts,
            normalize_embeddings=self._config.normalize_embeddings,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
