"""Embedding model configuration."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    """Configures which embedding model an Embedder implementation loads and how it runs.

    Kept as configuration rather than a hard-coded constant so the model can
    be swapped (e.g. for evaluation or future scaling) without code changes.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(
        ..., min_length=1, description="sentence-transformers model identifier, e.g. 'all-MiniLM-L6-v2'."
    )
    device: str = Field(default="cpu", description="Device to run embedding inference on.")
    normalize_embeddings: bool = Field(
        default=True, description="Whether to L2-normalize embedding vectors (required for cosine similarity)."
    )

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact configuration.

        Used to namespace cached embeddings so that a config change (e.g. a
        different model_name) can never silently serve a vector computed
        under a different configuration.
        """
        digest_input = f"{self.model_name}:{self.device}:{self.normalize_embeddings}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]
