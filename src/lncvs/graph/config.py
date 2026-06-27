"""Graph construction and retrieval configuration."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class GraphConfig(BaseModel):
    """Configures both graph construction (GraphIndex.index) and graph
    retrieval traversal (GraphIndex.query).

    fingerprint() makes graph-derived scores reproducible and auditable,
    the same role FusionConfig.fingerprint() and EmbeddingConfig.fingerprint()
    play for their respective stages: given the same chunks and the same
    config fingerprint, the entire graph and every retrieval score it
    produces must be independently recomputable.
    """

    model_config = ConfigDict(frozen=True)

    max_hops: int = Field(
        default=1, ge=1, le=2, description="Maximum BFS expansion depth from resolved entry entities."
    )
    min_entity_token_length: int = Field(
        default=2, gt=0, description="Minimum character length for a capitalized token to be treated as an entity."
    )

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact configuration."""
        digest_input = f"{self.max_hops}:{self.min_entity_token_length}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]
