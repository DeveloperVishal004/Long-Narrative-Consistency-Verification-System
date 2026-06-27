"""Reciprocal Rank Fusion configuration."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class FusionConfig(BaseModel):
    """Configures a single fuse_evidence run.

    fingerprint() makes rrf_score reproducible and auditable: given the
    ledger's retrieved_evidence (the single source of truth for ranks) plus
    a recorded FusionConfig fingerprint, any FusedEvidence.rrf_score can be
    independently recomputed and verified — this is what replaces the
    audit value a denormalized source_ranks field would have offered, per
    CLAUDE.md's Fusion section.
    """

    model_config = ConfigDict(frozen=True)

    rrf_k: int = Field(default=60, gt=0, description="Reciprocal Rank Fusion's rank-damping constant.")
    top_k_fused: int = Field(
        default=10, gt=0, description="Maximum number of fused evidence records retained per atomic claim."
    )

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact configuration."""
        digest_input = f"{self.rrf_k}:{self.top_k_fused}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]
