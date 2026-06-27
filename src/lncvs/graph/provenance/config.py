"""Provenance assignment configuration."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class ProvenanceConfig(BaseModel):
    """Configures tiered quote resolution (lncvs.graph.provenance.matching).

    fingerprint() makes provenance resolution reproducible and auditable,
    the same role every other *Config.fingerprint() plays in this system:
    given the same window text and quotes plus a recorded config
    fingerprint, every QuoteMatch is independently recomputable.
    """

    model_config = ConfigDict(frozen=True)

    fuzzy_overlap_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Minimum token-overlap ratio for a Tier-2 fuzzy match to be accepted."
    )
    fuzzy_uniqueness_margin: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Minimum score gap required between the best and second-best Tier-2 candidate for the match to be accepted as unambiguous.",
    )

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact configuration."""
        digest_input = f"{self.fuzzy_overlap_threshold}:{self.fuzzy_uniqueness_margin}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]
