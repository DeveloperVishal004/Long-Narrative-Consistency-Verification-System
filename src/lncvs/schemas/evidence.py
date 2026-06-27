"""Evidence contracts: per-source retrieval results, fused results, and NLI-derived evidence records."""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas.enums import RetrievalSource
from lncvs.schemas.provenance import Provenance


class RetrievedEvidence(BaseModel):
    """A single piece of evidence returned by one retrieval backend, pre-fusion.

    atomic_claim_id and query_id are Optional at construction time because
    the Retriever/Indexer that produce this record are claim-agnostic (see
    CLAUDE.md's retrieval orchestration section) — they cannot supply them.
    RetrievalOrchestrator stamps both fields (and re-derives evidence_id)
    before evidence is recorded in the ledger. LedgerService enforces that
    both are non-None at the ledger write boundary; the type itself does
    not enforce this, to keep Phase 1's retriever/indexer contracts unchanged.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(..., min_length=1, description="Unique identifier for this evidence record.")
    chunk_id: str = Field(..., min_length=1, description="ID of the DocumentChunk this evidence comes from.")
    text: str = Field(..., min_length=1, description="The evidence text.")
    source: RetrievalSource = Field(..., description="Which retrieval backend produced this result.")
    raw_score: float = Field(..., description="The backend's native relevance score (scale varies by source).")
    rank: int = Field(..., ge=1, description="1-indexed rank of this result within its source's ranked list.")
    provenance: Provenance = Field(..., description="Exact source location this evidence traces back to.")
    atomic_claim_id: str | None = Field(
        default=None, description="ID of the AtomicClaim this evidence was retrieved for. Stamped by the orchestrator."
    )
    query_id: str | None = Field(
        default=None, description="ID of the RetrievalQuery that produced this evidence. Stamped by the orchestrator."
    )


class FusedEvidence(BaseModel):
    """A deduplicated, claim-linked evidence record produced by Reciprocal Rank Fusion.

    Deliberately minimal: per-(query, source, rank) detail is NOT
    duplicated here. That detail lives in EvidenceLedger.retrieved_evidence,
    which remains the single source of truth for ranks — joining on
    chunk_id and atomic_claim_id reconstructs it fully. Denormalizing ranks
    onto this record would risk drift from that source of truth for no
    offsetting benefit (see CLAUDE.md's Fusion section for the full
    rationale on why source_ranks was considered and rejected).
    """

    model_config = ConfigDict(frozen=True)

    atomic_claim_id: str = Field(..., min_length=1, description="ID of the AtomicClaim this fused result supports.")
    chunk_id: str = Field(..., min_length=1, description="ID of the DocumentChunk this fused result represents.")
    text: str = Field(..., min_length=1, description="The evidence text.")
    rrf_score: float = Field(..., description="Reciprocal Rank Fusion score.")
    contributing_sources: list[RetrievalSource] = Field(
        ..., min_length=1, description="Which retrieval backends contributed to this fused result."
    )
    contributing_query_ids: list[str] = Field(
        ..., min_length=1, description="IDs of the RetrievalQuerys whose results contributed to this fused result."
    )


class SupportingEvidence(BaseModel):
    """A record linking an atomic claim to evidence that entails it."""

    model_config = ConfigDict(frozen=True)

    atomic_claim_id: str = Field(..., min_length=1, description="ID of the supported AtomicClaim.")
    evidence_chunk_id: str = Field(..., min_length=1, description="ID of the supporting DocumentChunk.")
    nli_score: float = Field(..., ge=0.0, le=1.0, description="NLI entailment confidence score.")
    explanation: str | None = Field(default=None, description="Optional human-readable rationale.")


class Contradiction(BaseModel):
    """A record linking an atomic claim to evidence that contradicts it."""

    model_config = ConfigDict(frozen=True)

    atomic_claim_id: str = Field(..., min_length=1, description="ID of the contradicted AtomicClaim.")
    evidence_chunk_id: str = Field(..., min_length=1, description="ID of the contradicting DocumentChunk.")
    nli_score: float = Field(..., ge=0.0, le=1.0, description="NLI contradiction confidence score.")
    explanation: str | None = Field(default=None, description="Optional human-readable rationale.")
