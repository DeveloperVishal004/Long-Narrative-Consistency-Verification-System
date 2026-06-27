"""NLI verification contract.

Both premise and hypothesis are stored on every result so the entailment
direction is auditable after the fact — this is a direct mitigation for the
silently-reversible "premise/hypothesis swapped" bug class. The fixed
direction is: premise = evidence text, hypothesis = atomic claim text.
"""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas.enums import NLILabel


class NLIResult(BaseModel):
    """The result of running NLI on a single (atomic claim, evidence) pair."""

    model_config = ConfigDict(frozen=True)

    atomic_claim_id: str = Field(..., min_length=1, description="ID of the AtomicClaim being verified.")
    evidence_chunk_id: str = Field(..., min_length=1, description="ID of the DocumentChunk used as evidence.")
    label: NLILabel = Field(..., description="ENTAILMENT, CONTRADICTION, or NEUTRAL.")
    score: float = Field(..., ge=0.0, le=1.0, description="Model confidence for the assigned label.")
    premise: str = Field(..., min_length=1, description="The evidence text — fixed as the NLI premise.")
    hypothesis: str = Field(..., min_length=1, description="The atomic claim text — fixed as the NLI hypothesis.")
