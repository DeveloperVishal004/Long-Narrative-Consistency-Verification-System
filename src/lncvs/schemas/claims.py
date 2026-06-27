"""Claim decomposition and question generation contracts."""

from pydantic import BaseModel, ConfigDict, Field


class AtomicClaim(BaseModel):
    """A single, indivisible factual assertion derived from the original claim.

    Example: "John played a two-handed piano piece in London" decomposes into
    atomic claims "John played piano", "John used both hands", and "the event
    occurred in London".
    """

    model_config = ConfigDict(frozen=True)

    claim_id: str = Field(..., min_length=1, description="Unique identifier for this atomic claim.")
    text: str = Field(..., min_length=1, description="The atomic claim's text.")
    parent_claim_id: str | None = Field(
        default=None,
        description="ID of the original (pre-decomposition) claim this atomic claim was derived from.",
    )
    index: int = Field(
        default=0,
        ge=0,
        description="0-indexed position of this atomic claim within its parent's decomposition output, "
        "for stable ordering and traceability.",
    )


class ProbeQuestion(BaseModel):
    """A retrieval-oriented question generated to increase evidence coverage for an atomic claim.

    Example: for the atomic claim "John used both hands", a probe question
    might be "Did John lose an arm?" — designed to surface evidence that is
    not semantically similar to the claim itself but would contradict it.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(..., min_length=1, description="Unique identifier for this question.")
    atomic_claim_id: str = Field(
        ..., min_length=1, description="ID of the AtomicClaim this question was generated for."
    )
    text: str = Field(..., min_length=1, description="The generated question's text.")
    index: int = Field(
        default=0,
        ge=0,
        description="0-indexed position of this question within its claim's generated question list, "
        "for stable ordering and traceability.",
    )
