"""RetrievalQuery — the unified query model bundling query text with full provenance."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lncvs.schemas.enums import QueryOrigin


class RetrievalQuery(BaseModel):
    """A single retrieval query, tied to the atomic claim it serves and, if
    applicable, the probe question that generated it.

    query_id is a deterministic content hash (see lncvs.retrieval.identity),
    never a uuid4 — the same discipline as every other ID in this system.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(..., min_length=1, description="Deterministic identifier for this query.")
    text: str = Field(..., min_length=1, description="The query text sent to a Retriever.")
    atomic_claim_id: str = Field(..., min_length=1, description="ID of the AtomicClaim this query serves.")
    question_id: str | None = Field(
        default=None,
        description="ID of the ProbeQuestion this query was generated from, if origin=QUESTION.",
    )
    origin: QueryOrigin = Field(..., description="Whether this query came from the claim text or a probe question.")

    @model_validator(mode="after")
    def _validate_origin_question_id_relationship(self) -> "RetrievalQuery":
        if self.origin is QueryOrigin.CLAIM and self.question_id is not None:
            raise ValueError("question_id must be None when origin is CLAIM")
        if self.origin is QueryOrigin.QUESTION and self.question_id is None:
            raise ValueError("question_id must be set when origin is QUESTION")
        return self
