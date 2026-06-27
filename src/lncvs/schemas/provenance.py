"""Traceability primitives shared by every evidence-bearing model."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provenance(BaseModel):
    """Pointer from a piece of evidence back to its exact source location.

    Every NLI result, contradiction, and supporting-evidence record must
    ultimately trace back to a Provenance so a verdict can be audited to a
    specific chunk and character span — never to "the model said so".
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        ..., min_length=1, description="ID of the DocumentChunk this evidence originates from."
    )
    char_start: int = Field(
        ..., ge=0, description="Start offset of the evidence span within the source narrative."
    )
    char_end: int = Field(
        ..., ge=0, description="End offset of the evidence span within the source narrative."
    )

    @model_validator(mode="after")
    def _validate_span(self) -> "Provenance":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be strictly greater than char_start")
        return self
