"""DocumentChunk — the unit of retrieval-friendly narrative text."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentChunk(BaseModel):
    """A contiguous, traceable span of the source narrative.

    chunk_id must be a deterministic content hash (not an incremental
    integer) so that re-chunking the same narrative reproduces identical
    IDs and so the same ID space can be shared across the Chroma and BM25
    indices.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        ..., min_length=1, description="Deterministic content-hash identifier for this chunk."
    )
    text: str = Field(..., min_length=1, description="The chunk's text content.")
    char_start: int = Field(
        ..., ge=0, description="Start offset of this chunk within the source narrative."
    )
    char_end: int = Field(
        ..., ge=0, description="End offset of this chunk within the source narrative."
    )
    chapter: str | None = Field(
        default=None, description="Chapter or section label, if the narrative has structure."
    )
    source_id: str = Field(
        ..., min_length=1, description="ID of the source narrative document this chunk belongs to."
    )

    @model_validator(mode="after")
    def _validate_span(self) -> "DocumentChunk":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be strictly greater than char_start")
        return self
