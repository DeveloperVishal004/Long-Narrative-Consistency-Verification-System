"""RawDocument — the typed output of ingestion, consumed by chunking."""

from pydantic import BaseModel, ConfigDict, Field


class RawDocument(BaseModel):
    """A loaded and cleaned narrative document, ready for chunking.

    cleaned_text is the text chunking operates over; character offsets on
    DocumentChunk are relative to cleaned_text, not raw_text. raw_text is
    retained for audit/debugging only — cleaning is not guaranteed to
    preserve a character-offset mapping back to it.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., min_length=1, description="Identifier for the source narrative document.")
    raw_text: str = Field(..., description="The original, unmodified text as loaded from disk.")
    cleaned_text: str = Field(..., min_length=1, description="Cleaned text that chunking operates over.")
