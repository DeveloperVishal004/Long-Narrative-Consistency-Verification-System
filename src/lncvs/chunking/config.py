"""Chunking configuration."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Configurable parameters for sliding-window chunking."""

    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(..., gt=0, description="Maximum number of characters per chunk.")
    overlap: int = Field(..., ge=0, description="Number of characters of overlap between consecutive chunks.")

    @model_validator(mode="after")
    def _validate_overlap(self) -> "ChunkingConfig":
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size")
        return self
