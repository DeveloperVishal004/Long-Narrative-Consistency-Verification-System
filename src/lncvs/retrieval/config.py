"""Retrieval orchestration configuration."""

from pydantic import BaseModel, ConfigDict, Field


class RetrievalConfig(BaseModel):
    """Configures a RetrievalOrchestrator."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=5, gt=0, description="Number of results to retrieve per query.")
