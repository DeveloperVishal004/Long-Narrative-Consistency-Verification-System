"""GraphState — the LangGraph state object, split into domain and control state.

GraphState is never a flat bag of fields. ledger holds domain/audit state
(the EvidenceLedger); control holds orchestration-only concerns. The rule
engine must only ever read GraphState.ledger, never GraphState.control.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from lncvs.schemas.enums import PipelineStage, RetrievalSource
from lncvs.schemas.ledger import EvidenceLedger


class StageError(BaseModel):
    """A single error encountered while executing a pipeline stage."""

    stage: PipelineStage = Field(..., description="Stage in which the error occurred.")
    message: str = Field(..., min_length=1, description="Human-readable error message.")
    timestamp: datetime = Field(..., description="When the error occurred.")


class ControlState(BaseModel):
    """Orchestration-only state: never read by the rule engine, never part of the audit trail."""

    current_stage: PipelineStage = Field(..., description="The pipeline stage currently executing.")
    errors: list[StageError] = Field(default_factory=list, description="Errors encountered so far.")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted for the current stage.")
    degraded_sources: list[RetrievalSource] = Field(
        default_factory=list, description="Retrieval backends that failed but were tolerated."
    )
    config_fingerprint: str = Field(
        ..., min_length=1, description="Hash of the run's configuration and model versions, for reproducibility."
    )


class GraphState(BaseModel):
    """Top-level LangGraph state: domain state (ledger) and control state, kept separate."""

    ledger: EvidenceLedger = Field(..., description="Domain state — the single source of truth.")
    control: ControlState = Field(..., description="Orchestration state — not part of the audit trail.")
