"""Audit-trail contracts: per-stage reasoning steps and ledger events."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas.enums import PipelineStage


class ReasoningStep(BaseModel):
    """A single, human-readable step in the explainable reasoning trace."""

    model_config = ConfigDict(frozen=True)

    stage: PipelineStage = Field(..., description="Pipeline stage this step belongs to.")
    description: str = Field(..., min_length=1, description="Human-readable description of what happened.")
    timestamp: datetime = Field(..., description="When this step occurred.")


class LedgerEvent(BaseModel):
    """A single append-only audit log entry recorded against the EvidenceLedger."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(..., min_length=1, description="Unique identifier for this log entry.")
    stage: PipelineStage = Field(..., description="Pipeline stage that produced this event.")
    message: str = Field(..., min_length=1, description="Human-readable log message.")
    timestamp: datetime = Field(..., description="When this event was recorded.")
