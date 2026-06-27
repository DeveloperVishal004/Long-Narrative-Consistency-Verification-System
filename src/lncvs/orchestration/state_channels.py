"""GraphChannels: the LangGraph compile-time state_schema.

Structurally identical to lncvs.schemas.state.GraphState ({ledger, control})
-- this binding exists only to attach Annotated reducer metadata that
LangGraph reads at compile time. schemas.GraphState itself stays unchanged,
per the approved Phase 7 design: GraphState is the canonical domain-facing
type; this is LangGraph-facing plumbing, used only inside orchestration/
and never imported elsewhere. Construct/read lncvs.schemas.GraphState at
the orchestration boundary (LangGraphPipeline.run()'s public contract);
nodes operate on this Annotated shape internally.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from lncvs.orchestration.reducers import last_write_wins
from lncvs.schemas import ControlState, EvidenceLedger


class GraphChannels(BaseModel):
    """The compiled graph's state_schema -- {ledger, control}, each with an
    explicit last_write_wins reducer (see reducers.py for the reconciliation
    rationale)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ledger: Annotated[EvidenceLedger, last_write_wins] = Field(...)
    control: Annotated[ControlState, last_write_wins] = Field(...)
