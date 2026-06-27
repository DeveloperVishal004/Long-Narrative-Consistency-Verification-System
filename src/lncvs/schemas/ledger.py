"""EvidenceLedger — the single, fully-typed source of truth for a verification run.

Every field is a typed model or a typed list of models; nothing here is a
raw dict or List[Any]. The ledger is mutated only through
lncvs.ledger.service.LedgerService — this model itself contains no
mutation logic beyond what Pydantic provides.
"""

from pydantic import BaseModel, Field

from lncvs.schemas.claims import AtomicClaim, ProbeQuestion
from lncvs.schemas.evidence import Contradiction, FusedEvidence, RetrievedEvidence, SupportingEvidence
from lncvs.schemas.nli import NLIResult
from lncvs.schemas.reasoning import LedgerEvent, ReasoningStep
from lncvs.schemas.retrieval_query import RetrievalQuery
from lncvs.schemas.verdict import FinalVerdict


class EvidenceLedger(BaseModel):
    """Explicit reasoning state for verifying a single narrative claim."""

    original_claim: str = Field(..., min_length=1, description="The claim being verified, as originally stated.")
    original_claim_id: str | None = Field(
        default=None,
        description="Deterministic content-hash ID of original_claim, set once by claim decomposition. "
        "The referent for every AtomicClaim.parent_claim_id derived from this claim.",
    )
    narrative_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of all DocumentChunks belonging to the narrative under verification. "
        "Chunk bodies live in an injected store, not here.",
    )
    atomic_claims: list[AtomicClaim] = Field(default_factory=list, description="Decomposed atomic claims.")
    probe_questions: list[ProbeQuestion] = Field(
        default_factory=list, description="Retrieval-oriented questions generated per atomic claim."
    )
    retrieval_queries: list[RetrievalQuery] = Field(
        default_factory=list, description="Unified retrieval queries built from atomic claims and probe questions."
    )
    retrieved_evidence: list[RetrievedEvidence] = Field(
        default_factory=list, description="Per-source evidence, pre-fusion. Claim-linked via atomic_claim_id."
    )
    fused_evidence: list[FusedEvidence] = Field(
        default_factory=list, description="Deduplicated, RRF-ranked evidence."
    )
    nli_results: list[NLIResult] = Field(default_factory=list, description="All NLI verification outcomes.")
    contradictions: list[Contradiction] = Field(
        default_factory=list, description="Claims found to be contradicted by evidence."
    )
    supporting_evidence: list[SupportingEvidence] = Field(
        default_factory=list, description="Claims found to be entailed by evidence."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Atomic claim IDs with neither entailing nor contradicting evidence.",
    )
    reasoning_trace: list[ReasoningStep] = Field(
        default_factory=list, description="Human-readable, stage-ordered reasoning trace."
    )
    ledger_log: list[LedgerEvent] = Field(default_factory=list, description="Append-only audit log.")
    final_verdict: FinalVerdict | None = Field(
        default=None, description="Set exactly once, by the rule engine, at the end of the run."
    )
