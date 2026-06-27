"""Fact verification contract (Phase H2).

FactVerification is the richer, audit-first sibling of NLIResult: it
carries verbatim supporting_quotes and a human-readable explanation that
NLIResult deliberately does not (NLIResult only ever needs label + score
for the rule engine). FactVerifier implementations are evidence-level
only, exactly mirroring NLIVerifier's documented design (one
FactVerification per (AtomicClaim, FusedEvidence) pair, no claim-level
aggregation) -- see lncvs.reasoning.nli.service's docstring for why that
symmetry matters: claim-level aggregation requires thresholds, and
thresholds belong to exactly one place (lncvs.rules.classification.classify),
which FactVerifier must never duplicate or bypass.
"""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas.enums import FactVerificationLabel


class FactVerification(BaseModel):
    """The result of verifying a single atomic fact against a single
    evidence record. supporting_quotes are intended to be verbatim
    substrings of the evidence text (enforced by LLMFactVerifier in
    Phase H3 via the existing graph-provenance quote-matching discipline;
    CrossEncoderFactVerifier has no quote-generation capability of its own
    and falls back to the full evidence text -- see service.py)."""

    model_config = ConfigDict(frozen=True)

    atomic_claim_id: str = Field(..., min_length=1, description="ID of the AtomicClaim being verified.")
    evidence_chunk_id: str = Field(..., min_length=1, description="ID of the DocumentChunk used as evidence.")
    label: FactVerificationLabel = Field(..., description="SUPPORTED, CONTRADICTED, or NOT_MENTIONED.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Verifier confidence for the assigned label.")
    # Phase H3 finding, disclosed: NOT_MENTIONED legitimately has ZERO real
    # supporting quotes -- forcing a non-empty tuple here would mean either
    # fabricating a quote (the exact hallucination this field exists to
    # prevent) or rejecting every correct NOT_MENTIONED verdict. Relaxed
    # from the original min_length=1 to default=() for that reason.
    # CrossEncoderFactVerifier's behavior is unaffected: it always supplies
    # a non-empty fallback quote (the full evidence text, itself a trivially
    # valid verbatim quote), so this relaxation only widens what is
    # *permitted*, never what CrossEncoderFactVerifier actually produces.
    supporting_quotes: tuple[str, ...] = Field(
        default=(), description="Verbatim quote(s) from the evidence text supporting this verdict; empty for NOT_MENTIONED."
    )
    explanation: str = Field(..., min_length=1, description="Human-readable rationale for the assigned label.")
