"""Fact verification: the FactVerifier protocol and its cross-encoder-backed
implementation (Phase H2).

FactVerifier is deliberately evidence-level only, exactly mirroring
NLIVerifier's documented design (lncvs.reasoning.nli.service): one
FactVerification per (AtomicClaim, FusedEvidence) pair, no claim-level
aggregation. Everything downstream of a FactVerifier depends only on this
protocol, never on a concrete implementation -- CrossEncoderFactVerifier
(this module) and LLMFactVerifier (Phase H3) are interchangeable.

CrossEncoderFactVerifier is a thin wrapper around the existing, frozen
CrossEncoderNLIVerifier: it changes no NLI inference behavior whatsoever,
it only re-labels and re-shapes the already-computed NLIResult into the
richer FactVerification contract. The premise/hypothesis direction, the
underlying NLIModel, and CachingNLIModel are all reused unmodified.
"""

from typing import Protocol, runtime_checkable

from lncvs.reasoning.nli.service import CrossEncoderNLIVerifier
from lncvs.schemas import AtomicClaim, FactVerification, FactVerificationLabel, FusedEvidence, NLILabel

_NLI_LABEL_TO_FACT_VERIFICATION_LABEL: dict[NLILabel, FactVerificationLabel] = {
    NLILabel.ENTAILMENT: FactVerificationLabel.SUPPORTED,
    NLILabel.CONTRADICTION: FactVerificationLabel.CONTRADICTED,
    NLILabel.NEUTRAL: FactVerificationLabel.NOT_MENTIONED,
}


@runtime_checkable
class FactVerifier(Protocol):
    """Contract for verifying an atomic fact against retrieved evidence.

    Implementations must be evidence-level only (one FactVerification per
    evidence record) and must never apply a threshold or aggregate across
    evidence records -- that is exactly classify()'s job
    (lncvs.rules.classification), and duplicating it here would create a
    second, divergent decision path.
    """

    def verify(self, claim: AtomicClaim, evidence: list[FusedEvidence]) -> list[FactVerification]:
        """Return one FactVerification per evidence record."""
        ...


class CrossEncoderFactVerifier:
    """FactVerifier backed by the existing CrossEncoderNLIVerifier.

    Holds no state beyond its injected dependency. Determinism is
    inherited entirely from CrossEncoderNLIVerifier's own (a real
    cross-encoder in eval mode is already deterministic; CachingNLIModel
    adds throughput on top, exactly as today).
    """

    def __init__(self, nli_verifier: CrossEncoderNLIVerifier) -> None:
        self._nli_verifier = nli_verifier

    def verify(self, claim: AtomicClaim, evidence: list[FusedEvidence]) -> list[FactVerification]:
        """Re-label each underlying NLIResult into a FactVerification.

        supporting_quotes falls back to the full evidence text:
        CrossEncoderFactVerifier has no mechanism of its own to extract a
        sub-span "quote" the way an LLM judge can (Phase H3); the evidence
        text itself is always a true, verbatim "quote" of itself, so this
        never violates the verbatim-quote invariant FactVerification
        documents -- it is simply the coarsest possible valid quote.
        """
        nli_results = self._nli_verifier.verify(claim, evidence)
        return [
            FactVerification(
                atomic_claim_id=result.atomic_claim_id,
                evidence_chunk_id=result.evidence_chunk_id,
                label=_NLI_LABEL_TO_FACT_VERIFICATION_LABEL[result.label],
                confidence=result.score,
                supporting_quotes=(result.premise,),
                explanation=(
                    f"Cross-encoder NLI predicted {result.label.value} "
                    f"(score={result.score:.3f}) for this (claim, evidence) pair."
                ),
            )
            for result in nli_results
        ]
