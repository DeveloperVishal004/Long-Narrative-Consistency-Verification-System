"""Compatibility adapter: FactVerification -> NLIResult (Phase H2).

Exists solely so the frozen rule engine (lncvs.rules.classification.classify,
lncvs.rules.threshold_engine.ThresholdRuleEngine) can keep consuming
NLIResult unchanged while the verification layer is upgraded to FactVerifier.
This is a pure, stateless mapping -- no model calls, no aggregation, no
thresholds. Premise/hypothesis are reconstructed from the FactVerification's
own fields (supporting_quotes joined as the premise text, since
FactVerification does not separately store the claim's hypothesis text --
see to_nli_results' docstring for why this is safe).

Phase H3 finding, disclosed: NLIResult.premise requires min_length=1, but
FactVerification.supporting_quotes is legitimately empty for NOT_MENTIONED
(see schemas/fact_verification.py's own H3 relaxation note) -- "".join(())
would violate that constraint. _NO_QUOTE_PREMISE_SENTINEL is the minimal
fix: a fixed, honest placeholder ("no real evidence text was cited, since
nothing in the evidence supports or contradicts this fact") used only when
supporting_quotes is empty. classify()/ThresholdRuleEngine never read
premise content (only label + score), so this sentinel affects nothing
verdict-relevant -- it exists purely so the audit-facing NLIResult record
remains valid and honestly says "there was no real premise" rather than
fabricating one.

This module is the ONLY place the FactVerificationLabel -> NLILabel mapping
exists; lncvs.reasoning.fact_verification.service defines the reverse
(NLILabel -> FactVerificationLabel) direction for CrossEncoderFactVerifier.
Both mappings are pinned by a direction-regression test (mirroring the NLI
premise/hypothesis pin) since an inverted mapping would silently invert
every verdict without raising.
"""

from lncvs.schemas import AtomicClaim, FactVerification, FactVerificationLabel, NLILabel, NLIResult

_NO_QUOTE_PREMISE_SENTINEL = "(no evidence text cited -- the verifier found nothing supporting or contradicting this fact)"

_FACT_VERIFICATION_LABEL_TO_NLI_LABEL: dict[FactVerificationLabel, NLILabel] = {
    FactVerificationLabel.SUPPORTED: NLILabel.ENTAILMENT,
    FactVerificationLabel.CONTRADICTED: NLILabel.CONTRADICTION,
    FactVerificationLabel.NOT_MENTIONED: NLILabel.NEUTRAL,
}


def to_nli_results(verifications: list[FactVerification], claim: AtomicClaim) -> list[NLIResult]:
    """Convert FactVerifications for one claim into the NLIResult shape the
    frozen rule engine expects.

    claim is passed explicitly (rather than trusting verification.atomic_claim_id
    alone) so hypothesis is always the real AtomicClaim.text, not a
    reconstruction -- NLIResult.hypothesis is documented as "the atomic
    claim text", and only the caller holding the AtomicClaim can supply
    that exactly. Every verification's atomic_claim_id must match claim.claim_id,
    or this raises rather than silently mixing results from different claims.
    """
    results = []
    for verification in verifications:
        if verification.atomic_claim_id != claim.claim_id:
            raise ValueError(
                f"FactVerification.atomic_claim_id {verification.atomic_claim_id!r} does not match "
                f"claim.claim_id {claim.claim_id!r}; refusing to mix verifications across claims."
            )
        results.append(
            NLIResult(
                atomic_claim_id=verification.atomic_claim_id,
                evidence_chunk_id=verification.evidence_chunk_id,
                label=_FACT_VERIFICATION_LABEL_TO_NLI_LABEL[verification.label],
                score=verification.confidence,
                premise=" ".join(verification.supporting_quotes) or _NO_QUOTE_PREMISE_SENTINEL,
                hypothesis=claim.text,
            )
        )
    return results
