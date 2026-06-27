"""Pure claim-classification helper for the deterministic rule engine.

Threshold application happens in exactly one place: here, called by
ThresholdRuleEngine.evaluate(). classify() is not a pipeline component and
writes nothing to the ledger itself -- it is a stateless function that
ThresholdRuleEngine and the Phase 5 driver (recording explainability
records via LedgerService.record_classification) both call with identical
inputs, so the verdict and the recorded Contradiction/SupportingEvidence/
unsupported_claims trace are guaranteed to agree, since classify() is pure.
"""

from typing import NamedTuple

from lncvs.rules.engine import ClaimStatus, RuleEngineConfig
from lncvs.schemas import Contradiction, NLILabel, NLIResult, SupportingEvidence


class ClassificationOutcome(NamedTuple):
    """Per-claim classification, as both a status map and typed ledger records."""

    statuses: dict[str, ClaimStatus]
    contradictions: list[Contradiction]
    supporting_evidence: list[SupportingEvidence]
    unsupported_claim_ids: list[str]


def classify(
    nli_results: list[NLIResult],
    atomic_claim_ids: list[str],
    config: RuleEngineConfig,
) -> ClassificationOutcome:
    """Classify every atomic_claim_id by its NLI results under config's thresholds.

    CONTRADICTED dominates SUPPORTED for the same claim: a single
    contradiction outweighs coexisting support. A claim with no NLI result
    clearing either threshold -- including a claim with zero NLI results at
    all, the missing-evidence case -- is UNRESOLVED, never CONTRADICTED.
    """
    results_by_claim: dict[str, list[NLIResult]] = {claim_id: [] for claim_id in atomic_claim_ids}
    for result in nli_results:
        results_by_claim.setdefault(result.atomic_claim_id, []).append(result)

    statuses: dict[str, ClaimStatus] = {}
    contradictions: list[Contradiction] = []
    supporting_evidence: list[SupportingEvidence] = []
    unsupported_claim_ids: list[str] = []

    for claim_id in atomic_claim_ids:
        claim_results = results_by_claim.get(claim_id, [])
        contradicting = [
            result
            for result in claim_results
            if result.label is NLILabel.CONTRADICTION and result.score >= config.contradiction_threshold
        ]
        entailing = [
            result
            for result in claim_results
            if result.label is NLILabel.ENTAILMENT and result.score >= config.entailment_threshold
        ]

        if contradicting:
            statuses[claim_id] = ClaimStatus.CONTRADICTED
            for result in contradicting:
                contradictions.append(
                    Contradiction(
                        atomic_claim_id=claim_id,
                        evidence_chunk_id=result.evidence_chunk_id,
                        nli_score=result.score,
                        explanation=(
                            f"NLI contradiction (score={result.score:.3f}) against evidence "
                            f"chunk {result.evidence_chunk_id!r}."
                        ),
                    )
                )
        elif entailing:
            statuses[claim_id] = ClaimStatus.SUPPORTED
            for result in entailing:
                supporting_evidence.append(
                    SupportingEvidence(
                        atomic_claim_id=claim_id,
                        evidence_chunk_id=result.evidence_chunk_id,
                        nli_score=result.score,
                        explanation=(
                            f"NLI entailment (score={result.score:.3f}) from evidence "
                            f"chunk {result.evidence_chunk_id!r}."
                        ),
                    )
                )
        else:
            statuses[claim_id] = ClaimStatus.UNRESOLVED
            unsupported_claim_ids.append(claim_id)

    return ClassificationOutcome(
        statuses=statuses,
        contradictions=contradictions,
        supporting_evidence=supporting_evidence,
        unsupported_claim_ids=unsupported_claim_ids,
    )
