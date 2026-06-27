"""classify() pure helper tests: dominance rule, threshold boundaries, missing-evidence handling."""

from lncvs.rules import ClaimStatus, RuleEngineConfig, classify
from lncvs.schemas import NLILabel, NLIResult


def _config(contradiction_threshold: float = 0.7, entailment_threshold: float = 0.7) -> RuleEngineConfig:
    return RuleEngineConfig(
        contradiction_threshold=contradiction_threshold, entailment_threshold=entailment_threshold
    )


def _result(
    claim_id: str, chunk_id: str, label: NLILabel, score: float, premise: str = "p", hypothesis: str = "h"
) -> NLIResult:
    return NLIResult(
        atomic_claim_id=claim_id,
        evidence_chunk_id=chunk_id,
        label=label,
        score=score,
        premise=premise,
        hypothesis=hypothesis,
    )


def test_claim_with_no_results_is_unresolved() -> None:
    outcome = classify([], ["claim-1"], _config())

    assert outcome.statuses["claim-1"] is ClaimStatus.UNRESOLVED
    assert outcome.unsupported_claim_ids == ["claim-1"]
    assert outcome.contradictions == []
    assert outcome.supporting_evidence == []


def test_claim_with_only_neutral_results_is_unresolved() -> None:
    results = [_result("claim-1", "chunk-1", NLILabel.NEUTRAL, 0.99)]
    outcome = classify(results, ["claim-1"], _config())

    assert outcome.statuses["claim-1"] is ClaimStatus.UNRESOLVED


def test_claim_with_contradiction_above_threshold_is_contradicted() -> None:
    results = [_result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.9)]
    outcome = classify(results, ["claim-1"], _config(contradiction_threshold=0.7))

    assert outcome.statuses["claim-1"] is ClaimStatus.CONTRADICTED
    assert len(outcome.contradictions) == 1
    assert outcome.contradictions[0].evidence_chunk_id == "chunk-1"


def test_claim_with_contradiction_below_threshold_is_unresolved() -> None:
    results = [_result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.5)]
    outcome = classify(results, ["claim-1"], _config(contradiction_threshold=0.7))

    assert outcome.statuses["claim-1"] is ClaimStatus.UNRESOLVED


def test_contradiction_threshold_boundary_is_inclusive() -> None:
    results = [_result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.7)]
    outcome = classify(results, ["claim-1"], _config(contradiction_threshold=0.7))

    assert outcome.statuses["claim-1"] is ClaimStatus.CONTRADICTED


def test_claim_with_entailment_above_threshold_is_supported() -> None:
    results = [_result("claim-1", "chunk-1", NLILabel.ENTAILMENT, 0.85)]
    outcome = classify(results, ["claim-1"], _config(entailment_threshold=0.7))

    assert outcome.statuses["claim-1"] is ClaimStatus.SUPPORTED
    assert len(outcome.supporting_evidence) == 1


def test_entailment_threshold_boundary_is_inclusive() -> None:
    results = [_result("claim-1", "chunk-1", NLILabel.ENTAILMENT, 0.7)]
    outcome = classify(results, ["claim-1"], _config(entailment_threshold=0.7))

    assert outcome.statuses["claim-1"] is ClaimStatus.SUPPORTED


def test_contradiction_dominates_supporting_evidence_for_the_same_claim() -> None:
    """A claim with both a strong entailment and a strong contradiction must
    resolve to CONTRADICTED -- the dominance rule, not a fourth ambiguous state."""
    results = [
        _result("claim-1", "chunk-entails", NLILabel.ENTAILMENT, 0.95),
        _result("claim-1", "chunk-contradicts", NLILabel.CONTRADICTION, 0.8),
    ]
    outcome = classify(results, ["claim-1"], _config())

    assert outcome.statuses["claim-1"] is ClaimStatus.CONTRADICTED
    assert outcome.supporting_evidence == []
    assert len(outcome.contradictions) == 1


def test_classification_is_independent_per_claim() -> None:
    results = [
        _result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.9),
        _result("claim-2", "chunk-2", NLILabel.ENTAILMENT, 0.9),
    ]
    outcome = classify(results, ["claim-1", "claim-2"], _config())

    assert outcome.statuses["claim-1"] is ClaimStatus.CONTRADICTED
    assert outcome.statuses["claim-2"] is ClaimStatus.SUPPORTED
    assert outcome.unsupported_claim_ids == []


def test_classify_is_deterministic_across_calls() -> None:
    results = [
        _result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.9),
        _result("claim-2", "chunk-2", NLILabel.ENTAILMENT, 0.9),
    ]
    config = _config()

    first = classify(results, ["claim-1", "claim-2"], config)
    second = classify(results, ["claim-1", "claim-2"], config)

    assert first.statuses == second.statuses
    assert first.contradictions == second.contradictions
    assert first.supporting_evidence == second.supporting_evidence
