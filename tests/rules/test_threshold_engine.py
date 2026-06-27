"""ThresholdRuleEngine tests: exhaustive truth table over the three verdicts, threshold boundaries.

Per CLAUDE.md's Rule Engine testing requirement, this is the exhaustive
truth-table suite over {has CONTRADICTED claim, has UNRESOLVED claim, all
SUPPORTED} that was previously a Phase 1 placeholder in
test_rule_engine_contract.py.
"""

import pytest

from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine
from lncvs.schemas import AtomicClaim, EvidenceLedger, NLILabel, NLIResult, VerdictEnum


def _config(contradiction_threshold: float = 0.7, entailment_threshold: float = 0.7) -> RuleEngineConfig:
    return RuleEngineConfig(
        contradiction_threshold=contradiction_threshold, entailment_threshold=entailment_threshold
    )


def _ledger_with_claims(*claim_ids: str) -> EvidenceLedger:
    ledger = EvidenceLedger(original_claim="A multi-part narrative claim.")
    for claim_id in claim_ids:
        ledger.atomic_claims.append(AtomicClaim(claim_id=claim_id, text=f"text for {claim_id}"))
    return ledger


def _result(claim_id: str, chunk_id: str, label: NLILabel, score: float) -> NLIResult:
    return NLIResult(
        atomic_claim_id=claim_id, evidence_chunk_id=chunk_id, label=label, score=score, premise="p", hypothesis="h"
    )


def test_evaluate_raises_on_ledger_with_no_atomic_claims() -> None:
    engine = ThresholdRuleEngine(_config())
    ledger = EvidenceLedger(original_claim="A claim.")

    with pytest.raises(ValueError, match="no atomic_claims"):
        engine.evaluate(ledger)


def test_all_claims_supported_yields_consistent() -> None:
    ledger = _ledger_with_claims("claim-1", "claim-2")
    ledger.nli_results.extend(
        [
            _result("claim-1", "chunk-1", NLILabel.ENTAILMENT, 0.9),
            _result("claim-2", "chunk-2", NLILabel.ENTAILMENT, 0.85),
        ]
    )
    engine = ThresholdRuleEngine(_config())

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONSISTENT
    assert verdict.fired_rule == "rule_3_all_supported"
    assert verdict.contradicted_claim_ids == []
    assert verdict.unresolved_claim_ids == []


def test_any_contradicted_claim_yields_contradictory() -> None:
    ledger = _ledger_with_claims("claim-1", "claim-2")
    ledger.nli_results.extend(
        [
            _result("claim-1", "chunk-1", NLILabel.ENTAILMENT, 0.9),
            _result("claim-2", "chunk-2", NLILabel.CONTRADICTION, 0.95),
        ]
    )
    engine = ThresholdRuleEngine(_config())

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONTRADICTORY
    assert verdict.fired_rule == "rule_1_contradiction"
    assert verdict.contradicted_claim_ids == ["claim-2"]


def test_any_unresolved_claim_with_no_contradiction_yields_insufficient_evidence() -> None:
    """The cardinal invariant: missing evidence must never silently become CONTRADICTORY."""
    ledger = _ledger_with_claims("claim-1", "claim-2")
    ledger.nli_results.extend([_result("claim-1", "chunk-1", NLILabel.ENTAILMENT, 0.9)])
    # claim-2 has zero NLI results -- the missing-evidence case.
    engine = ThresholdRuleEngine(_config())

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.INSUFFICIENT_EVIDENCE
    assert verdict.fired_rule == "rule_2_unresolved"
    assert verdict.unresolved_claim_ids == ["claim-2"]
    assert verdict.contradicted_claim_ids == []


def test_lenient_policy_unresolved_without_contradiction_yields_consistent() -> None:
    """Lenient consistency policy (consistency_requires_entailment=False): a
    claim set with no contradiction and only neutral/unresolved evidence is
    CONSISTENT, not INSUFFICIENT_EVIDENCE. Fits datasets whose 'consistent'
    means 'not contradicted by the source'."""
    ledger = _ledger_with_claims("claim-1", "claim-2")
    ledger.nli_results.extend([_result("claim-1", "chunk-1", NLILabel.ENTAILMENT, 0.9)])
    # claim-2 has zero NLI results (neutral/unresolved) -- under lenient policy it does not block CONSISTENT.
    config = RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7, consistency_requires_entailment=False)
    engine = ThresholdRuleEngine(config)

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONSISTENT
    assert verdict.fired_rule == "rule_3_no_contradiction"
    assert verdict.unresolved_claim_ids == ["claim-2"]


def test_lenient_policy_still_yields_contradictory_on_contradiction() -> None:
    """Lenient policy must NOT weaken contradiction detection: a contradiction still wins."""
    ledger = _ledger_with_claims("claim-1", "claim-2")
    ledger.nli_results.extend([_result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.9)])
    config = RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7, consistency_requires_entailment=False)
    engine = ThresholdRuleEngine(config)

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONTRADICTORY
    assert verdict.fired_rule == "rule_1_contradiction"


def test_contradiction_and_unresolved_together_yields_contradictory() -> None:
    """A confirmed contradiction outweighs a coexisting coverage gap on a different claim."""
    ledger = _ledger_with_claims("claim-1", "claim-2", "claim-3")
    ledger.nli_results.extend([_result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.9)])
    # claim-2 and claim-3 have zero NLI results.
    engine = ThresholdRuleEngine(_config())

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONTRADICTORY
    assert verdict.fired_rule == "rule_1_contradiction"
    assert verdict.contradicted_claim_ids == ["claim-1"]
    assert set(verdict.unresolved_claim_ids) == {"claim-2", "claim-3"}


def test_threshold_boundary_below_does_not_fire_contradiction_rule() -> None:
    ledger = _ledger_with_claims("claim-1")
    ledger.nli_results.append(_result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.69))
    engine = ThresholdRuleEngine(_config(contradiction_threshold=0.7))

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.INSUFFICIENT_EVIDENCE


def test_evaluate_is_deterministic_across_repeated_calls() -> None:
    ledger = _ledger_with_claims("claim-1", "claim-2")
    ledger.nli_results.extend(
        [
            _result("claim-1", "chunk-1", NLILabel.CONTRADICTION, 0.9),
            _result("claim-2", "chunk-2", NLILabel.ENTAILMENT, 0.9),
        ]
    )
    engine = ThresholdRuleEngine(_config())

    first = engine.evaluate(ledger)
    second = engine.evaluate(ledger)

    assert first.verdict == second.verdict
    assert first.fired_rule == second.fired_rule
    assert first.contradicted_claim_ids == second.contradicted_claim_ids
    assert first.unresolved_claim_ids == second.unresolved_claim_ids


def test_dummy_case_section_14_resolves_to_contradictory() -> None:
    """PROJECT_SPEC.md Section 14: John lost his left arm / John used both hands
    -- the standing end-to-end acceptance scenario, exercised here at the
    rule-engine layer directly against scripted NLI results."""
    ledger = _ledger_with_claims("claim-piano", "claim-hands", "claim-london")
    ledger.nli_results.extend(
        [
            _result("claim-piano", "chunk-london", NLILabel.NEUTRAL, 0.6),
            _result("claim-hands", "chunk-arm", NLILabel.CONTRADICTION, 0.95),
            _result("claim-london", "chunk-london", NLILabel.ENTAILMENT, 0.9),
        ]
    )
    engine = ThresholdRuleEngine(_config())

    verdict = engine.evaluate(ledger)

    assert verdict.verdict is VerdictEnum.CONTRADICTORY
    assert verdict.contradicted_claim_ids == ["claim-hands"]
