"""RuleEngine contract tests.

Phase 0 ships the interface only — no verdict logic. These tests assert the
contract (abstractness, config thresholds, the evaluate() signature) and
leave a clearly-marked placeholder for the Phase 1 exhaustive truth-table
suite described in CLAUDE.md's Rule Engine Specification.
"""

import pytest

from lncvs.rules import ClaimStatus, RuleEngine, RuleEngineConfig
from lncvs.schemas import EvidenceLedger, FinalVerdict, VerdictEnum


def test_rule_engine_cannot_be_instantiated_directly() -> None:
    config = RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7)
    with pytest.raises(TypeError):
        RuleEngine(config)  # type: ignore[abstract]


def test_rule_engine_config_rejects_thresholds_outside_unit_interval() -> None:
    with pytest.raises(ValueError):
        RuleEngineConfig(contradiction_threshold=1.5, entailment_threshold=0.7)


def test_claim_status_has_exactly_three_members() -> None:
    assert {member.value for member in ClaimStatus} == {
        "CONTRADICTED",
        "SUPPORTED",
        "UNRESOLVED",
    }


def test_concrete_subclass_satisfies_the_contract() -> None:
    """A concrete RuleEngine implementing evaluate() must be instantiable and callable.

    This stub returns a fixed verdict purely to prove the interface shape;
    it is not the Phase 1 rule logic.
    """

    class _StubRuleEngine(RuleEngine):
        def evaluate(self, ledger: EvidenceLedger) -> FinalVerdict:
            return FinalVerdict(
                verdict=VerdictEnum.INSUFFICIENT_EVIDENCE,
                fired_rule="stub",
                rationale="Stub engine always returns INSUFFICIENT_EVIDENCE.",
            )

    config = RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7)
    engine = _StubRuleEngine(config)
    ledger = EvidenceLedger(original_claim="John played a two-handed piano piece in London.")

    result = engine.evaluate(ledger)

    assert result.verdict is VerdictEnum.INSUFFICIENT_EVIDENCE
    assert engine.config.contradiction_threshold == 0.7


# The exhaustive truth table over {has CONTRADICTED, has UNRESOLVED, all
# SUPPORTED}, threshold-boundary tests, and the PROJECT_SPEC.md Section 14
# dummy case now live in tests/rules/test_threshold_engine.py, alongside
# the concrete ThresholdRuleEngine they exercise.
