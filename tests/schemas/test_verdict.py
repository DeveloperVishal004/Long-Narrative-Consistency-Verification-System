"""FinalVerdict validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import FinalVerdict, VerdictEnum


def test_final_verdict_valid_construction() -> None:
    verdict = FinalVerdict(
        verdict=VerdictEnum.CONTRADICTORY,
        fired_rule="rule_1_contradiction",
        rationale="Claim 'John used both hands' is contradicted by evidence of his lost arm.",
        confidence=0.93,
        contradicted_claim_ids=["claim-1"],
    )
    assert verdict.verdict is VerdictEnum.CONTRADICTORY
    assert verdict.contradicted_claim_ids == ["claim-1"]


def test_final_verdict_confidence_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        FinalVerdict(
            verdict=VerdictEnum.CONSISTENT,
            fired_rule="rule_3_all_supported",
            rationale="All atomic claims supported.",
            confidence=1.1,
        )


def test_final_verdict_rejects_empty_fired_rule() -> None:
    with pytest.raises(ValidationError):
        FinalVerdict(
            verdict=VerdictEnum.INSUFFICIENT_EVIDENCE,
            fired_rule="",
            rationale="No evidence found for one atomic claim.",
        )


def test_final_verdict_defaults_have_no_contradicted_or_unresolved_claims() -> None:
    verdict = FinalVerdict(
        verdict=VerdictEnum.CONSISTENT,
        fired_rule="rule_3_all_supported",
        rationale="All atomic claims supported.",
    )
    assert verdict.contradicted_claim_ids == []
    assert verdict.unresolved_claim_ids == []
