"""NLIResult validation tests, including the fixed premise/hypothesis direction."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import NLILabel, NLIResult


def test_nli_result_valid_construction_with_fixed_direction() -> None:
    result = NLIResult(
        atomic_claim_id="claim-1",
        evidence_chunk_id="chunk-0001",
        label=NLILabel.CONTRADICTION,
        score=0.93,
        premise="John lost his left arm in an accident in 2010.",
        hypothesis="John used both hands.",
    )
    # Direction is fixed: premise = evidence, hypothesis = atomic claim.
    assert result.premise == "John lost his left arm in an accident in 2010."
    assert result.hypothesis == "John used both hands."


def test_nli_result_score_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        NLIResult(
            atomic_claim_id="claim-1",
            evidence_chunk_id="chunk-0001",
            label=NLILabel.ENTAILMENT,
            score=1.2,
            premise="evidence text",
            hypothesis="claim text",
        )


def test_nli_result_rejects_empty_premise() -> None:
    with pytest.raises(ValidationError):
        NLIResult(
            atomic_claim_id="claim-1",
            evidence_chunk_id="chunk-0001",
            label=NLILabel.NEUTRAL,
            score=0.5,
            premise="",
            hypothesis="claim text",
        )
