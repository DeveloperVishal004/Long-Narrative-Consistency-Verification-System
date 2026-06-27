"""FactVerification / FactVerificationLabel schema tests."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import FactVerification, FactVerificationLabel


def _verification(**overrides) -> FactVerification:
    defaults = dict(
        atomic_claim_id="claim-1",
        evidence_chunk_id="chunk-1",
        label=FactVerificationLabel.SUPPORTED,
        confidence=0.9,
        supporting_quotes=("John lost his left arm.",),
        explanation="The evidence directly states this fact.",
    )
    defaults.update(overrides)
    return FactVerification(**defaults)


def test_fact_verification_round_trip() -> None:
    verification = _verification()
    assert verification.label is FactVerificationLabel.SUPPORTED
    assert verification.confidence == 0.9
    assert verification.supporting_quotes == ("John lost his left arm.",)


def test_fact_verification_is_frozen() -> None:
    verification = _verification()
    with pytest.raises(ValidationError):
        verification.label = FactVerificationLabel.CONTRADICTED


def test_confidence_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _verification(confidence=1.5)
    with pytest.raises(ValidationError):
        _verification(confidence=-0.1)


def test_supporting_quotes_defaults_to_empty_tuple() -> None:
    """Phase H3: NOT_MENTIONED legitimately has zero real quotes -- forcing
    a non-empty tuple would mean fabricating one, the exact hallucination
    this field exists to prevent. Relaxed from the original H2 min_length=1."""
    verification = _verification(supporting_quotes=())
    assert verification.supporting_quotes == ()


def test_supporting_quotes_omitted_entirely_defaults_to_empty_tuple() -> None:
    verification = FactVerification(
        atomic_claim_id="claim-1",
        evidence_chunk_id="chunk-1",
        label=FactVerificationLabel.NOT_MENTIONED,
        confidence=0.8,
        explanation="The evidence does not address this fact.",
    )
    assert verification.supporting_quotes == ()


def test_explanation_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        _verification(explanation="")


def test_label_rejects_values_outside_closed_vocabulary() -> None:
    with pytest.raises(ValidationError):
        _verification(label="MAYBE")


def test_fact_verification_label_has_exactly_three_values() -> None:
    assert {label.value for label in FactVerificationLabel} == {"SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"}
