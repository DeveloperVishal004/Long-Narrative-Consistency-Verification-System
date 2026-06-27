"""to_nli_results compatibility-adapter tests: the label-mapping
direction-regression test is the highest-priority test in this file --
an inverted mapping would silently invert every verdict the frozen rule
engine produces without raising."""

import pytest

from lncvs.reasoning.fact_verification import to_nli_results
from lncvs.schemas import AtomicClaim, FactVerification, FactVerificationLabel, NLILabel


def _claim() -> AtomicClaim:
    return AtomicClaim(claim_id="claim-1", text="John used both hands")


def _verification(label: FactVerificationLabel, **overrides) -> FactVerification:
    defaults = dict(
        atomic_claim_id="claim-1",
        evidence_chunk_id="chunk-1",
        label=label,
        confidence=0.9,
        supporting_quotes=("some evidence text",),
        explanation="explanation",
    )
    defaults.update(overrides)
    return FactVerification(**defaults)


@pytest.mark.parametrize(
    "fact_label,expected_nli_label",
    [
        (FactVerificationLabel.SUPPORTED, NLILabel.ENTAILMENT),
        (FactVerificationLabel.CONTRADICTED, NLILabel.CONTRADICTION),
        (FactVerificationLabel.NOT_MENTIONED, NLILabel.NEUTRAL),
    ],
)
def test_label_mapping_direction_regression(fact_label, expected_nli_label) -> None:
    """Pins the exact, fixed direction: SUPPORTED->ENTAILMENT,
    CONTRADICTED->CONTRADICTION, NOT_MENTIONED->NEUTRAL. Must never invert."""
    claim = _claim()
    verification = _verification(fact_label)

    results = to_nli_results([verification], claim)

    assert results[0].label is expected_nli_label


def test_confidence_becomes_score() -> None:
    claim = _claim()
    verification = _verification(FactVerificationLabel.CONTRADICTED, confidence=0.87)

    results = to_nli_results([verification], claim)

    assert results[0].score == 0.87


def test_hypothesis_is_the_real_atomic_claim_text() -> None:
    claim = AtomicClaim(claim_id="claim-1", text="John used both hands to play the piano")
    verification = _verification(FactVerificationLabel.SUPPORTED)

    results = to_nli_results([verification], claim)

    assert results[0].hypothesis == "John used both hands to play the piano"


def test_premise_is_joined_supporting_quotes() -> None:
    claim = _claim()
    verification = _verification(
        FactVerificationLabel.SUPPORTED, supporting_quotes=("John played piano.", "He used both hands.")
    )

    results = to_nli_results([verification], claim)

    assert results[0].premise == "John played piano. He used both hands."


def test_empty_supporting_quotes_falls_back_to_a_valid_non_empty_premise() -> None:
    """Regression test (Phase H3): NOT_MENTIONED legitimately has zero
    supporting_quotes, but NLIResult.premise requires min_length=1 --
    '' '.join(())' would otherwise violate that constraint and crash."""
    claim = _claim()
    verification = _verification(FactVerificationLabel.NOT_MENTIONED, supporting_quotes=())

    results = to_nli_results([verification], claim)

    assert results[0].premise != ""
    assert len(results[0].premise) > 0


def test_atomic_claim_id_and_evidence_chunk_id_are_preserved() -> None:
    claim = _claim()
    verification = _verification(FactVerificationLabel.SUPPORTED, evidence_chunk_id="chunk-42")

    results = to_nli_results([verification], claim)

    assert results[0].atomic_claim_id == "claim-1"
    assert results[0].evidence_chunk_id == "chunk-42"


def test_empty_verifications_list_returns_empty_results() -> None:
    assert to_nli_results([], _claim()) == []


def test_mismatched_claim_id_raises() -> None:
    """Refuses to silently mix a different claim's verification into this
    claim's NLIResult list -- a mismatch here means a caller bug upstream."""
    claim = _claim()
    other_claims_verification = _verification(FactVerificationLabel.SUPPORTED, atomic_claim_id="claim-2")

    with pytest.raises(ValueError, match="does not match"):
        to_nli_results([other_claims_verification], claim)


def test_multiple_verifications_for_the_same_claim_all_convert() -> None:
    claim = _claim()
    verifications = [
        _verification(FactVerificationLabel.NOT_MENTIONED, evidence_chunk_id="chunk-1"),
        _verification(FactVerificationLabel.CONTRADICTED, evidence_chunk_id="chunk-2"),
    ]

    results = to_nli_results(verifications, claim)

    assert len(results) == 2
    assert {r.evidence_chunk_id for r in results} == {"chunk-1", "chunk-2"}
