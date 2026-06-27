"""GoldSpan/GoldExample/EvaluationDataset validation and fingerprint tests."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import EvaluationDataset, GoldExample, GoldSpan, VerdictEnum


def test_gold_span_rejects_non_positive_width() -> None:
    with pytest.raises(ValidationError):
        GoldSpan(char_start=10, char_end=10)


def test_gold_span_valid_construction() -> None:
    span = GoldSpan(char_start=0, char_end=10, note="arm chunk")
    assert span.char_end == 10


def test_gold_example_defaults_to_empty_evidence_lists() -> None:
    example = GoldExample(
        example_id="ex-1",
        narrative_path="data/sample_narrative/john_test.txt",
        original_claim="John played piano in London.",
        expected_verdict=VerdictEnum.CONTRADICTORY,
    )
    assert example.gold_evidence == []
    assert example.gold_contradicting_spans == []


def test_evaluation_dataset_requires_at_least_one_example() -> None:
    with pytest.raises(ValidationError):
        EvaluationDataset(dataset_id="ds-1", examples=[])


def test_evaluation_dataset_fingerprint_is_stable_across_identical_datasets() -> None:
    example = GoldExample(
        example_id="ex-1",
        narrative_path="some/path.txt",
        original_claim="A claim.",
        expected_verdict=VerdictEnum.CONSISTENT,
    )
    dataset_a = EvaluationDataset(dataset_id="ds-1", examples=[example])
    dataset_b = EvaluationDataset(dataset_id="ds-1", examples=[example])
    assert dataset_a.fingerprint() == dataset_b.fingerprint()


def test_evaluation_dataset_fingerprint_differs_for_different_content() -> None:
    example_a = GoldExample(
        example_id="ex-1", narrative_path="a.txt", original_claim="claim a", expected_verdict=VerdictEnum.CONSISTENT
    )
    example_b = GoldExample(
        example_id="ex-2", narrative_path="b.txt", original_claim="claim b", expected_verdict=VerdictEnum.CONTRADICTORY
    )
    dataset_a = EvaluationDataset(dataset_id="ds-1", examples=[example_a])
    dataset_b = EvaluationDataset(dataset_id="ds-1", examples=[example_b])
    assert dataset_a.fingerprint() != dataset_b.fingerprint()
