"""RawFactVerdict / parse_fact_verdict tests -- pure, no LLM involved."""

import pytest

from lncvs.reasoning.fact_verification.llm_raw import parse_fact_verdict


def test_valid_supported_payload_parses() -> None:
    raw = {"verdict": "SUPPORTED", "confidence": 0.9, "quotes": ["John lost his left arm."], "explanation": "Directly stated."}

    result = parse_fact_verdict(raw)

    assert result.verdict == "SUPPORTED"
    assert result.confidence == 0.9
    assert result.quotes == ("John lost his left arm.",)


def test_not_mentioned_with_empty_quotes_parses() -> None:
    raw = {"verdict": "NOT_MENTIONED", "confidence": 0.8, "quotes": [], "explanation": "Passage is unrelated."}

    result = parse_fact_verdict(raw)

    assert result.verdict == "NOT_MENTIONED"
    assert result.quotes == ()


def test_invalid_verdict_value_raises_value_error_not_bare_validation_error() -> None:
    raw = {"verdict": "MAYBE", "confidence": 0.5, "quotes": [], "explanation": "x"}

    with pytest.raises(ValueError, match="schema validation"):
        parse_fact_verdict(raw)


def test_confidence_out_of_range_raises() -> None:
    raw = {"verdict": "SUPPORTED", "confidence": 1.5, "quotes": ["q"], "explanation": "x"}

    with pytest.raises(ValueError, match="schema validation"):
        parse_fact_verdict(raw)


def test_missing_explanation_raises() -> None:
    raw = {"verdict": "NOT_MENTIONED", "confidence": 0.5, "quotes": []}

    with pytest.raises(ValueError, match="schema validation"):
        parse_fact_verdict(raw)


def test_empty_explanation_raises() -> None:
    raw = {"verdict": "NOT_MENTIONED", "confidence": 0.5, "quotes": [], "explanation": ""}

    with pytest.raises(ValueError, match="schema validation"):
        parse_fact_verdict(raw)
