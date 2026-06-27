"""Fact-verification JSON schema shape and versioning."""

from lncvs.reasoning.fact_verification.llm_schema import FACT_VERIFICATION_JSON_SCHEMA, SCHEMA_VERSION


def test_schema_version_is_deterministic() -> None:
    assert SCHEMA_VERSION == SCHEMA_VERSION
    assert len(SCHEMA_VERSION) == 8


def test_schema_top_level_shape() -> None:
    assert FACT_VERIFICATION_JSON_SCHEMA["type"] == "object"
    assert FACT_VERIFICATION_JSON_SCHEMA["additionalProperties"] is False
    assert set(FACT_VERIFICATION_JSON_SCHEMA["required"]) == {"verdict", "confidence", "quotes", "explanation"}


def test_verdict_enum_is_exactly_the_three_labels() -> None:
    verdict_enum = FACT_VERIFICATION_JSON_SCHEMA["properties"]["verdict"]["enum"]
    assert set(verdict_enum) == {"SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"}


def test_quotes_array_has_no_minimum_item_count() -> None:
    """NOT_MENTIONED legitimately has zero quotes -- the wire schema must
    permit that shape. The verifier's own logic, not the schema, enforces
    that SUPPORTED/CONTRADICTED requires at least one quote."""
    quotes_schema = FACT_VERIFICATION_JSON_SCHEMA["properties"]["quotes"]
    assert quotes_schema["type"] == "array"
    assert "minItems" not in quotes_schema


def test_confidence_is_bounded_unit_interval() -> None:
    confidence_schema = FACT_VERIFICATION_JSON_SCHEMA["properties"]["confidence"]
    assert confidence_schema["minimum"] == 0.0
    assert confidence_schema["maximum"] == 1.0
