"""LLMWindowExtractor / parse_window_extraction tests."""

import pytest

from lncvs.graph.llm_extraction.schema import WindowExtraction
from lncvs.graph.llm_extraction.service import LLMWindowExtractor, WindowExtractor, parse_window_extraction
from tests.llm.fakes import FakeStructuredLLMClient

_VALID_RESPONSE = {
    "entities": [
        {
            "local_id": "e1",
            "name": "John",
            "type": "PERSON",
            "aliases": [],
            "evidence_quotes": ["John lost his left arm in an accident in 2010."],
        },
        {
            "local_id": "e2",
            "name": "London",
            "type": "LOCATION",
            "aliases": [],
            "evidence_quotes": ["John moved to London in 2012."],
        },
    ],
    "relations": [
        {
            "subject_local_id": "e1",
            "object_local_id": "e2",
            "relation_type": "LOCATED_AT",
            "evidence_quotes": ["John moved to London in 2012."],
        }
    ],
    "events": [
        {
            "local_id": "v1",
            "predicate": "lose",
            "participants": [{"entity_local_id": "e1", "role": "PATIENT"}],
            "temporal": {"time_expression": "in 2010", "kind": "ABSOLUTE"},
            "evidence_quotes": ["John lost his left arm in an accident in 2010."],
        }
    ],
}


def test_llm_window_extractor_satisfies_protocol() -> None:
    fake = FakeStructuredLLMClient(default_response=_VALID_RESPONSE)
    assert isinstance(LLMWindowExtractor(fake), WindowExtractor)


def test_extract_returns_validated_window_extraction() -> None:
    fake = FakeStructuredLLMClient(default_response=_VALID_RESPONSE)
    extractor = LLMWindowExtractor(fake)

    result = extractor.extract("John lost his left arm... John moved to London...", chapter_index=1, window_index=None)

    assert isinstance(result, WindowExtraction)
    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert len(result.events) == 1


def test_extract_passes_the_extraction_schema_to_the_client() -> None:
    from lncvs.graph.llm_extraction.json_schema import EXTRACTION_JSON_SCHEMA

    fake = FakeStructuredLLMClient(default_response=_VALID_RESPONSE)
    extractor = LLMWindowExtractor(fake)

    extractor.extract("Some text.", chapter_index=1, window_index=None)

    assert len(fake.calls) == 1
    _, schema_used = fake.calls[0]
    assert schema_used == EXTRACTION_JSON_SCHEMA


def test_extract_rejects_empty_window_text() -> None:
    fake = FakeStructuredLLMClient(default_response=_VALID_RESPONSE)
    extractor = LLMWindowExtractor(fake)
    with pytest.raises(ValueError):
        extractor.extract("   ", chapter_index=1, window_index=None)


def test_parse_window_extraction_accepts_valid_payload() -> None:
    result = parse_window_extraction(_VALID_RESPONSE)
    assert len(result.entities) == 2


def test_parse_window_extraction_raises_value_error_not_validation_error_on_malformed_payload() -> None:
    malformed = {"entities": [{"local_id": "not-a-valid-id", "name": "John"}], "relations": [], "events": []}
    with pytest.raises(ValueError) as exc_info:
        parse_window_extraction(malformed)
    assert "schema validation" in str(exc_info.value)


def test_parse_window_extraction_is_deterministic() -> None:
    first = parse_window_extraction(_VALID_RESPONSE)
    second = parse_window_extraction(_VALID_RESPONSE)
    assert first == second


def test_parse_window_extraction_handles_response_with_no_events_key() -> None:
    """Hackathon cost optimization: EXTRACTION_JSON_SCHEMA no longer asks
    Gemini for events, so a real response from this point on never
    contains an "events" key at all. WindowExtraction.events must still
    default to () with no special-casing required downstream."""
    response_without_events = {
        "entities": _VALID_RESPONSE["entities"],
        "relations": _VALID_RESPONSE["relations"],
    }
    result = parse_window_extraction(response_without_events)
    assert result.events == ()
    assert len(result.entities) == 2
    assert len(result.relations) == 1


def test_extract_passes_schema_with_no_events_property_to_the_client() -> None:
    """The literal schema sent to Gemini must not request events at all --
    this is the actual cost-optimization wire-format change."""
    fake = FakeStructuredLLMClient(default_response=_VALID_RESPONSE)
    extractor = LLMWindowExtractor(fake)

    extractor.extract("Some text.", chapter_index=1, window_index=None)

    _, schema_used = fake.calls[0]
    assert "events" not in schema_used["properties"]
    assert "events" not in schema_used["required"]
