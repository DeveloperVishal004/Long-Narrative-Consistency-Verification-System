"""Extraction JSON schema shape and versioning."""

from lncvs.graph.llm_extraction.json_schema import EXTRACTION_JSON_SCHEMA, SCHEMA_VERSION


def test_schema_version_is_deterministic() -> None:
    assert SCHEMA_VERSION == SCHEMA_VERSION
    assert len(SCHEMA_VERSION) == 8


def test_schema_top_level_shape() -> None:
    assert EXTRACTION_JSON_SCHEMA["type"] == "object"
    assert EXTRACTION_JSON_SCHEMA["additionalProperties"] is False
    assert set(EXTRACTION_JSON_SCHEMA["required"]) == {"entities", "relations"}


def test_schema_omits_events_for_cost_optimization() -> None:
    """Hackathon cost optimization: events are never requested from Gemini
    since EventRecord/EventParticipation never reach retrieval (see the
    architectural verification on record). WindowExtraction.events
    (schema.py) still defaults to () so this is a wire-format change only,
    not a domain-model change."""
    assert "events" not in EXTRACTION_JSON_SCHEMA["properties"]
    assert "events" not in EXTRACTION_JSON_SCHEMA["required"]


def test_schema_entity_type_enum_includes_object() -> None:
    entity_type_enum = EXTRACTION_JSON_SCHEMA["properties"]["entities"]["items"]["properties"]["type"]["enum"]
    assert "OBJECT" in entity_type_enum
    assert set(entity_type_enum) == {"PERSON", "LOCATION", "ORGANIZATION", "OBJECT", "OTHER"}


def test_schema_relation_type_enum_excludes_co_occurs() -> None:
    """CO_OCCURS is the G1 builder's exclusive value; the LLM-facing schema
    must never offer it as an option."""
    relation_type_enum = EXTRACTION_JSON_SCHEMA["properties"]["relations"]["items"]["properties"]["relation_type"][
        "enum"
    ]
    assert "CO_OCCURS" not in relation_type_enum
    assert set(relation_type_enum) == {"FAMILY_OF", "ALLY_OF", "ENEMY_OF", "MEMBER_OF", "LOCATED_AT", "POSSESSES", "SAME_AS"}


def test_schema_every_object_has_additional_properties_false() -> None:
    """Strict structured-output mode requires this at every nesting level."""

    def _walk(node: object) -> list[dict]:
        objects = []
        if isinstance(node, dict):
            if node.get("type") == "object" or (
                isinstance(node.get("type"), list) and "object" in node.get("type", [])
            ):
                objects.append(node)
            for value in node.values():
                objects.extend(_walk(value))
        elif isinstance(node, list):
            for item in node:
                objects.extend(_walk(item))
        return objects

    for obj in _walk(EXTRACTION_JSON_SCHEMA):
        assert obj.get("additionalProperties") is False, f"object missing additionalProperties=False: {obj}"
