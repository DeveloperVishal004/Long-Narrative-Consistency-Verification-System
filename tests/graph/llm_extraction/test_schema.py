"""Raw extraction DTO validation: local_id patterns, closed vocabularies,
evidence_quotes non-emptiness, frozen-ness, and the RelationType
CO_OCCURS/G2 separation."""

import pytest
from pydantic import ValidationError

from lncvs.graph.llm_extraction.schema import (
    RawEntityMention,
    RawEvent,
    RawParticipant,
    RawRelation,
    RawTemporal,
    WindowExtraction,
)
from lncvs.schemas import EntityType, ParticipantRole, RelationType, TemporalKind


def _entity(local_id: str = "e1", entity_type: EntityType = EntityType.PERSON) -> RawEntityMention:
    return RawEntityMention(local_id=local_id, name="John", type=entity_type, evidence_quotes=("John ran.",))


def test_raw_entity_mention_accepts_object_type() -> None:
    entity = _entity(entity_type=EntityType.OBJECT)
    assert entity.type is EntityType.OBJECT


def test_raw_entity_mention_rejects_malformed_local_id() -> None:
    with pytest.raises(ValidationError):
        RawEntityMention(local_id="entity1", name="John", type=EntityType.PERSON, evidence_quotes=("q",))


def test_raw_entity_mention_requires_at_least_one_evidence_quote() -> None:
    with pytest.raises(ValidationError):
        RawEntityMention(local_id="e1", name="John", type=EntityType.PERSON, evidence_quotes=())


def test_raw_entity_mention_is_frozen() -> None:
    entity = _entity()
    with pytest.raises(ValidationError):
        entity.name = "Someone Else"


def test_raw_relation_accepts_g2_relation_types() -> None:
    relation = RawRelation(
        subject_local_id="e1", object_local_id="e2", relation_type=RelationType.FAMILY_OF, evidence_quotes=("q",)
    )
    assert relation.relation_type is RelationType.FAMILY_OF


def test_raw_relation_rejects_co_occurs() -> None:
    """CO_OCCURS is reserved for the G1 deterministic builder; the LLM
    extractor must never emit it, even though it's a valid RelationType
    member overall."""
    with pytest.raises(ValueError):
        RawRelation(
            subject_local_id="e1", object_local_id="e2", relation_type=RelationType.CO_OCCURS, evidence_quotes=("q",)
        )


def test_raw_relation_accepts_non_conforming_local_id_for_downstream_quarantine() -> None:
    """A malformed local_id reference (e.g. a hallucinated 'OTHER') is not
    rejected here -- lncvs.graph.construction resolves references against
    the window's known entities and quarantines dangling ones there."""
    relation = RawRelation(
        subject_local_id="OTHER", object_local_id="e2", relation_type=RelationType.ALLY_OF, evidence_quotes=("q",)
    )
    assert relation.subject_local_id == "OTHER"


def test_raw_event_requires_at_least_one_participant() -> None:
    with pytest.raises(ValidationError):
        RawEvent(local_id="v1", predicate="marry", participants=(), evidence_quotes=("q",))


def test_raw_event_accepts_null_temporal() -> None:
    event = RawEvent(
        local_id="v1",
        predicate="marry",
        participants=(RawParticipant(entity_local_id="e1", role=ParticipantRole.AGENT),),
        temporal=None,
        evidence_quotes=("q",),
    )
    assert event.temporal is None


def test_raw_event_accepts_populated_temporal() -> None:
    event = RawEvent(
        local_id="v1",
        predicate="imprison",
        participants=(RawParticipant(entity_local_id="e1", role=ParticipantRole.PATIENT),),
        temporal=RawTemporal(time_expression="in 1815", kind=TemporalKind.ABSOLUTE),
        evidence_quotes=("q",),
    )
    assert event.temporal.time_expression == "in 1815"
    assert event.temporal.kind is TemporalKind.ABSOLUTE


def test_raw_event_rejects_malformed_local_id() -> None:
    with pytest.raises(ValidationError):
        RawEvent(
            local_id="event1",
            predicate="marry",
            participants=(RawParticipant(entity_local_id="e1", role=ParticipantRole.AGENT),),
            evidence_quotes=("q",),
        )


def test_window_extraction_defaults_to_empty_for_all_fields() -> None:
    extraction = WindowExtraction()
    assert extraction.entities == ()
    assert extraction.relations == ()
    assert extraction.events == ()


def test_window_extraction_round_trips_full_payload() -> None:
    extraction = WindowExtraction(
        entities=(_entity(),),
        relations=(
            RawRelation(
                subject_local_id="e1", object_local_id="e2", relation_type=RelationType.ALLY_OF, evidence_quotes=("q",)
            ),
        ),
        events=(
            RawEvent(
                local_id="v1",
                predicate="meet",
                participants=(RawParticipant(entity_local_id="e1", role=ParticipantRole.AGENT),),
                evidence_quotes=("q",),
            ),
        ),
    )
    assert len(extraction.entities) == 1
    assert len(extraction.relations) == 1
    assert len(extraction.events) == 1
