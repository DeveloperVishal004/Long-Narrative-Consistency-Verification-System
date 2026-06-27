"""build_graph: relation/event resolution through Slice 5's mapping,
dangling-reference quarantine, directional-relation preservation, and
deterministic aggregation."""

from lncvs.graph.construction.service import build_graph
from lncvs.graph.entity_resolution.models import EntityResolutionResult
from lncvs.graph.llm_extraction.schema import RawEvent, RawParticipant, RawRelation
from lncvs.graph.provenance.matching import MatchTier, QuoteMatch
from lncvs.graph.provenance.models import ResolvedFact
from lncvs.schemas import EntityRecord, EntityType, ParticipantRole, Provenance, RelationType, TemporalKind

ENTITY_A = EntityRecord(
    entity_id="a" * 16, canonical_name="John", entity_type=EntityType.PERSON, provenance=(Provenance(chunk_id="c1", char_start=0, char_end=4),)
)
ENTITY_B = EntityRecord(
    entity_id="b" * 16, canonical_name="London", entity_type=EntityType.LOCATION, provenance=(Provenance(chunk_id="c1", char_start=10, char_end=16),)
)

RESOLUTION = EntityResolutionResult(
    entities=(ENTITY_A, ENTITY_B),
    local_to_global={(0, None, "e1"): ENTITY_A.entity_id, (0, None, "e2"): ENTITY_B.entity_id},
)


def _relation_fact(
    chapter_index: int,
    window_index: int | None,
    subject_local_id: str,
    object_local_id: str,
    relation_type: RelationType,
    chunk_id: str,
) -> ResolvedFact:
    raw = RawRelation(subject_local_id=subject_local_id, object_local_id=object_local_id, relation_type=relation_type, evidence_quotes=("q",))
    provenance = (Provenance(chunk_id=chunk_id, char_start=0, char_end=5),)
    quote_match = QuoteMatch(quote="q", tier=MatchTier.EXACT, char_start=0, char_end=1)
    return ResolvedFact(raw=raw, chapter_index=chapter_index, window_index=window_index, provenance=provenance, quote_matches=(quote_match,))


def _event_fact(
    chapter_index: int, window_index: int | None, predicate: str, participants: tuple[RawParticipant, ...], chunk_id: str, char_start: int = 0
) -> ResolvedFact:
    raw = RawEvent(local_id="v1", predicate=predicate, participants=participants, evidence_quotes=("q",))
    provenance = (Provenance(chunk_id=chunk_id, char_start=char_start, char_end=char_start + 5),)
    quote_match = QuoteMatch(quote="q", tier=MatchTier.EXACT, char_start=0, char_end=1)
    return ResolvedFact(raw=raw, chapter_index=chapter_index, window_index=window_index, provenance=provenance, quote_matches=(quote_match,))


def test_relation_direction_is_preserved_exactly_as_extracted() -> None:
    """ENTITY_A.entity_id ("aaa...") < ENTITY_B.entity_id ("bbb...")
    lexicographically. This fact's true subject is e2 (ENTITY_B, the
    larger ID) and true object is e1 (ENTITY_A, the smaller ID) --
    exactly the case an ascending-ID sort would have silently inverted."""
    fact = _relation_fact(0, None, "e2", "e1", RelationType.POSSESSES, "c1")  # London POSSESSES John
    graph = build_graph(RESOLUTION, [fact], [])

    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert relation.subject_entity_id == ENTITY_B.entity_id
    assert relation.object_entity_id == ENTITY_A.entity_id
    assert relation.relation_type is RelationType.POSSESSES


def test_relation_with_dangling_subject_is_quarantined() -> None:
    fact = _relation_fact(0, None, "e99", "e1", RelationType.ALLY_OF, "c1")
    graph = build_graph(RESOLUTION, [fact], [])
    assert len(graph.relations) == 0
    assert len(graph.rejected_relations) == 1
    assert "dangling" in graph.rejected_relations[0].reason


def test_self_relation_is_quarantined() -> None:
    fact = _relation_fact(0, None, "e1", "e1", RelationType.ALLY_OF, "c1")
    graph = build_graph(RESOLUTION, [fact], [])
    assert len(graph.relations) == 0
    assert len(graph.rejected_relations) == 1
    assert "self-relation" in graph.rejected_relations[0].reason


def test_duplicate_relations_aggregate_weight_and_provenance() -> None:
    facts = [
        _relation_fact(0, None, "e1", "e2", RelationType.LOCATED_AT, "c1"),
        _relation_fact(0, None, "e1", "e2", RelationType.LOCATED_AT, "c2"),
    ]
    graph = build_graph(RESOLUTION, facts, [])
    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert relation.weight == 2
    assert {p.chunk_id for p in relation.provenance} == {"c1", "c2"}


def test_different_relation_types_between_same_pair_stay_separate_edges() -> None:
    facts = [
        _relation_fact(0, None, "e1", "e2", RelationType.LOCATED_AT, "c1"),
        _relation_fact(0, None, "e1", "e2", RelationType.ALLY_OF, "c1"),
    ]
    graph = build_graph(RESOLUTION, facts, [])
    assert len(graph.relations) == 2
    assert {r.relation_type for r in graph.relations} == {RelationType.LOCATED_AT, RelationType.ALLY_OF}


def test_event_resolves_with_correct_participant_roles() -> None:
    fact = _event_fact(0, None, "lose", (RawParticipant(entity_local_id="e1", role=ParticipantRole.PATIENT),), "c1")
    graph = build_graph(RESOLUTION, [], [fact])

    assert len(graph.events) == 1
    assert graph.events[0].predicate == "lose"
    assert len(graph.participations) == 1
    participation = graph.participations[0]
    assert participation.entity_id == ENTITY_A.entity_id
    assert participation.role is ParticipantRole.PATIENT


def test_event_with_dangling_participant_is_quarantined() -> None:
    fact = _event_fact(0, None, "lose", (RawParticipant(entity_local_id="e99", role=ParticipantRole.PATIENT),), "c1")
    graph = build_graph(RESOLUTION, [], [fact])
    assert len(graph.events) == 0
    assert len(graph.rejected_events) == 1
    assert "dangling" in graph.rejected_events[0].reason


def test_identical_event_extracted_twice_dedupes_to_one_event_record() -> None:
    facts = [
        _event_fact(0, None, "lose", (RawParticipant(entity_local_id="e1", role=ParticipantRole.PATIENT),), "c1", char_start=0),
        _event_fact(0, 1, "lose", (RawParticipant(entity_local_id="e1", role=ParticipantRole.PATIENT),), "c1", char_start=0),
    ]
    graph = build_graph(RESOLUTION, [], facts)
    assert len(graph.events) == 1
    assert len(graph.participations) == 1
    assert graph.participations[0].weight == 1  # same chunk_id in both, dedupes to 1 distinct chunk


def test_event_with_no_temporal_gets_temporal_kind_none() -> None:
    fact = _event_fact(0, None, "lose", (RawParticipant(entity_local_id="e1", role=ParticipantRole.PATIENT),), "c1")
    graph = build_graph(RESOLUTION, [], [fact])
    assert graph.events[0].temporal_kind is TemporalKind.NONE
    assert graph.events[0].temporal_expression is None


def test_build_graph_is_deterministic_across_calls() -> None:
    relation_facts = [_relation_fact(0, None, "e1", "e2", RelationType.LOCATED_AT, "c1")]
    event_facts = [_event_fact(0, None, "lose", (RawParticipant(entity_local_id="e1", role=ParticipantRole.PATIENT),), "c1")]

    first = build_graph(RESOLUTION, relation_facts, event_facts)
    second = build_graph(RESOLUTION, relation_facts, event_facts)

    assert first.fingerprint == second.fingerprint
    assert first.relations == second.relations
    assert first.events == second.events
    assert first.participations == second.participations


def test_fingerprint_changes_when_relation_content_changes() -> None:
    base = build_graph(RESOLUTION, [_relation_fact(0, None, "e1", "e2", RelationType.LOCATED_AT, "c1")], [])
    changed = build_graph(RESOLUTION, [_relation_fact(0, None, "e1", "e2", RelationType.ALLY_OF, "c1")], [])
    assert base.fingerprint != changed.fingerprint


def test_empty_input_produces_empty_graph_with_stable_fingerprint() -> None:
    empty_resolution = EntityResolutionResult(entities=(), local_to_global={})
    graph = build_graph(empty_resolution, [], [])
    assert graph.entities == ()
    assert graph.relations == ()
    assert graph.events == ()
    assert graph.fingerprint == build_graph(empty_resolution, [], []).fingerprint
