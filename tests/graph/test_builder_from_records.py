"""EntityGraph.from_records: building a traversable graph directly from
already-resolved G2 EntityRecord/EntityRelation content."""

from lncvs.graph.builder import EntityGraph
from lncvs.schemas import EntityRecord, EntityType, Provenance, RelationType
from lncvs.schemas import EntityRelation as Relation


def _entity(entity_id: str, name: str, chunk_id: str = "c1") -> EntityRecord:
    return EntityRecord(
        entity_id=entity_id, canonical_name=name, entity_type=EntityType.PERSON, provenance=(Provenance(chunk_id=chunk_id, char_start=0, char_end=10),)
    )


def _relation(subject_id: str, object_id: str, relation_type: RelationType, chunk_id: str = "c1", weight: int = 1) -> Relation:
    return Relation(
        subject_entity_id=subject_id, object_entity_id=object_id, relation_type=relation_type, weight=weight, provenance=(Provenance(chunk_id=chunk_id, char_start=0, char_end=10),)
    )


def test_builds_nodes_with_correct_name_lookup() -> None:
    entities = (_entity("e1", "John"), _entity("e2", "London"))
    graph = EntityGraph.from_records(entities, ())

    assert graph.entity_count() == 2
    assert graph.entity_id_by_name("John") == "e1"
    assert graph.entity_id_by_name("london") == "e2"  # case-insensitive


def test_single_relation_preserves_exact_direction() -> None:
    entities = (_entity("e1", "John"), _entity("e2", "London"))
    relations = (_relation("e2", "e1", RelationType.POSSESSES),)  # London POSSESSES John
    graph = EntityGraph.from_records(entities, relations)

    relation = graph.neighbor_relations("e1")[0]
    assert relation.subject_entity_id == "e2"
    assert relation.object_entity_id == "e1"


def test_two_relations_same_direction_between_same_pair_merge() -> None:
    entities = (_entity("e1", "John"), _entity("e2", "London"))
    relations = (
        _relation("e1", "e2", RelationType.LOCATED_AT, chunk_id="c1"),
        _relation("e1", "e2", RelationType.LOCATED_AT, chunk_id="c2"),
    )
    graph = EntityGraph.from_records(entities, relations)

    assert graph.relation_count() == 1
    relation = graph.neighbor_relations("e1")[0]
    assert relation.weight == 2
    assert {p.chunk_id for p in relation.provenance} == {"c1", "c2"}


def test_swapped_direction_relations_between_same_pair_are_merged_not_overwritten() -> None:
    """The bug this test pins down: A->B and B->A relations between the
    same two nodes must both survive (merged), never silently overwrite
    each other via networkx's undirected add_edge()."""
    entities = (_entity("e1", "John"), _entity("e2", "London"))
    relations = (
        _relation("e1", "e2", RelationType.POSSESSES, chunk_id="c1"),  # John POSSESSES London
        _relation("e2", "e1", RelationType.ALLY_OF, chunk_id="c2"),  # London ALLY_OF John
    )
    graph = EntityGraph.from_records(entities, relations)

    assert graph.relation_count() == 1  # one traversal edge, but content from both survives
    relation = graph.neighbor_relations("e1")[0]
    assert relation.weight == 2
    assert {p.chunk_id for p in relation.provenance} == {"c1", "c2"}


def test_different_relation_types_between_different_pairs_stay_distinct() -> None:
    entities = (_entity("e1", "John"), _entity("e2", "London"), _entity("e3", "Mary"))
    relations = (
        _relation("e1", "e2", RelationType.LOCATED_AT),
        _relation("e1", "e3", RelationType.FAMILY_OF),
    )
    graph = EntityGraph.from_records(entities, relations)

    assert graph.relation_count() == 2
    assert len(graph.neighbor_relations("e1")) == 2


def test_empty_input_produces_empty_graph() -> None:
    graph = EntityGraph.from_records((), ())
    assert graph.entity_count() == 0
    assert graph.relation_count() == 0


def test_is_deterministic() -> None:
    entities = (_entity("e1", "John"), _entity("e2", "London"))
    relations = (_relation("e1", "e2", RelationType.LOCATED_AT),)

    first = EntityGraph.from_records(entities, relations)
    second = EntityGraph.from_records(entities, relations)
    assert first.entity("e1") == second.entity("e1")
    assert first.neighbor_relations("e1") == second.neighbor_relations("e1")
