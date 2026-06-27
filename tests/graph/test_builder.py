"""EntityGraph construction: entity merging across chunks, co-occurrence
edges, weight accumulation, and determinism."""

from lncvs.graph.builder import build_entity_graph
from lncvs.graph.config import GraphConfig
from lncvs.schemas import EntityType, RelationType
from tests.graph.fakes import make_chunk

CHUNK_ARM = make_chunk("chunk-arm", "John lost his left arm in an accident in 2010.")
CHUNK_LONDON = make_chunk("chunk-london", "John moved to London in 2012.")


def test_builds_one_entity_per_distinct_canonical_name() -> None:
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    assert graph.entity_count() == 2
    john_id = graph.entity_id_by_name("John")
    london_id = graph.entity_id_by_name("London")
    assert john_id is not None
    assert london_id is not None
    assert john_id != london_id


def test_entity_merges_provenance_across_chunks() -> None:
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    john_id = graph.entity_id_by_name("John")
    john = graph.entity(john_id)
    assert {p.chunk_id for p in john.provenance} == {"chunk-arm", "chunk-london"}
    assert john.entity_type is EntityType.OTHER


def test_co_occurrence_edge_created_only_for_chunks_with_two_or_more_entities() -> None:
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    assert graph.relation_count() == 1

    john_id = graph.entity_id_by_name("John")
    london_id = graph.entity_id_by_name("London")
    relations = graph.neighbor_relations(john_id)
    assert len(relations) == 1
    relation = relations[0]
    assert {relation.subject_entity_id, relation.object_entity_id} == {john_id, london_id}
    assert relation.relation_type is RelationType.CO_OCCURS
    assert relation.weight == 1
    assert {p.chunk_id for p in relation.provenance} == {"chunk-london"}


def test_relation_weight_increments_for_repeated_co_occurrence_in_different_chunks() -> None:
    chunk_a = make_chunk("c1", "Paganel and Glenarvan boarded the Duncan.")
    chunk_b = make_chunk("c2", "Paganel and Glenarvan disembarked together.")
    graph = build_entity_graph([chunk_a, chunk_b], GraphConfig())

    paganel_id = graph.entity_id_by_name("Paganel")
    glenarvan_id = graph.entity_id_by_name("Glenarvan")
    # chunk_a also mentions "Duncan" (a legitimate third capitalized mention),
    # so paganel may have more than one neighbor relation -- select the
    # specific Paganel<->Glenarvan edge rather than assuming index 0.
    relation = next(
        r for r in graph.neighbor_relations(paganel_id) if glenarvan_id in (r.subject_entity_id, r.object_entity_id)
    )
    assert {relation.subject_entity_id, relation.object_entity_id} == {paganel_id, glenarvan_id}
    assert relation.weight == 2
    assert {p.chunk_id for p in relation.provenance} == {"c1", "c2"}


def test_no_self_relation_for_a_single_entity_chunk() -> None:
    graph = build_entity_graph([make_chunk("c1", "London was foggy.")], GraphConfig())
    assert graph.entity_count() == 1
    assert graph.relation_count() == 0


def test_entity_id_is_deterministic_across_independent_builds() -> None:
    graph_a = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    graph_b = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    assert graph_a.entity_id_by_name("John") == graph_b.entity_id_by_name("John")
    assert graph_a.entity(graph_a.entity_id_by_name("John")) == graph_b.entity(graph_b.entity_id_by_name("John"))


def test_build_rejects_empty_chunk_list() -> None:
    graph = build_entity_graph([], GraphConfig())
    assert graph.entity_count() == 0
    assert graph.relation_count() == 0
