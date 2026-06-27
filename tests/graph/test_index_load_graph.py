"""GraphIndex.load_graph: querying a pre-built (G2) EntityGraph exactly
like an index()-built (G1) one."""

import pytest

from lncvs.graph.builder import EntityGraph
from lncvs.graph.index import GraphIndex
from lncvs.schemas import EntityRecord, EntityType, Provenance, RelationType
from lncvs.schemas import EntityRelation as Relation
from tests.graph.fakes import make_chunk

CHUNK_ARM = make_chunk("chunk-arm", "John lost his left arm in an accident in 2010.")
CHUNK_LONDON = make_chunk("chunk-london", "John moved to London in 2012.")


def _build_graph() -> EntityGraph:
    entities = (
        EntityRecord(entity_id="e-john", canonical_name="John", entity_type=EntityType.PERSON, provenance=(
            Provenance(chunk_id="chunk-arm", char_start=0, char_end=48),
            Provenance(chunk_id="chunk-london", char_start=0, char_end=29),
        )),
        EntityRecord(entity_id="e-london", canonical_name="London", entity_type=EntityType.LOCATION, provenance=(
            Provenance(chunk_id="chunk-london", char_start=0, char_end=29),
        )),
    )
    relations = (
        Relation(subject_entity_id="e-john", object_entity_id="e-london", relation_type=RelationType.LOCATED_AT, weight=1, provenance=(
            Provenance(chunk_id="chunk-london", char_start=0, char_end=29),
        )),
    )
    return EntityGraph.from_records(entities, relations)


def test_load_graph_then_query_returns_real_chunk_ids() -> None:
    index = GraphIndex()
    index.load_graph(_build_graph(), [CHUNK_ARM, CHUNK_LONDON])

    results = index.query("John played a piano piece in London.", top_k=5)
    assert {r.chunk_id for r in results} == {"chunk-arm", "chunk-london"}


def test_load_graph_rejects_empty_chunk_list() -> None:
    index = GraphIndex()
    with pytest.raises(ValueError):
        index.load_graph(_build_graph(), [])


def test_query_before_load_graph_raises() -> None:
    index = GraphIndex()
    with pytest.raises(ValueError):
        index.query("John", top_k=5)
