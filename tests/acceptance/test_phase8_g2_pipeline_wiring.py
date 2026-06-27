"""Phase 8 / G2 end-to-end wiring acceptance test: the complete
construction pipeline (segmentation -> extraction -> provenance ->
entity resolution -> construction -> EntityGraph -> GraphIndex ->
GraphRetriever), offline and deterministic via FakeStructuredLLMClient
(no real API calls, no cost).

Proves the full chain is wired correctly end-to-end on the §14 dummy-case
text, including a real entity, a real directional relation, and a real
event surviving all the way through to a working graph query.
"""

from lncvs.graph.builder import EntityGraph
from lncvs.graph.construction.pipeline import build_graph_for_novel
from lncvs.graph.index import GraphIndex
from lncvs.graph.llm_extraction.service import LLMWindowExtractor
from lncvs.schemas import DocumentChunk, EntityType, ParticipantRole, RelationType, TemporalKind
from tests.llm.fakes import FakeStructuredLLMClient

NARRATIVE = "John lost his left arm in an accident in 2010. John moved to London in 2012."
SPLIT_POINT = NARRATIVE.index("John moved to London")

CHUNKS = [
    DocumentChunk(chunk_id="chunk-arm", text=NARRATIVE[:SPLIT_POINT], char_start=0, char_end=SPLIT_POINT, source_id="dummy"),
    DocumentChunk(chunk_id="chunk-london", text=NARRATIVE[SPLIT_POINT:], char_start=SPLIT_POINT, char_end=len(NARRATIVE), source_id="dummy"),
]

EXTRACTION_RESPONSE = {
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


def test_full_pipeline_wiring_end_to_end() -> None:
    fake_client = FakeStructuredLLMClient(default_response=EXTRACTION_RESPONSE)
    extractor = LLMWindowExtractor(fake_client)

    constructed, entity_graph = build_graph_for_novel(NARRATIVE, CHUNKS, extractor)

    # Construction layer: entities, relation, event all survived.
    assert len(constructed.entities) == 2
    assert len(constructed.relations) == 1
    assert len(constructed.events) == 1
    assert len(constructed.participations) == 1
    assert constructed.rejected_relations == ()
    assert constructed.rejected_events == ()

    john = next(e for e in constructed.entities if e.canonical_name == "John")
    london = next(e for e in constructed.entities if e.canonical_name == "London")
    assert john.entity_type is EntityType.PERSON
    assert london.entity_type is EntityType.LOCATION

    relation = constructed.relations[0]
    assert relation.subject_entity_id == john.entity_id
    assert relation.object_entity_id == london.entity_id
    assert relation.relation_type is RelationType.LOCATED_AT

    event = constructed.events[0]
    assert event.predicate == "lose"
    assert event.temporal_kind is TemporalKind.ABSOLUTE
    assert event.temporal_expression == "in 2010"
    assert constructed.participations[0].role is ParticipantRole.PATIENT

    # EntityGraph: queryable directly.
    assert isinstance(entity_graph, EntityGraph)
    assert entity_graph.entity_id_by_name("John") == john.entity_id
    assert entity_graph.entity_id_by_name("London") == london.entity_id

    # GraphIndex/GraphRetriever: the actual retrieval surface.
    index = GraphIndex()
    index.load_graph(entity_graph, CHUNKS)

    results = index.query("John played a piano piece in London.", top_k=5)
    assert {r.chunk_id for r in results} == {"chunk-arm", "chunk-london"}


def test_pipeline_is_deterministic_across_independent_runs() -> None:
    extractor_a = LLMWindowExtractor(FakeStructuredLLMClient(default_response=EXTRACTION_RESPONSE))
    extractor_b = LLMWindowExtractor(FakeStructuredLLMClient(default_response=EXTRACTION_RESPONSE))

    constructed_a, _ = build_graph_for_novel(NARRATIVE, CHUNKS, extractor_a)
    constructed_b, _ = build_graph_for_novel(NARRATIVE, CHUNKS, extractor_b)

    assert constructed_a.fingerprint == constructed_b.fingerprint
    assert constructed_a.entities == constructed_b.entities
    assert constructed_a.relations == constructed_b.relations
