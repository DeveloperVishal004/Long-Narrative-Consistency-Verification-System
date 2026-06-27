"""GraphIndex: Indexer-protocol conformance, RetrievedEvidence shape,
multi-hop chunk surfacing end-to-end, determinism, and zero-entry handling."""

import pytest

from lncvs.graph import GraphConfig, GraphIndex
from lncvs.indexing import Indexer
from lncvs.schemas import RetrievalSource
from tests.graph.fakes import make_chunk

CHUNK_ARM = make_chunk("chunk-arm", "John lost his left arm in an accident in 2010.")
CHUNK_LONDON = make_chunk("chunk-london", "John moved to London in 2012.")


def test_graph_index_satisfies_indexer_protocol() -> None:
    assert isinstance(GraphIndex(), Indexer)


def test_query_before_index_raises() -> None:
    with pytest.raises(ValueError):
        GraphIndex().query("John", top_k=5)


def test_index_rejects_empty_chunk_list() -> None:
    with pytest.raises(ValueError):
        GraphIndex().index([])


def test_query_rejects_non_positive_top_k() -> None:
    index = GraphIndex()
    index.index([CHUNK_ARM, CHUNK_LONDON])
    with pytest.raises(ValueError):
        index.query("John", top_k=0)


def test_query_returns_empty_list_for_unresolvable_entry() -> None:
    index = GraphIndex()
    index.index([CHUNK_ARM, CHUNK_LONDON])
    assert index.query("Mars was distant.", top_k=5) == []


def test_query_returns_graph_sourced_evidence_ranked_best_first() -> None:
    index = GraphIndex()
    index.index([CHUNK_ARM, CHUNK_LONDON])

    results = index.query("John played a complex two-handed piano piece at a pub in London.", top_k=5)

    assert {r.chunk_id for r in results} == {"chunk-arm", "chunk-london"}
    assert all(r.source is RetrievalSource.GRAPH for r in results)
    assert [r.rank for r in results] == sorted(r.rank for r in results)
    assert results[0].chunk_id == "chunk-london"  # both entries anchor here
    assert results[0].raw_score >= results[1].raw_score


def test_query_one_hop_from_london_alone_surfaces_arm_chunk() -> None:
    """The multi-hop acceptance case: a query mentioning only "London"
    still retrieves chunk-arm via the John neighbor."""
    index = GraphIndex(GraphConfig(max_hops=1))
    index.index([CHUNK_ARM, CHUNK_LONDON])

    results = index.query("Tell me about London.", top_k=5)

    assert {r.chunk_id for r in results} == {"chunk-arm", "chunk-london"}


def test_query_is_deterministic_across_independent_indexes() -> None:
    index_a = GraphIndex()
    index_a.index([CHUNK_ARM, CHUNK_LONDON])
    index_b = GraphIndex()
    index_b.index([CHUNK_ARM, CHUNK_LONDON])

    query = "John played a piano piece in London."
    results_a = index_a.query(query, top_k=5)
    results_b = index_b.query(query, top_k=5)

    assert [(r.evidence_id, r.chunk_id, r.rank, r.raw_score) for r in results_a] == [
        (r.evidence_id, r.chunk_id, r.rank, r.raw_score) for r in results_b
    ]
