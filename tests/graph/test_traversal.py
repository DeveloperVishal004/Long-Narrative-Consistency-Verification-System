"""Entry resolution, bounded BFS chunk scoring, and ranking -- pure functions,
tested independently of GraphIndex/GraphRetriever."""

from lncvs.graph.builder import build_entity_graph
from lncvs.graph.config import GraphConfig
from lncvs.graph.traversal import rank_chunks, resolve_entry_entities, score_chunks
from tests.graph.fakes import make_chunk

CHUNK_ARM = make_chunk("chunk-arm", "John lost his left arm in an accident in 2010.")
CHUNK_LONDON = make_chunk("chunk-london", "John moved to London in 2012.")


def test_resolve_entry_entities_exact_match() -> None:
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    entry_ids = resolve_entry_entities(graph, "John played piano in London.", GraphConfig())
    assert len(entry_ids) == 2
    assert set(entry_ids) == {graph.entity_id_by_name("John"), graph.entity_id_by_name("London")}


def test_resolve_entry_entities_returns_empty_for_unknown_mentions() -> None:
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    assert resolve_entry_entities(graph, "Mars was distant.", GraphConfig()) == []


def test_score_chunks_entry_only_anchors_via_own_provenance() -> None:
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    john_id = graph.entity_id_by_name("John")

    scores = score_chunks(graph, [john_id], max_hops=0)
    assert scores == {"chunk-arm": 1.0, "chunk-london": 1.0}


def test_score_chunks_one_hop_pulls_in_neighbors_other_mention_chunk() -> None:
    """The central multi-hop case: starting from "London" alone, 1-hop
    expansion reaches "John" and surfaces chunk-arm -- a chunk that
    mentions neither London nor co-occurs with it -- via John's own
    provenance. This is the behavior that gives graph retrieval value
    beyond plain co-occurrence lookup."""
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    london_id = graph.entity_id_by_name("London")

    scores = score_chunks(graph, [london_id], max_hops=1)
    assert scores["chunk-london"] == 1.0 + 1 / 2  # entry anchor + John's hop-1 provenance (he's also in chunk-london)
    assert scores["chunk-arm"] == 1 / 2


def test_score_chunks_does_not_revisit_entry_entities() -> None:
    graph = build_entity_graph([CHUNK_ARM, CHUNK_LONDON], GraphConfig())
    john_id = graph.entity_id_by_name("John")
    london_id = graph.entity_id_by_name("London")

    scores = score_chunks(graph, [john_id, london_id], max_hops=1)
    assert scores["chunk-london"] == 1.0 + 1.0
    assert scores["chunk-arm"] == 1.0


def test_rank_chunks_orders_best_first_with_deterministic_tie_break() -> None:
    ranked = rank_chunks({"b": 1.0, "a": 1.0, "c": 2.0}, top_k=5)
    assert ranked == [("c", 2.0), ("a", 1.0), ("b", 1.0)]


def test_rank_chunks_respects_top_k() -> None:
    ranked = rank_chunks({"a": 3.0, "b": 2.0, "c": 1.0}, top_k=2)
    assert ranked == [("a", 3.0), ("b", 2.0)]
