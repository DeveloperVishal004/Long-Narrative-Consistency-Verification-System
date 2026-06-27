"""Pure, deterministic graph-retrieval algorithm: entry resolution, bounded
BFS expansion, and explicit chunk scoring.

Every function here takes an EntityGraph and returns plain data (lists,
dicts of str -> float) -- no networkx types escape this module's callers
beyond the EntityGraph parameter itself, and no learned ranker or model
call is involved anywhere in this file, per the Phase 8 architecture
review's "explicit algorithms over LLM reasoning" requirement.
"""

from lncvs.graph.builder import EntityGraph
from lncvs.graph.config import GraphConfig
from lncvs.graph.extraction import extract_mentions


def resolve_entry_entities(graph: EntityGraph, query_text: str, config: GraphConfig) -> list[str]:
    """Stage G1 entry resolution: exact, case-insensitive canonical-name
    lookup only (no fuzzy fallback -- deferred to a later stage). A mention
    that resolves to nothing is silently dropped, never an error; zero
    entry entities is a valid result handled by the caller."""
    entry_ids: list[str] = []
    for mention in extract_mentions(query_text, config.min_entity_token_length):
        entity_id = graph.entity_id_by_name(mention)
        if entity_id is not None and entity_id not in entry_ids:
            entry_ids.append(entity_id)
    return entry_ids


def score_chunks(graph: EntityGraph, entry_entity_ids: list[str], max_hops: int) -> dict[str, float]:
    """Explicit, deterministic chunk-scoring formula:

        score(chunk) = sum over every visited entity's provenance chunk of
                       (discovery_weight / (1 + hop_distance))

    Entry entities (hop 0) anchor via their own EntityRecord.provenance
    with a discovery_weight of 1. An entity first reached at hop >= 1
    anchors via its own full EntityRecord.provenance (not just the
    traversed edge's provenance), weighted by the EntityRelation.weight of
    the edge that discovered it. This is the deliberate, central
    multi-hop behavior: expanding from "London" to its co-occurring
    neighbor "John" is what lets John's *other* mentions (e.g. a chunk
    about John that never mentions London) surface as candidate evidence
    -- scoring only the edge's own provenance would collapse graph
    retrieval to plain co-occurrence lookup with no multi-hop value.

    If the same neighbor is reachable via more than one edge within a
    single hop, the discovering edge is the first encountered while
    iterating the (deterministically sorted) frontier -- the neighbor's
    provenance is anchored once per hop, never once per incoming edge, to
    avoid double-counting the same chunk set.

    Bounded BFS: visits each entity at most once, expands at most
    max_hops steps from the entry set, neighbors visited in ascending
    entity_id order (via EntityGraph.neighbor_relations) for fully
    deterministic iteration regardless of dict/set iteration order.
    """
    chunk_scores: dict[str, float] = {}
    visited = set(entry_entity_ids)

    for entity_id in entry_entity_ids:
        record = graph.entity(entity_id)
        for provenance in record.provenance:
            chunk_scores[provenance.chunk_id] = chunk_scores.get(provenance.chunk_id, 0.0) + 1.0

    frontier = set(entry_entity_ids)
    for hop in range(1, max_hops + 1):
        discovery_weight: dict[str, int] = {}
        for entity_id in sorted(frontier):
            for relation in graph.neighbor_relations(entity_id):
                neighbor_id = (
                    relation.object_entity_id
                    if relation.subject_entity_id == entity_id
                    else relation.subject_entity_id
                )
                if neighbor_id in visited or neighbor_id in discovery_weight:
                    continue
                discovery_weight[neighbor_id] = relation.weight

        if not discovery_weight:
            break

        for neighbor_id, weight in discovery_weight.items():
            record = graph.entity(neighbor_id)
            contribution = weight / (1 + hop)
            for provenance in record.provenance:
                chunk_scores[provenance.chunk_id] = chunk_scores.get(provenance.chunk_id, 0.0) + contribution

        visited |= set(discovery_weight)
        frontier = set(discovery_weight)

    return chunk_scores


def rank_chunks(chunk_scores: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    """Rank (chunk_id, score) pairs best-first, ties broken deterministically
    by ascending chunk_id, capped at top_k."""
    ranked = sorted(chunk_scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]
