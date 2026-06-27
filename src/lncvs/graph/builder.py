"""Deterministic entity-graph construction.

build_entity_graph() is the only function in this module that imports
networkx -- the same source-isolation discipline ChromaIndex applies to
chromadb and BM25Index applies to rank_bm25. Construction is fully
deterministic: identical chunks always produce an identical EntityGraph
(same entity_ids, same edges, same provenance), which is what the
determinism tests in tests/graph/test_builder.py pin down.

Per the Phase 8 architecture review, Chunk is never a graph node -- chunk
identity here exists only as Provenance on EntityRecord/EntityRelation.
Provenance anchors to the whole chunk span (char_start/char_end of the
chunk itself), the same precision ChromaIndex/BM25Index already use for
RetrievedEvidence.provenance -- Stage G1's deterministic extractor does
not track a mention's exact in-chunk character offset.
"""

from itertools import combinations

import networkx as nx

from lncvs.graph.config import GraphConfig
from lncvs.graph.extraction import extract_mentions
from lncvs.graph.identity import make_entity_id
from lncvs.schemas import DocumentChunk, EntityRecord, EntityRelation, EntityType, Provenance, RelationType


class EntityGraph:
    """Thin wrapper around a networkx.Graph plus a canonical-name lookup index.

    networkx is confined entirely to this class -- no other module in
    lncvs.graph, and no module outside lncvs.graph, ever touches a
    networkx object directly. Node/edge attributes are always the typed
    EntityRecord/EntityRelation models from schemas/, never a raw dict, so
    crossing into this wrapper's public methods never crosses an untyped
    boundary even though networkx itself stores attributes as dicts
    internally.
    """

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._name_to_id: dict[str, str] = {}

    def upsert_entity(self, canonical_name: str, chunk: DocumentChunk) -> str:
        """Add a mention of canonical_name anchored to chunk, merging into an
        existing entity if one already exists for this exact canonical name
        (case-insensitive). Returns the entity_id."""
        lookup_key = canonical_name.lower()
        entity_id = self._name_to_id.get(lookup_key)
        new_provenance = Provenance(chunk_id=chunk.chunk_id, char_start=chunk.char_start, char_end=chunk.char_end)

        if entity_id is None:
            entity_id = make_entity_id(canonical_name, EntityType.OTHER.value)
            self._name_to_id[lookup_key] = entity_id
            record = EntityRecord(
                entity_id=entity_id,
                canonical_name=canonical_name,
                entity_type=EntityType.OTHER,
                provenance=(new_provenance,),
            )
            self._graph.add_node(entity_id, record=record)
            return entity_id

        existing: EntityRecord = self._graph.nodes[entity_id]["record"]
        if new_provenance not in existing.provenance:
            updated = existing.model_copy(update={"provenance": existing.provenance + (new_provenance,)})
            self._graph.nodes[entity_id]["record"] = updated
        return entity_id

    def upsert_relation(self, entity_id_a: str, entity_id_b: str, chunk: DocumentChunk) -> None:
        """Record a CO_OCCURS observation between two distinct entities in chunk,
        merging into an existing edge (incrementing weight, appending
        provenance) if one already exists for this entity pair."""
        if entity_id_a == entity_id_b:
            raise ValueError("Cannot relate an entity to itself")

        subject_id, object_id = sorted((entity_id_a, entity_id_b))
        new_provenance = Provenance(chunk_id=chunk.chunk_id, char_start=chunk.char_start, char_end=chunk.char_end)

        if not self._graph.has_edge(subject_id, object_id):
            relation = EntityRelation(
                subject_entity_id=subject_id,
                object_entity_id=object_id,
                relation_type=RelationType.CO_OCCURS,
                weight=1,
                provenance=(new_provenance,),
            )
            self._graph.add_edge(subject_id, object_id, record=relation)
            return

        existing: EntityRelation = self._graph.edges[subject_id, object_id]["record"]
        if new_provenance in existing.provenance:
            return
        updated = existing.model_copy(
            update={"provenance": existing.provenance + (new_provenance,), "weight": existing.weight + 1}
        )
        self._graph.edges[subject_id, object_id]["record"] = updated

    @classmethod
    def from_records(cls, entities: tuple[EntityRecord, ...], relations: tuple[EntityRelation, ...]) -> "EntityGraph":
        """Build an EntityGraph directly from already-resolved, globally-
        identified EntityRecord/EntityRelation content -- the Phase 8 / G2
        LLM-based construction pipeline's output (lncvs.graph.construction),
        as opposed to build_entity_graph()'s raw-text-mention upsert path.
        No merging happens here: G2 entities/relations are already fully
        resolved (lncvs.graph.entity_resolution), so this is pure assembly.

        Disclosed simplification: the underlying graph is a plain
        networkx.Graph (one edge per node pair, inherited from G1, which
        never has more than one relation type per pair). If G2 supplies
        more than one EntityRelation for the same (subject, object) pair
        with different relation_types, they are merged into a single
        traversal edge -- weight summed, provenance unioned, relation_type
        set to whichever sorts first by value -- for retrieval-traversal
        purposes only. The complete, untouched, per-type relation list
        remains available from ConstructedGraph.relations for any other
        purpose; only this BFS-facing graph view collapses parallel edges,
        a deliberate scope choice under the hackathon deadline rather than
        retrofitting networkx.MultiGraph into G1's frozen, tested class.
        """
        graph = cls()

        for entity in entities:
            graph._graph.add_node(entity.entity_id, record=entity)
            graph._name_to_id[entity.canonical_name.lower()] = entity.entity_id

        # Grouped by the *unordered* node pair, not the raw (subject, object)
        # tuple: networkx.Graph treats A-B and B-A as the same edge slot, so
        # two relations between the same two nodes with subject/object
        # swapped (e.g. "A POSSESSES B" extracted separately from
        # "B LOCATED_AT A") would otherwise silently overwrite each other
        # via add_edge() rather than being detected and merged.
        relations_by_pair: dict[tuple[str, str], list[EntityRelation]] = {}
        for relation in relations:
            pair = tuple(sorted((relation.subject_entity_id, relation.object_entity_id)))
            relations_by_pair.setdefault(pair, []).append(relation)

        for (subject_id, object_id), pair_relations in relations_by_pair.items():
            if len(pair_relations) == 1:
                only = pair_relations[0]
                graph._graph.add_edge(only.subject_entity_id, only.object_entity_id, record=only)
                continue

            ordered = sorted(pair_relations, key=lambda r: r.relation_type.value)
            merged_provenance = tuple(
                sorted({p for r in ordered for p in r.provenance}, key=lambda p: (p.chunk_id, p.char_start))
            )
            merged = ordered[0].model_copy(
                update={"weight": sum(r.weight for r in ordered), "provenance": merged_provenance}
            )
            graph._graph.add_edge(subject_id, object_id, record=merged)

        return graph

    def entity_id_by_name(self, canonical_name: str) -> str | None:
        """Exact, case-insensitive canonical-name lookup. Returns None if absent."""
        return self._name_to_id.get(canonical_name.lower())

    def entity(self, entity_id: str) -> EntityRecord:
        return self._graph.nodes[entity_id]["record"]

    def neighbor_relations(self, entity_id: str) -> list[EntityRelation]:
        """All EntityRelation edges directly touching entity_id, ascending by
        the neighbor's entity_id for deterministic iteration order."""
        relations = [self._graph.edges[entity_id, neighbor]["record"] for neighbor in self._graph.neighbors(entity_id)]
        return sorted(relations, key=lambda r: (r.subject_entity_id, r.object_entity_id))

    def entity_count(self) -> int:
        return self._graph.number_of_nodes()

    def relation_count(self) -> int:
        return self._graph.number_of_edges()


def build_entity_graph(chunks: list[DocumentChunk], config: GraphConfig) -> EntityGraph:
    """Build an EntityGraph deterministically from chunks.

    For each chunk: extract mentions, upsert one entity per mention
    (merging across chunks by exact case-insensitive canonical name), then
    add a CO_OCCURS relation for every unordered pair of distinct entities
    mentioned within that same chunk.
    """
    graph = EntityGraph()

    for chunk in chunks:
        mentions = extract_mentions(chunk.text, config.min_entity_token_length)
        entity_ids = [graph.upsert_entity(mention, chunk) for mention in mentions]

        for entity_id_a, entity_id_b in combinations(sorted(set(entity_ids)), 2):
            graph.upsert_relation(entity_id_a, entity_id_b, chunk)

    return graph
