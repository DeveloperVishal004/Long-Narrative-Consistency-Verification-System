"""GraphIndex -- the only Indexer-protocol implementation in lncvs.graph.

Mirrors ChromaIndex/BM25Index exactly: index() builds the backend (here, an
EntityGraph) from chunks, query() returns a ranked list[RetrievedEvidence].
GraphRetriever (retriever.py) is the thin Retriever-protocol wrapper that
delegates to this class, the same split SemanticRetriever/ChromaIndex and
BM25Retriever/BM25Index already use.

Per the "graph never leaves the chunk-ID space" invariant, query() returns
only chunk_ids (wrapped in RetrievedEvidence) -- never node IDs, paths, or
traversal objects as the evidence payload itself.
"""

import logging

from lncvs.graph.builder import EntityGraph, build_entity_graph
from lncvs.graph.config import GraphConfig
from lncvs.graph.identity import make_raw_evidence_id
from lncvs.graph.traversal import rank_chunks, resolve_entry_entities, score_chunks
from lncvs.schemas import DocumentChunk, Provenance, RetrievalSource, RetrievedEvidence

logger = logging.getLogger(__name__)


class GraphIndex:
    """Graph-backed Indexer: builds a deterministic EntityGraph over chunks,
    then answers queries via exact entry resolution + bounded BFS + explicit
    chunk scoring (lncvs.graph.traversal). No persistence, no NLP model --
    matching ChromaIndex's documented no-persistence limitation and Stage
    G1's "zero models" scope respectively.
    """

    def __init__(self, config: GraphConfig | None = None) -> None:
        self._config = config or GraphConfig()
        self._graph: EntityGraph | None = None
        self._chunks_by_id: dict[str, DocumentChunk] = {}

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Build the entity graph over chunks via G1's deterministic,
        text-extraction-based builder."""
        if not chunks:
            raise ValueError("Cannot index an empty list of chunks")

        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._graph = build_entity_graph(chunks, self._config)
        logger.info(
            "Built entity graph: %d entities, %d relations, from %d chunks",
            self._graph.entity_count(),
            self._graph.relation_count(),
            len(chunks),
        )

    def load_graph(self, entity_graph: EntityGraph, chunks: list[DocumentChunk]) -> None:
        """Load an already-built EntityGraph (e.g. from Phase 8 / G2's LLM-
        based construction pipeline via EntityGraph.from_records()) instead
        of building one from chunks via index(). chunks must be the same
        chunk-ID space the graph's provenance was resolved against -- this
        is what query() uses to recover chunk text for RetrievedEvidence.

        This is the second, alternate way to populate a GraphIndex; index()
        remains unchanged for G1's deterministic, model-free use.
        """
        if not chunks:
            raise ValueError("Cannot load a graph with an empty chunk list")

        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._graph = entity_graph
        logger.info(
            "Loaded entity graph: %d entities, %d relations, over %d chunks",
            entity_graph.entity_count(),
            entity_graph.relation_count(),
            len(chunks),
        )

    def query(self, query_text: str, top_k: int) -> list[RetrievedEvidence]:
        """Resolve entry entities in query_text, expand via bounded BFS, score
        and rank chunks, and return the top_k as RetrievedEvidence(source=GRAPH).

        Zero resolved entry entities is not an error -- it returns an empty
        list, the correct input to INSUFFICIENT_EVIDENCE downstream.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self._graph is None:
            raise ValueError("Index has not been built yet; call index() first.")

        entry_entity_ids = resolve_entry_entities(self._graph, query_text, self._config)
        if not entry_entity_ids:
            return []

        chunk_scores = score_chunks(self._graph, entry_entity_ids, self._config.max_hops)
        ranked = rank_chunks(chunk_scores, top_k)

        evidence: list[RetrievedEvidence] = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            chunk = self._chunks_by_id[chunk_id]
            provenance = Provenance(chunk_id=chunk.chunk_id, char_start=chunk.char_start, char_end=chunk.char_end)
            evidence.append(
                RetrievedEvidence(
                    evidence_id=make_raw_evidence_id(query_text, chunk_id, rank),
                    chunk_id=chunk_id,
                    text=chunk.text,
                    source=RetrievalSource.GRAPH,
                    raw_score=score,
                    rank=rank,
                    provenance=provenance,
                )
            )
        return evidence
