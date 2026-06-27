"""Graph subsystem (Phase 8 / Version 2): a deterministic entity-graph index
over the existing chunk-ID space.

Opt-in only -- no module outside lncvs.graph constructs or wires
GraphIndex/GraphRetriever. Wiring one into a RetrievalOrchestrator's
retriever list is a caller decision, exactly like choosing to include
BM25Retriever alongside SemanticRetriever.

Stage G1 scope: entity-only graph, deterministic rule-based extraction
(zero NLP models), exact-match entry resolution, bounded BFS, explicit
chunk scoring. See CLAUDE.md's Version 2 Roadmap and the Phase 8
architecture review for the full G1-G5 roadmap and the V2 entry-gate
decision record.
"""

from lncvs.graph.builder import EntityGraph, build_entity_graph
from lncvs.graph.config import GraphConfig
from lncvs.graph.index import GraphIndex
from lncvs.graph.retriever import GraphRetriever
from lncvs.graph.segmentation import (
    ChapterSpan,
    ExtractionWindow,
    count_tokens,
    segment_into_chapters,
    segment_into_extraction_windows,
)

__all__ = [
    "ChapterSpan",
    "EntityGraph",
    "ExtractionWindow",
    "GraphConfig",
    "GraphIndex",
    "GraphRetriever",
    "build_entity_graph",
    "count_tokens",
    "segment_into_chapters",
    "segment_into_extraction_windows",
]
