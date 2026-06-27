"""End-to-end graph-construction pipeline (Phase 8 / G2): cleaned
narrative text + chunks + an injected WindowExtractor -> ConstructedGraph
+ a ready-to-query EntityGraph.

Ties together every G2 stage -- segmentation (graph.segmentation) ->
extraction (graph.llm_extraction) -> provenance assignment
(graph.provenance) -> entity resolution (graph.entity_resolution) ->
construction (this package's service.py) -- into one reusable function.
Run once per novel; the returned EntityGraph is then loaded into a
GraphIndex (via EntityGraph -> GraphIndex.load_graph) and reused across
every claim that references that novel, never rebuilt per claim.
"""

import logging

from lncvs.graph.builder import EntityGraph
from lncvs.graph.construction.models import ConstructedGraph
from lncvs.graph.construction.service import build_graph
from lncvs.graph.entity_resolution.service import resolve_entities
from lncvs.graph.llm_extraction.service import WindowExtractor
from lncvs.graph.provenance.models import ResolvedFact
from lncvs.graph.provenance.service import resolve_window_provenance
from lncvs.graph.segmentation import segment_into_extraction_windows
from lncvs.schemas import DocumentChunk

logger = logging.getLogger(__name__)


def build_graph_for_novel(
    cleaned_text: str, chunks: list[DocumentChunk], extractor: WindowExtractor
) -> tuple[ConstructedGraph, EntityGraph]:
    """Run the complete G2 construction pipeline over cleaned_text.

    extractor is injected (an LLMWindowExtractor wrapping a real or fake
    StructuredLLMClient) -- this function itself makes no model calls and
    is otherwise pure deterministic orchestration.
    """
    windows = segment_into_extraction_windows(cleaned_text)

    entity_facts: list[ResolvedFact] = []
    relation_facts: list[ResolvedFact] = []
    event_facts: list[ResolvedFact] = []

    for window in windows:
        window_text = cleaned_text[window.char_start : window.char_end]
        try:
            extraction = extractor.extract(window_text, window.chapter_index, window.window_index)
        except ValueError as exc:
            # Hackathon-mode minimal fix: an extraction call that fails after
            # exhausting all provider-level retries (e.g. a deterministic
            # MAX_TOKENS truncation on one unusually dense window) must not
            # abort graph construction for the entire novel -- it is
            # isolated and skipped, exactly as a retrieval backend failure
            # is isolated elsewhere in this codebase (never crash the whole
            # run over one failed unit). Logged loudly, not swallowed: this
            # window simply contributes zero entities/relations/events.
            logger.warning(
                "Window (chapter=%s, window=%s) extraction failed and is skipped "
                "(contributes zero entities/relations/events to the graph): %s",
                window.chapter_index,
                window.window_index,
                exc,
            )
            continue
        result = resolve_window_provenance(
            extraction, window_text, window.chapter_index, window.window_index, window.char_start, chunks
        )
        entity_facts.extend(result.resolved_entities)
        relation_facts.extend(result.resolved_relations)
        event_facts.extend(result.resolved_events)

        logger.info(
            "Window (chapter=%s, window=%s): %d/%d entities, %d/%d relations, %d/%d events resolved",
            window.chapter_index,
            window.window_index,
            len(result.resolved_entities),
            len(result.resolved_entities) + len(result.rejected_entities),
            len(result.resolved_relations),
            len(result.resolved_relations) + len(result.rejected_relations),
            len(result.resolved_events),
            len(result.resolved_events) + len(result.rejected_events),
        )

    resolution = resolve_entities(entity_facts)
    constructed = build_graph(resolution, relation_facts, event_facts)
    entity_graph = EntityGraph.from_records(constructed.entities, constructed.relations)

    logger.info(
        "Built graph for novel: %d entities, %d relations, %d events, %d participations (fingerprint=%s)",
        len(constructed.entities),
        len(constructed.relations),
        len(constructed.events),
        len(constructed.participations),
        constructed.fingerprint,
    )

    return constructed, entity_graph
