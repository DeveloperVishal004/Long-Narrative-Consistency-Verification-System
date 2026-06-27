"""Provenance assignment service: ties tiered quote matching
(matching.py) to chunk-overlap resolution (lncvs.chunking) and produces
the deterministic ResolvedFact/RejectedFact partition for one extraction
window (frozen G2 spec §3).

This is the trust boundary between LLM output and the deterministic
graph: a fact reaches ResolvedFact only if at least one of its
evidence_quotes resolved (Tier 1 or Tier 2) to at least one real,
indexed chunk. Everything else -- zero quotes resolved, an ambiguous
fuzzy candidate, a quote not found at all -- is quarantined into
RejectedFact and never reaches entity resolution or the graph builder.
"""

from lncvs.chunking import chunks_overlapping_span
from lncvs.graph.llm_extraction.schema import WindowExtraction
from lncvs.graph.provenance.config import ProvenanceConfig
from lncvs.graph.provenance.matching import MatchTier, QuoteMatch, resolve_quote
from lncvs.graph.provenance.models import RawFact, RejectedFact, ResolvedFact, WindowProvenanceResult
from lncvs.schemas import DocumentChunk, Provenance


def _resolve_fact(
    raw: RawFact,
    chapter_index: int,
    window_index: int | None,
    window_text: str,
    window_char_start: int,
    chunks: list[DocumentChunk],
    chunks_by_id: dict[str, DocumentChunk],
    config: ProvenanceConfig,
) -> ResolvedFact | RejectedFact:
    quote_matches: list[QuoteMatch] = [resolve_quote(quote, window_text, config) for quote in raw.evidence_quotes]

    provenance_by_key: dict[tuple[str, int, int], Provenance] = {}
    for match in quote_matches:
        if match.tier is MatchTier.FAILED:
            continue

        global_start = window_char_start + match.char_start
        global_end = window_char_start + match.char_end

        for chunk_id in sorted(chunks_overlapping_span(global_start, global_end, chunks)):
            chunk = chunks_by_id[chunk_id]
            clipped_start = max(global_start, chunk.char_start)
            clipped_end = min(global_end, chunk.char_end)
            key = (chunk_id, clipped_start, clipped_end)
            provenance_by_key[key] = Provenance(chunk_id=chunk_id, char_start=clipped_start, char_end=clipped_end)

    if not provenance_by_key:
        return RejectedFact(
            raw=raw,
            chapter_index=chapter_index,
            window_index=window_index,
            quote_matches=tuple(quote_matches),
            reason="no evidence_quotes resolved to any indexed chunk",
        )

    provenance = tuple(sorted(provenance_by_key.values(), key=lambda p: (p.chunk_id, p.char_start)))
    return ResolvedFact(
        raw=raw,
        chapter_index=chapter_index,
        window_index=window_index,
        provenance=provenance,
        quote_matches=tuple(quote_matches),
    )


def resolve_window_provenance(
    extraction: WindowExtraction,
    window_text: str,
    chapter_index: int,
    window_index: int | None,
    window_char_start: int,
    chunks: list[DocumentChunk],
    config: ProvenanceConfig | None = None,
) -> WindowProvenanceResult:
    """Resolve every entity/relation/event in extraction against window_text,
    mapping resolved spans to real chunk_ids via chunks.

    window_char_start is the global cleaned-text offset window_text begins
    at (an ExtractionWindow's char_start); every QuoteMatch's char_start/
    char_end is in window-local coordinates and is converted to global
    coordinates here before chunk-overlap resolution. chunks must be the
    same chunk list (and chunk-ID space) the rest of the pipeline indexes.
    """
    if not chunks:
        raise ValueError("chunks must not be empty")

    config = config or ProvenanceConfig()
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    resolved_entities: list[ResolvedFact] = []
    rejected_entities: list[RejectedFact] = []
    for entity in extraction.entities:
        result = _resolve_fact(entity, chapter_index, window_index, window_text, window_char_start, chunks, chunks_by_id, config)
        (resolved_entities if isinstance(result, ResolvedFact) else rejected_entities).append(result)

    resolved_relations: list[ResolvedFact] = []
    rejected_relations: list[RejectedFact] = []
    for relation in extraction.relations:
        result = _resolve_fact(relation, chapter_index, window_index, window_text, window_char_start, chunks, chunks_by_id, config)
        (resolved_relations if isinstance(result, ResolvedFact) else rejected_relations).append(result)

    resolved_events: list[ResolvedFact] = []
    rejected_events: list[RejectedFact] = []
    for event in extraction.events:
        result = _resolve_fact(event, chapter_index, window_index, window_text, window_char_start, chunks, chunks_by_id, config)
        (resolved_events if isinstance(result, ResolvedFact) else rejected_events).append(result)

    return WindowProvenanceResult(
        chapter_index=chapter_index,
        window_index=window_index,
        resolved_entities=tuple(resolved_entities),
        resolved_relations=tuple(resolved_relations),
        resolved_events=tuple(resolved_events),
        rejected_entities=tuple(rejected_entities),
        rejected_relations=tuple(rejected_relations),
        rejected_events=tuple(rejected_events),
    )
