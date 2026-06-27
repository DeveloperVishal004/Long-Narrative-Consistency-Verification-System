"""Generic character-span-to-chunk overlap mapping.

Relocated here from lncvs.evaluation.dataset (Phase 8 / G2 Slice 4) so it
can be shared by any module upstream of evaluation/ -- e.g. lncvs.graph's
provenance assignment -- without inverting CLAUDE.md's dependency
direction (... -> rules -> orchestration -> evaluation; nothing upstream
may import something downstream). chunking/ is the most upstream module
for which this is still domain-correct: "given chunks and a character
span, which chunks does it cover" is chunk-domain logic, not
evaluation-domain logic. This mirrors the identical relocation pattern
already used in Phase 7 for AblationVariant/FusionStrategy/round_robin_fuse.

Operates on plain (char_start, char_end) ints, never the evaluation-specific
GoldSpan type -- lncvs.evaluation.dataset.map_spans_to_chunks is kept as a
thin adapter over this primitive so no existing import site changes.
"""

from lncvs.schemas import DocumentChunk


def chunks_overlapping_span(char_start: int, char_end: int, chunks: list[DocumentChunk]) -> set[str]:
    """Return the chunk_ids of every chunk in chunks whose span overlaps
    [char_start, char_end)."""
    return {chunk.chunk_id for chunk in chunks if chunk.char_start < char_end and char_start < chunk.char_end}


def chunks_overlapping_any_span(spans: list[tuple[int, int]], chunks: list[DocumentChunk]) -> set[str]:
    """Return the chunk_ids of every chunk whose span overlaps ANY span in spans."""
    relevant: set[str] = set()
    for char_start, char_end in spans:
        relevant |= chunks_overlapping_span(char_start, char_end, chunks)
    return relevant
