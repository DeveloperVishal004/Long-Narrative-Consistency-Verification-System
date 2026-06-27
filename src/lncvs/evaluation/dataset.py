"""Gold evaluation dataset loading and span-to-chunk mapping."""

from pathlib import Path

from lncvs.chunking import chunks_overlapping_any_span
from lncvs.schemas import DocumentChunk, EvaluationDataset, GoldExample, GoldSpan


def load_dataset(path: Path, dataset_id: str) -> EvaluationDataset:
    """Load a gold evaluation dataset from a JSONL file, one GoldExample per line."""
    examples: list[GoldExample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldExample.model_validate_json(line))

    if not examples:
        raise ValueError(f"Gold dataset {path} contains no examples")

    return EvaluationDataset(dataset_id=dataset_id, examples=examples)


def map_spans_to_chunks(spans: list[GoldSpan], chunks: list[DocumentChunk]) -> set[str]:
    """Return the chunk_ids of every chunk whose span overlaps any gold span.

    Span-based gold labels are deliberately decoupled from chunk_id, which
    is a content hash that changes whenever chunking config changes --
    mapping at evaluation time means the same gold dataset stays valid
    across any chunking configuration.

    Thin GoldSpan adapter over lncvs.chunking.chunks_overlapping_any_span,
    the relocated generic primitive (Phase 8 / G2 Slice 4) -- relocated so
    a module upstream of evaluation/ (e.g. lncvs.graph's provenance
    assignment) can share the identical overlap logic without inverting
    CLAUDE.md's dependency direction. Kept here, with this exact
    signature, so no existing import site needs to change.
    """
    return chunks_overlapping_any_span([(span.char_start, span.char_end) for span in spans], chunks)
