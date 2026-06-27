"""chunks_overlapping_span / chunks_overlapping_any_span: the generic
span-overlap primitive relocated from evaluation/dataset.py (Phase 8 / G2
Slice 4) so it can be shared upstream without inverting the dependency
direction. See lncvs.evaluation.dataset.map_spans_to_chunks for the
GoldSpan-specific adapter that still wraps this exact function."""

from lncvs.chunking import chunks_overlapping_any_span, chunks_overlapping_span
from lncvs.schemas import DocumentChunk


def _chunk(chunk_id: str, start: int, end: int) -> DocumentChunk:
    return DocumentChunk(chunk_id=chunk_id, text="x" * (end - start), char_start=start, char_end=end, source_id="doc")


def test_no_overlap_returns_empty_set() -> None:
    chunks = [_chunk("a", 0, 10), _chunk("b", 10, 20)]
    assert chunks_overlapping_span(20, 30, chunks) == set()


def test_span_fully_inside_one_chunk() -> None:
    chunks = [_chunk("a", 0, 100)]
    assert chunks_overlapping_span(10, 20, chunks) == {"a"}


def test_span_overlapping_chunk_boundary_matches_both_chunks() -> None:
    chunks = [_chunk("a", 0, 100), _chunk("b", 80, 180)]  # overlap region [80,100)
    assert chunks_overlapping_span(90, 110, chunks) == {"a", "b"}


def test_span_spanning_three_chunks() -> None:
    chunks = [_chunk("a", 0, 50), _chunk("b", 40, 90), _chunk("c", 80, 130)]
    assert chunks_overlapping_span(45, 85, chunks) == {"a", "b", "c"}


def test_adjacent_non_overlapping_spans_do_not_match() -> None:
    chunks = [_chunk("a", 0, 10)]
    assert chunks_overlapping_span(10, 20, chunks) == set()


def test_overlapping_any_span_unions_across_multiple_spans() -> None:
    chunks = [_chunk("a", 0, 10), _chunk("b", 10, 20), _chunk("c", 20, 30)]
    result = chunks_overlapping_any_span([(0, 5), (25, 30)], chunks)
    assert result == {"a", "c"}
