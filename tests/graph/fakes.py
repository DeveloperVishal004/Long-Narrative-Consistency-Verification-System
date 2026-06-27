"""Test helper for building DocumentChunk fixtures for the graph test suite."""

from lncvs.schemas import DocumentChunk


def make_chunk(chunk_id: str, text: str, source_id: str = "doc") -> DocumentChunk:
    """Build a minimal, valid DocumentChunk with an arbitrary deterministic
    span -- char offsets are not semantically meaningful in these tests,
    only that char_end > char_start, per DocumentChunk's validator."""
    return DocumentChunk(chunk_id=chunk_id, text=text, char_start=0, char_end=len(text), source_id=source_id)
