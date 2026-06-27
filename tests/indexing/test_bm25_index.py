"""BM25Index tests."""

import pytest

from lncvs.indexing import BM25Index
from lncvs.schemas import DocumentChunk, RetrievalSource

ARM_CHUNK_ID = "chunk-arm"
LONDON_CHUNK_ID = "chunk-london"


def _build_index() -> BM25Index:
    index = BM25Index(collection_name="test-bm25-index")
    index.index(
        [
            DocumentChunk(
                chunk_id=ARM_CHUNK_ID,
                text="John lost his left arm in an accident in 2010.",
                char_start=0,
                char_end=48,
                chapter=None,
                source_id="john_test",
            ),
            DocumentChunk(
                chunk_id=LONDON_CHUNK_ID,
                text="John moved to London in 2012.",
                char_start=49,
                char_end=79,
                chapter=None,
                source_id="john_test",
            ),
        ]
    )
    return index


def test_index_rejects_empty_chunk_list() -> None:
    index = BM25Index(collection_name="empty-test")
    with pytest.raises(ValueError):
        index.index([])


def test_query_before_index_raises() -> None:
    index = BM25Index(collection_name="unbuilt-test")
    with pytest.raises(ValueError, match="has not been built yet"):
        index.query("John", top_k=1)


def test_query_rejects_non_positive_top_k() -> None:
    index = _build_index()
    with pytest.raises(ValueError):
        index.query("John lost his arm", top_k=0)


def test_query_retrieves_the_arm_chunk_for_a_lexically_overlapping_query() -> None:
    index = _build_index()

    results = index.query("Did John lose his arm in an accident?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == ARM_CHUNK_ID
    assert results[0].source is RetrievalSource.LEXICAL
    assert results[0].rank == 1


def test_query_preserves_input_chunk_ids_unmodified() -> None:
    """BM25Index must never generate its own chunk IDs — this is what lets
    fusion dedup across BM25 and Chroma by chunk_id with no reconciliation."""
    index = _build_index()

    results = index.query("John", top_k=2)

    result_chunk_ids = {r.chunk_id for r in results}
    assert result_chunk_ids == {ARM_CHUNK_ID, LONDON_CHUNK_ID}


def test_query_with_no_token_overlap_still_returns_top_k_deterministically() -> None:
    """A query with zero vocabulary overlap returns all-zero BM25 scores;
    ordering must still be deterministic (tie-broken by chunk_id), not
    dependent on internal iteration order."""
    index = _build_index()

    first_run = index.query("xyzzyplugh", top_k=2)
    second_run = index.query("xyzzyplugh", top_k=2)

    assert [r.chunk_id for r in first_run] == [r.chunk_id for r in second_run]
    assert [r.chunk_id for r in first_run] == sorted([ARM_CHUNK_ID, LONDON_CHUNK_ID])


def test_results_are_deterministic_across_repeated_queries() -> None:
    index = _build_index()

    first_run = index.query("Did John lose his arm in an accident?", top_k=2)
    second_run = index.query("Did John lose his arm in an accident?", top_k=2)

    assert [r.evidence_id for r in first_run] == [r.evidence_id for r in second_run]
