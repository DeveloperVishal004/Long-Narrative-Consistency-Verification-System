"""ChromaIndex tests using the deterministic FakeEmbedder (no model download)."""

import pytest

from lncvs.indexing import ChromaIndex
from lncvs.schemas import RetrievalSource
from tests.indexing.fakes import FakeEmbedder

ARM_CHUNK_ID = "chunk-arm"
LONDON_CHUNK_ID = "chunk-london"


def _build_index() -> ChromaIndex:
    from lncvs.schemas import DocumentChunk

    index = ChromaIndex(embedder=FakeEmbedder(), collection_name="test-chroma-index")
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
    index = ChromaIndex(embedder=FakeEmbedder(), collection_name="empty-test")
    with pytest.raises(ValueError):
        index.index([])


def test_query_rejects_non_positive_top_k() -> None:
    index = _build_index()
    with pytest.raises(ValueError):
        index.query("What happened to John's arm?", top_k=0)


def test_query_retrieves_the_arm_chunk_for_a_related_query() -> None:
    """FakeEmbedder is bag-of-words, not truly semantic — use a query with clear word overlap
    against the arm chunk and minimal overlap against the london chunk to exercise the
    indexing/retrieval plumbing deterministically. Genuine semantic generalization is proven
    separately by the real-model acceptance test."""
    index = _build_index()

    results = index.query("Did John lose his arm in an accident?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == ARM_CHUNK_ID
    assert "lost his left arm" in results[0].text
    assert results[0].source is RetrievalSource.SEMANTIC
    assert results[0].rank == 1
    assert results[0].provenance.chunk_id == ARM_CHUNK_ID


def test_query_returns_ranked_results_for_top_k_greater_than_one() -> None:
    index = _build_index()

    results = index.query("John", top_k=2)

    assert len(results) == 2
    assert [evidence.rank for evidence in results] == [1, 2]
