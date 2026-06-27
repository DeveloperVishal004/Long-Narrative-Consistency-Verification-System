"""evidence_id determinism tests.

Phase 1 used uuid4(), producing a different evidence_id on every call even
for the identical query against the identical index. This violates the
project's reproducibility mandate. These tests pin the deterministic
replacement and guard that ranking/ordering is unaffected by the change.
"""

from lncvs.indexing import ChromaIndex
from lncvs.schemas import DocumentChunk
from tests.indexing.fakes import FakeEmbedder

ARM_CHUNK = DocumentChunk(
    chunk_id="chunk-arm",
    text="John lost his left arm in an accident in 2010.",
    char_start=0,
    char_end=48,
    chapter=None,
    source_id="john_test",
)
LONDON_CHUNK = DocumentChunk(
    chunk_id="chunk-london",
    text="John moved to London in 2012.",
    char_start=49,
    char_end=79,
    chapter=None,
    source_id="john_test",
)


def _build_index(collection_name: str) -> ChromaIndex:
    index = ChromaIndex(embedder=FakeEmbedder(), collection_name=collection_name)
    index.index([ARM_CHUNK, LONDON_CHUNK])
    return index


def test_same_query_against_same_index_produces_identical_evidence_ids() -> None:
    index = _build_index("determinism-test-1")

    first_run = index.query("Did John lose his arm in an accident?", top_k=2)
    second_run = index.query("Did John lose his arm in an accident?", top_k=2)

    assert [e.evidence_id for e in first_run] == [e.evidence_id for e in second_run]


def test_same_query_against_a_fresh_equivalent_index_produces_identical_evidence_ids() -> None:
    """Determinism must hold across process/index instances, not just within one."""
    index_a = _build_index("determinism-test-2a")
    index_b = _build_index("determinism-test-2b")

    results_a = index_a.query("Did John lose his arm in an accident?", top_k=2)
    results_b = index_b.query("Did John lose his arm in an accident?", top_k=2)

    assert [e.evidence_id for e in results_a] == [e.evidence_id for e in results_b]


def test_different_chunk_ids_or_ranks_produce_different_evidence_ids() -> None:
    index = _build_index("determinism-test-3")

    results = index.query("Did John lose his arm in an accident?", top_k=2)

    assert results[0].evidence_id != results[1].evidence_id


def test_different_queries_produce_different_evidence_ids_for_the_same_chunk() -> None:
    index = _build_index("determinism-test-4")

    results_query_a = index.query("Did John lose his arm in an accident?", top_k=1)
    results_query_b = index.query("Where did John move?", top_k=1)

    assert results_query_a[0].evidence_id != results_query_b[0].evidence_id


def test_evidence_id_no_longer_looks_like_a_uuid() -> None:
    """uuid4() produces 36-character hyphenated strings; the deterministic ID must not."""
    index = _build_index("determinism-test-5")
    results = index.query("Did John lose his arm in an accident?", top_k=1)

    assert "-" not in results[0].evidence_id
    assert len(results[0].evidence_id) != 36


def test_ranking_is_unchanged_by_the_evidence_id_fix() -> None:
    """The deterministic evidence_id change must not alter which chunk ranks where."""
    index = _build_index("determinism-test-6")

    results = index.query("Did John lose his arm in an accident?", top_k=2)

    assert results[0].chunk_id == "chunk-arm"
    assert results[0].rank == 1
    assert results[1].rank == 2
