"""round_robin_fuse() tests, mirroring tests/fusion/test_rrf.py's structure."""

import pytest

from lncvs.evaluation import round_robin_fuse
from lncvs.schemas import Provenance, RetrievalSource, RetrievedEvidence


def _evidence(
    evidence_id: str,
    chunk_id: str,
    source: RetrievalSource,
    rank: int,
    atomic_claim_id: str | None,
    query_id: str | None,
    text: str = "evidence text",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        text=text,
        source=source,
        raw_score=0.9,
        rank=rank,
        provenance=Provenance(chunk_id=chunk_id, char_start=0, char_end=10),
        atomic_claim_id=atomic_claim_id,
        query_id=query_id,
    )


def test_round_robin_fuse_rejects_unstamped_evidence() -> None:
    unstamped = _evidence("ev-1", "chunk-1", RetrievalSource.SEMANTIC, 1, atomic_claim_id=None, query_id=None)
    with pytest.raises(ValueError, match="atomic_claim_id is not set"):
        round_robin_fuse([unstamped], top_k_fused=10)


def test_round_robin_fuse_keeps_best_rank_across_sources() -> None:
    evidence = [
        _evidence("ev-1", "chunk-arm", RetrievalSource.SEMANTIC, 3, "claim-1", "query-a"),
        _evidence("ev-2", "chunk-arm", RetrievalSource.LEXICAL, 1, "claim-1", "query-b"),
    ]

    fused = round_robin_fuse(evidence, top_k_fused=10)

    assert len(fused) == 1
    assert fused[0].rrf_score == pytest.approx(1.0 / 2)  # 1 / (1 + best_rank=1)
    assert set(fused[0].contributing_sources) == {RetrievalSource.SEMANTIC, RetrievalSource.LEXICAL}


def test_round_robin_fuse_deduplicates_by_chunk_per_claim() -> None:
    evidence = [
        _evidence("ev-1", "chunk-1", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-2", RetrievalSource.SEMANTIC, 2, "claim-1", "query-1"),
    ]

    fused = round_robin_fuse(evidence, top_k_fused=10)

    assert [f.chunk_id for f in fused] == ["chunk-1", "chunk-2"]


def test_round_robin_fuse_caps_results_per_claim() -> None:
    evidence = [
        _evidence(f"ev-{i}", f"chunk-{i}", RetrievalSource.SEMANTIC, i, "claim-1", "query-1") for i in range(1, 6)
    ]

    fused = round_robin_fuse(evidence, top_k_fused=2)

    assert len(fused) == 2
    assert fused[0].chunk_id == "chunk-1"
    assert fused[1].chunk_id == "chunk-2"


def test_round_robin_fuse_is_deterministic_across_calls() -> None:
    evidence = [
        _evidence("ev-1", "chunk-z", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-a", RetrievalSource.LEXICAL, 1, "claim-1", "query-2"),
    ]

    first = round_robin_fuse(evidence, top_k_fused=10)
    second = round_robin_fuse(evidence, top_k_fused=10)

    assert [f.chunk_id for f in first] == [f.chunk_id for f in second]
    assert [f.rrf_score for f in first] == [f.rrf_score for f in second]


def test_round_robin_fuse_ties_break_by_chunk_id() -> None:
    evidence = [
        _evidence("ev-1", "chunk-z", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-a", RetrievalSource.LEXICAL, 1, "claim-1", "query-2"),
    ]

    fused = round_robin_fuse(evidence, top_k_fused=10)

    assert [f.chunk_id for f in fused] == ["chunk-a", "chunk-z"]
