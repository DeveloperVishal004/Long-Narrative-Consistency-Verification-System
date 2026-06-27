"""Reciprocal Rank Fusion tests — pure, hand-computed expected scores."""

import pytest

from lncvs.fusion import FusionConfig, fuse_evidence
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


def test_fuse_evidence_rejects_unstamped_evidence() -> None:
    unstamped = _evidence("ev-1", "chunk-1", RetrievalSource.SEMANTIC, 1, atomic_claim_id=None, query_id=None)
    with pytest.raises(ValueError, match="atomic_claim_id is not set"):
        fuse_evidence([unstamped], FusionConfig())


def test_single_source_single_query_degrades_to_rank_order() -> None:
    evidence = [
        _evidence("ev-1", "chunk-1", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-2", RetrievalSource.SEMANTIC, 2, "claim-1", "query-1"),
    ]
    config = FusionConfig(rrf_k=60)

    fused = fuse_evidence(evidence, config)

    assert [f.chunk_id for f in fused] == ["chunk-1", "chunk-2"]
    assert fused[0].rrf_score == pytest.approx(1.0 / 61)
    assert fused[1].rrf_score == pytest.approx(1.0 / 62)


def test_hand_computed_rrf_score_for_chunk_in_two_lists() -> None:
    """chunk-arm is rank 1 from semantic (query-a) and rank 2 from lexical (query-b)
    for the same claim: rrf_score = 1/(60+1) + 1/(60+2)."""
    evidence = [
        _evidence("ev-1", "chunk-arm", RetrievalSource.SEMANTIC, 1, "claim-1", "query-a"),
        _evidence("ev-2", "chunk-arm", RetrievalSource.LEXICAL, 2, "claim-1", "query-b"),
    ]
    config = FusionConfig(rrf_k=60)

    fused = fuse_evidence(evidence, config)

    assert len(fused) == 1
    expected_score = 1.0 / 61 + 1.0 / 62
    assert fused[0].rrf_score == pytest.approx(expected_score)
    assert fused[0].chunk_id == "chunk-arm"


def test_contributing_sources_and_query_ids_are_deduped_and_complete() -> None:
    evidence = [
        _evidence("ev-1", "chunk-arm", RetrievalSource.SEMANTIC, 1, "claim-1", "query-a"),
        _evidence("ev-2", "chunk-arm", RetrievalSource.LEXICAL, 2, "claim-1", "query-b"),
        _evidence("ev-3", "chunk-arm", RetrievalSource.SEMANTIC, 1, "claim-1", "query-c"),
    ]
    config = FusionConfig()

    fused = fuse_evidence(evidence, config)

    assert len(fused) == 1
    assert set(fused[0].contributing_sources) == {RetrievalSource.SEMANTIC, RetrievalSource.LEXICAL}
    assert set(fused[0].contributing_query_ids) == {"query-a", "query-b", "query-c"}


def test_same_chunk_under_different_claims_produces_separate_fused_records() -> None:
    evidence = [
        _evidence("ev-1", "chunk-shared", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-shared", RetrievalSource.SEMANTIC, 1, "claim-2", "query-2"),
    ]
    config = FusionConfig()

    fused = fuse_evidence(evidence, config)

    assert len(fused) == 2
    assert {f.atomic_claim_id for f in fused} == {"claim-1", "claim-2"}


def test_tied_rrf_scores_break_ties_deterministically_by_chunk_id() -> None:
    evidence = [
        _evidence("ev-1", "chunk-z", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-a", RetrievalSource.LEXICAL, 1, "claim-1", "query-2"),
    ]
    config = FusionConfig()

    fused = fuse_evidence(evidence, config)

    # Both chunks have identical rrf_score (rank 1 from one list each);
    # tie-break must order by chunk_id ascending: "chunk-a" before "chunk-z".
    assert fused[0].rrf_score == pytest.approx(fused[1].rrf_score)
    assert [f.chunk_id for f in fused] == ["chunk-a", "chunk-z"]


def test_top_k_fused_caps_results_per_claim() -> None:
    evidence = [
        _evidence(f"ev-{i}", f"chunk-{i}", RetrievalSource.SEMANTIC, i, "claim-1", "query-1")
        for i in range(1, 6)
    ]
    config = FusionConfig(top_k_fused=2)

    fused = fuse_evidence(evidence, config)

    assert len(fused) == 2
    assert fused[0].chunk_id == "chunk-1"
    assert fused[1].chunk_id == "chunk-2"


def test_top_k_fused_caps_independently_per_claim() -> None:
    evidence = [
        _evidence("ev-1", "chunk-1a", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-2a", RetrievalSource.SEMANTIC, 2, "claim-1", "query-1"),
        _evidence("ev-3", "chunk-1b", RetrievalSource.SEMANTIC, 1, "claim-2", "query-2"),
    ]
    config = FusionConfig(top_k_fused=1)

    fused = fuse_evidence(evidence, config)

    claim_1_results = [f for f in fused if f.atomic_claim_id == "claim-1"]
    claim_2_results = [f for f in fused if f.atomic_claim_id == "claim-2"]
    assert len(claim_1_results) == 1
    assert len(claim_2_results) == 1


def test_fused_text_is_preserved_from_the_underlying_evidence() -> None:
    evidence = [
        _evidence(
            "ev-1", "chunk-arm", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1",
            text="John lost his left arm in 2010.",
        )
    ]
    fused = fuse_evidence(evidence, FusionConfig())
    assert fused[0].text == "John lost his left arm in 2010."


def test_fuse_evidence_is_deterministic_across_calls() -> None:
    evidence = [
        _evidence("ev-1", "chunk-z", RetrievalSource.SEMANTIC, 1, "claim-1", "query-1"),
        _evidence("ev-2", "chunk-a", RetrievalSource.LEXICAL, 1, "claim-1", "query-2"),
    ]
    config = FusionConfig()

    first_run = fuse_evidence(evidence, config)
    second_run = fuse_evidence(evidence, config)

    assert [f.chunk_id for f in first_run] == [f.chunk_id for f in second_run]
    assert [f.rrf_score for f in first_run] == [f.rrf_score for f in second_run]
