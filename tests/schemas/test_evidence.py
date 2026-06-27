"""RetrievedEvidence, FusedEvidence, SupportingEvidence, and Contradiction validation tests."""

import pytest
from pydantic import ValidationError

from lncvs.schemas import (
    Contradiction,
    FusedEvidence,
    Provenance,
    RetrievalSource,
    RetrievedEvidence,
    SupportingEvidence,
)


def _provenance() -> Provenance:
    return Provenance(chunk_id="chunk-0001", char_start=0, char_end=10)


def test_retrieved_evidence_valid_construction() -> None:
    evidence = RetrievedEvidence(
        evidence_id="ev-1",
        chunk_id="chunk-0001",
        text="John lost his left arm.",
        source=RetrievalSource.SEMANTIC,
        raw_score=0.87,
        rank=1,
        provenance=_provenance(),
    )
    assert evidence.rank == 1


def test_retrieved_evidence_atomic_claim_id_and_query_id_default_to_none() -> None:
    """A claim-agnostic Retriever can construct RetrievedEvidence without these fields."""
    evidence = RetrievedEvidence(
        evidence_id="ev-1",
        chunk_id="chunk-0001",
        text="John lost his left arm.",
        source=RetrievalSource.SEMANTIC,
        raw_score=0.87,
        rank=1,
        provenance=_provenance(),
    )
    assert evidence.atomic_claim_id is None
    assert evidence.query_id is None


def test_retrieved_evidence_accepts_explicit_atomic_claim_id_and_query_id() -> None:
    evidence = RetrievedEvidence(
        evidence_id="ev-1",
        chunk_id="chunk-0001",
        text="John lost his left arm.",
        source=RetrievalSource.SEMANTIC,
        raw_score=0.87,
        rank=1,
        provenance=_provenance(),
        atomic_claim_id="claim-1",
        query_id="query-1",
    )
    assert evidence.atomic_claim_id == "claim-1"
    assert evidence.query_id == "query-1"


def test_retrieved_evidence_rejects_rank_below_one() -> None:
    with pytest.raises(ValidationError):
        RetrievedEvidence(
            evidence_id="ev-1",
            chunk_id="chunk-0001",
            text="John lost his left arm.",
            source=RetrievalSource.SEMANTIC,
            raw_score=0.87,
            rank=0,
            provenance=_provenance(),
        )


def test_fused_evidence_requires_at_least_one_contributing_source() -> None:
    with pytest.raises(ValidationError):
        FusedEvidence(
            atomic_claim_id="claim-1",
            chunk_id="chunk-0001",
            text="John lost his left arm.",
            rrf_score=0.5,
            contributing_sources=[],
            contributing_query_ids=["query-1"],
        )


def test_fused_evidence_requires_at_least_one_contributing_query_id() -> None:
    with pytest.raises(ValidationError):
        FusedEvidence(
            atomic_claim_id="claim-1",
            chunk_id="chunk-0001",
            text="John lost his left arm.",
            rrf_score=0.5,
            contributing_sources=[RetrievalSource.SEMANTIC],
            contributing_query_ids=[],
        )


def test_fused_evidence_valid_construction() -> None:
    fused = FusedEvidence(
        atomic_claim_id="claim-1",
        chunk_id="chunk-0001",
        text="John lost his left arm.",
        rrf_score=0.5,
        contributing_sources=[RetrievalSource.SEMANTIC, RetrievalSource.LEXICAL],
        contributing_query_ids=["query-1", "query-2"],
    )
    assert RetrievalSource.SEMANTIC in fused.contributing_sources
    assert fused.atomic_claim_id == "claim-1"
    assert not hasattr(fused, "source_ranks")


def test_supporting_evidence_nli_score_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        SupportingEvidence(atomic_claim_id="claim-1", evidence_chunk_id="chunk-0001", nli_score=1.5)


def test_contradiction_nli_score_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Contradiction(atomic_claim_id="claim-1", evidence_chunk_id="chunk-0001", nli_score=-0.1)


def test_contradiction_valid_construction() -> None:
    contradiction = Contradiction(
        atomic_claim_id="claim-1",
        evidence_chunk_id="chunk-0001",
        nli_score=0.93,
        explanation="John lost his left arm, contradicting two-handed piano playing.",
    )
    assert contradiction.nli_score == 0.93
