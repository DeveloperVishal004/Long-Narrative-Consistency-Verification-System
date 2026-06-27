"""group_evidence_by_claim tests."""

import pytest

from lncvs.retrieval import group_evidence_by_claim
from lncvs.schemas import Provenance, RetrievalSource, RetrievedEvidence


def _stamped_evidence(evidence_id: str, chunk_id: str, atomic_claim_id: str, query_id: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        text="some evidence text",
        source=RetrievalSource.SEMANTIC,
        raw_score=0.9,
        rank=1,
        provenance=Provenance(chunk_id=chunk_id, char_start=0, char_end=10),
        atomic_claim_id=atomic_claim_id,
        query_id=query_id,
    )


def test_groups_evidence_by_atomic_claim_id() -> None:
    evidence = [
        _stamped_evidence("ev-1", "chunk-1", "claim-a", "query-1"),
        _stamped_evidence("ev-2", "chunk-2", "claim-a", "query-2"),
        _stamped_evidence("ev-3", "chunk-3", "claim-b", "query-3"),
    ]

    grouped = group_evidence_by_claim(evidence)

    assert set(grouped.keys()) == {"claim-a", "claim-b"}
    assert len(grouped["claim-a"]) == 2
    assert len(grouped["claim-b"]) == 1


def test_empty_evidence_list_returns_empty_dict() -> None:
    assert group_evidence_by_claim([]) == {}


def test_unstamped_evidence_raises() -> None:
    unstamped = RetrievedEvidence(
        evidence_id="ev-1",
        chunk_id="chunk-1",
        text="some evidence text",
        source=RetrievalSource.SEMANTIC,
        raw_score=0.9,
        rank=1,
        provenance=Provenance(chunk_id="chunk-1", char_start=0, char_end=10),
    )

    with pytest.raises(ValueError, match="atomic_claim_id is not set"):
        group_evidence_by_claim([unstamped])
