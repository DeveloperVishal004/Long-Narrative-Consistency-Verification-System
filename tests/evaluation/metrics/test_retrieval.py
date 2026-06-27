"""compute_retrieval_metrics() tests, hand-computed."""

import pytest

from lncvs.evaluation.metrics.retrieval import compute_retrieval_metrics
from lncvs.schemas import EvidenceLedger, FusedEvidence, RetrievalSource


def _fused(claim_id: str, chunk_id: str, rrf_score: float) -> FusedEvidence:
    return FusedEvidence(
        atomic_claim_id=claim_id,
        chunk_id=chunk_id,
        text="evidence text",
        rrf_score=rrf_score,
        contributing_sources=[RetrievalSource.SEMANTIC],
        contributing_query_ids=["query-1"],
    )


def test_returns_none_when_gold_chunk_ids_is_empty() -> None:
    ledger = EvidenceLedger(original_claim="claim")
    ledger.fused_evidence.append(_fused("claim-1", "chunk-1", 0.5))

    metrics = compute_retrieval_metrics(ledger, gold_chunk_ids=set(), k_cutoffs=[5])

    assert metrics is None


def test_hand_computed_mrr_and_recall_precision_at_k() -> None:
    """Ranked order (by rrf_score desc): chunk-3 (0.9), chunk-1 (0.5), chunk-2 (0.1).
    gold = {chunk-1}. chunk-1 is at rank 2 -> MRR = 1/2.
    Recall@1 = 0/1 = 0.0, Precision@1 = 0/1 = 0.0.
    Recall@2 = 1/1 = 1.0, Precision@2 = 1/2 = 0.5.
    """
    ledger = EvidenceLedger(original_claim="claim")
    ledger.fused_evidence.extend(
        [
            _fused("claim-1", "chunk-1", 0.5),
            _fused("claim-1", "chunk-2", 0.1),
            _fused("claim-1", "chunk-3", 0.9),
        ]
    )

    metrics = compute_retrieval_metrics(ledger, gold_chunk_ids={"chunk-1"}, k_cutoffs=[1, 2])

    assert metrics is not None
    assert metrics.mrr == pytest.approx(0.5)

    by_k = {cutoff.k: cutoff for cutoff in metrics.cutoffs}
    assert by_k[1].recall == pytest.approx(0.0)
    assert by_k[1].precision == pytest.approx(0.0)
    assert by_k[2].recall == pytest.approx(1.0)
    assert by_k[2].precision == pytest.approx(0.5)


def test_mrr_is_zero_when_gold_chunk_never_retrieved() -> None:
    ledger = EvidenceLedger(original_claim="claim")
    ledger.fused_evidence.append(_fused("claim-1", "chunk-1", 0.9))

    metrics = compute_retrieval_metrics(ledger, gold_chunk_ids={"chunk-never-retrieved"}, k_cutoffs=[5])

    assert metrics is not None
    assert metrics.mrr == 0.0
    assert metrics.cutoffs[0].recall == 0.0


def test_retrieval_pools_across_atomic_claims_for_one_example() -> None:
    """Evidence for two different claims within the same example, both ranked together."""
    ledger = EvidenceLedger(original_claim="claim")
    ledger.fused_evidence.extend(
        [
            _fused("claim-1", "chunk-arm", 0.9),
            _fused("claim-2", "chunk-london", 0.5),
        ]
    )

    metrics = compute_retrieval_metrics(ledger, gold_chunk_ids={"chunk-arm", "chunk-london"}, k_cutoffs=[2])

    assert metrics is not None
    assert metrics.cutoffs[0].recall == pytest.approx(1.0)


def test_retrieval_deduplicates_chunk_appearing_for_multiple_claims() -> None:
    ledger = EvidenceLedger(original_claim="claim")
    ledger.fused_evidence.extend(
        [
            _fused("claim-1", "chunk-shared", 0.9),
            _fused("claim-2", "chunk-shared", 0.9),
        ]
    )

    metrics = compute_retrieval_metrics(ledger, gold_chunk_ids={"chunk-shared"}, k_cutoffs=[1])

    assert metrics is not None
    assert metrics.cutoffs[0].precision == pytest.approx(1.0)
