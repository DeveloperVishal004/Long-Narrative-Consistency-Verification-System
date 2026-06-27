"""compute_citation_metrics() tests, hand-computed."""

import pytest

from lncvs.evaluation.metrics.citation import compute_citation_metrics
from lncvs.schemas import Contradiction, EvidenceLedger, SupportingEvidence


def test_returns_none_when_nothing_cited() -> None:
    ledger = EvidenceLedger(original_claim="claim")
    metrics = compute_citation_metrics(ledger, gold_chunk_ids={"chunk-1"})
    assert metrics is None


def test_hand_computed_citation_accuracy_and_hallucination_rate() -> None:
    """3 citations total: 2 grounded in gold, 1 not.
    citation_accuracy = 2/3, hallucination_rate = 1/3."""
    ledger = EvidenceLedger(original_claim="claim")
    ledger.contradictions.append(Contradiction(atomic_claim_id="claim-1", evidence_chunk_id="chunk-gold-1", nli_score=0.9))
    ledger.supporting_evidence.append(
        SupportingEvidence(atomic_claim_id="claim-2", evidence_chunk_id="chunk-gold-2", nli_score=0.8)
    )
    ledger.supporting_evidence.append(
        SupportingEvidence(atomic_claim_id="claim-3", evidence_chunk_id="chunk-not-gold", nli_score=0.7)
    )

    metrics = compute_citation_metrics(ledger, gold_chunk_ids={"chunk-gold-1", "chunk-gold-2"})

    assert metrics is not None
    assert metrics.cited_count == 3
    assert metrics.grounded_count == 2
    assert metrics.citation_accuracy == pytest.approx(2 / 3)
    assert metrics.hallucination_rate == pytest.approx(1 / 3)


def test_all_citations_grounded_yields_perfect_accuracy() -> None:
    ledger = EvidenceLedger(original_claim="claim")
    ledger.contradictions.append(Contradiction(atomic_claim_id="claim-1", evidence_chunk_id="chunk-gold", nli_score=0.9))

    metrics = compute_citation_metrics(ledger, gold_chunk_ids={"chunk-gold"})

    assert metrics is not None
    assert metrics.citation_accuracy == 1.0
    assert metrics.hallucination_rate == 0.0


def test_all_citations_ungrounded_yields_full_hallucination_rate() -> None:
    ledger = EvidenceLedger(original_claim="claim")
    ledger.contradictions.append(Contradiction(atomic_claim_id="claim-1", evidence_chunk_id="chunk-bad", nli_score=0.9))

    metrics = compute_citation_metrics(ledger, gold_chunk_ids={"chunk-gold"})

    assert metrics is not None
    assert metrics.citation_accuracy == 0.0
    assert metrics.hallucination_rate == 1.0
