"""LedgerService mutation API tests."""

import pytest

from lncvs.ledger import LedgerService
from lncvs.schemas import (
    AtomicClaim,
    Contradiction,
    EvidenceLedger,
    FinalVerdict,
    FusedEvidence,
    NLILabel,
    NLIResult,
    Provenance,
    QueryOrigin,
    RetrievalQuery,
    RetrievalSource,
    RetrievedEvidence,
    VerdictEnum,
)


def _service() -> LedgerService:
    ledger = EvidenceLedger(original_claim="John played a two-handed piano piece in London.")
    return LedgerService(ledger)


def test_add_atomic_claim_appends_and_logs() -> None:
    service = _service()
    service.add_atomic_claim(AtomicClaim(claim_id="claim-1", text="John used both hands"))

    assert len(service.ledger.atomic_claims) == 1
    assert service.ledger.atomic_claims[0].claim_id == "claim-1"
    assert len(service.ledger.ledger_log) == 1
    assert "claim-1" in service.ledger.ledger_log[0].message


def test_mark_unsupported_is_idempotent() -> None:
    service = _service()
    service.mark_unsupported("claim-2")
    service.mark_unsupported("claim-2")

    assert service.ledger.unsupported_claims == ["claim-2"]


def test_add_nli_result_appends_and_logs() -> None:
    service = _service()
    service.add_nli_result(
        NLIResult(
            atomic_claim_id="claim-1",
            evidence_chunk_id="chunk-0001",
            label=NLILabel.CONTRADICTION,
            score=0.93,
            premise="John lost his left arm in an accident in 2010.",
            hypothesis="John used both hands.",
        )
    )

    assert len(service.ledger.nli_results) == 1
    assert service.ledger.nli_results[0].label is NLILabel.CONTRADICTION


def test_set_final_verdict_succeeds_once() -> None:
    service = _service()
    verdict = FinalVerdict(
        verdict=VerdictEnum.CONTRADICTORY,
        fired_rule="rule_1_contradiction",
        rationale="Claim contradicted by lost-arm evidence.",
    )

    service.set_final_verdict(verdict)

    assert service.ledger.final_verdict is verdict


def test_set_final_verdict_raises_on_second_call() -> None:
    service = _service()
    first = FinalVerdict(
        verdict=VerdictEnum.CONTRADICTORY,
        fired_rule="rule_1_contradiction",
        rationale="Claim contradicted by lost-arm evidence.",
    )
    second = FinalVerdict(
        verdict=VerdictEnum.CONSISTENT,
        fired_rule="rule_3_all_supported",
        rationale="Should never be reached.",
    )

    service.set_final_verdict(first)

    with pytest.raises(ValueError, match="write-once"):
        service.set_final_verdict(second)


def _service_with_one_claim() -> LedgerService:
    service = _service()
    service.add_atomic_claim(AtomicClaim(claim_id="claim-1", text="John used both hands"))
    return service


def test_record_retrieval_queries_populates_ledger() -> None:
    service = _service_with_one_claim()
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )

    service.record_retrieval_queries([query])

    assert service.ledger.retrieval_queries == [query]


def test_record_retrieval_queries_rejects_unknown_atomic_claim_id() -> None:
    service = _service_with_one_claim()
    query = RetrievalQuery(
        query_id="query-1", text="some text", atomic_claim_id="claim-not-in-ledger", origin=QueryOrigin.CLAIM
    )

    with pytest.raises(ValueError, match="unknown atomic_claim_id"):
        service.record_retrieval_queries([query])


def test_record_retrieval_queries_is_write_once() -> None:
    service = _service_with_one_claim()
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    service.record_retrieval_queries([query])

    with pytest.raises(ValueError, match="write-once"):
        service.record_retrieval_queries([query])


def _stamped_evidence(evidence_id: str, atomic_claim_id: str, query_id: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk_id="chunk-0001",
        text="John lost his left arm in an accident in 2010.",
        source=RetrievalSource.SEMANTIC,
        raw_score=0.9,
        rank=1,
        provenance=Provenance(chunk_id="chunk-0001", char_start=0, char_end=10),
        atomic_claim_id=atomic_claim_id,
        query_id=query_id,
    )


def test_record_retrieved_evidence_populates_ledger() -> None:
    service = _service_with_one_claim()
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    service.record_retrieval_queries([query])
    evidence = _stamped_evidence("ev-1", "claim-1", "query-1")

    service.record_retrieved_evidence([evidence])

    assert service.ledger.retrieved_evidence == [evidence]


def test_record_retrieved_evidence_rejects_missing_linkage() -> None:
    service = _service_with_one_claim()
    unstamped = RetrievedEvidence(
        evidence_id="ev-1",
        chunk_id="chunk-0001",
        text="John lost his left arm in an accident in 2010.",
        source=RetrievalSource.SEMANTIC,
        raw_score=0.9,
        rank=1,
        provenance=Provenance(chunk_id="chunk-0001", char_start=0, char_end=10),
    )

    with pytest.raises(ValueError, match="must be stamped by RetrievalOrchestrator"):
        service.record_retrieved_evidence([unstamped])


def test_record_retrieved_evidence_rejects_unknown_query_id() -> None:
    service = _service_with_one_claim()
    evidence = _stamped_evidence("ev-1", "claim-1", "query-not-recorded")

    with pytest.raises(ValueError, match="unknown query_id"):
        service.record_retrieved_evidence([evidence])


def test_record_retrieved_evidence_is_write_once() -> None:
    service = _service_with_one_claim()
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    service.record_retrieval_queries([query])
    evidence = _stamped_evidence("ev-1", "claim-1", "query-1")
    service.record_retrieved_evidence([evidence])

    with pytest.raises(ValueError, match="write-once"):
        service.record_retrieved_evidence([evidence])


def _service_with_one_claim_and_evidence() -> tuple[LedgerService, RetrievedEvidence]:
    service = _service_with_one_claim()
    query = RetrievalQuery(
        query_id="query-1", text="John used both hands", atomic_claim_id="claim-1", origin=QueryOrigin.CLAIM
    )
    service.record_retrieval_queries([query])
    evidence = _stamped_evidence("ev-1", "claim-1", "query-1")
    service.record_retrieved_evidence([evidence])
    return service, evidence


def _fused_evidence(chunk_id: str, atomic_claim_id: str, query_id: str) -> FusedEvidence:
    return FusedEvidence(
        atomic_claim_id=atomic_claim_id,
        chunk_id=chunk_id,
        text="John lost his left arm in an accident in 2010.",
        rrf_score=0.5,
        contributing_sources=[RetrievalSource.SEMANTIC],
        contributing_query_ids=[query_id],
    )


def test_record_fused_evidence_populates_ledger() -> None:
    service, evidence = _service_with_one_claim_and_evidence()
    fused = _fused_evidence(evidence.chunk_id, "claim-1", "query-1")

    service.record_fused_evidence([fused])

    assert service.ledger.fused_evidence == [fused]


def test_record_fused_evidence_rejects_unknown_atomic_claim_id() -> None:
    service, evidence = _service_with_one_claim_and_evidence()
    fused = _fused_evidence(evidence.chunk_id, "claim-not-in-ledger", "query-1")

    with pytest.raises(ValueError, match="unknown atomic_claim_id"):
        service.record_fused_evidence([fused])


def test_record_fused_evidence_rejects_chunk_with_no_retrieved_evidence() -> None:
    service, _ = _service_with_one_claim_and_evidence()
    fused = _fused_evidence("chunk-never-retrieved", "claim-1", "query-1")

    with pytest.raises(ValueError, match="no corresponding entry in retrieved_evidence"):
        service.record_fused_evidence([fused])


def test_record_fused_evidence_is_write_once() -> None:
    service, evidence = _service_with_one_claim_and_evidence()
    fused = _fused_evidence(evidence.chunk_id, "claim-1", "query-1")
    service.record_fused_evidence([fused])

    with pytest.raises(ValueError, match="write-once"):
        service.record_fused_evidence([fused])


def _service_with_one_claim_and_fused_evidence() -> tuple[LedgerService, FusedEvidence]:
    service, evidence = _service_with_one_claim_and_evidence()
    fused = _fused_evidence(evidence.chunk_id, "claim-1", "query-1")
    service.record_fused_evidence([fused])
    return service, fused


def _nli_result(claim_id: str, chunk_id: str, label: NLILabel = NLILabel.CONTRADICTION, score: float = 0.9) -> NLIResult:
    return NLIResult(
        atomic_claim_id=claim_id,
        evidence_chunk_id=chunk_id,
        label=label,
        score=score,
        premise="evidence text",
        hypothesis="claim text",
    )


def test_record_nli_results_populates_ledger() -> None:
    service, fused = _service_with_one_claim_and_fused_evidence()
    result = _nli_result("claim-1", fused.chunk_id)

    service.record_nli_results([result])

    assert service.ledger.nli_results == [result]


def test_record_nli_results_rejects_unknown_atomic_claim_id() -> None:
    service, fused = _service_with_one_claim_and_fused_evidence()
    result = _nli_result("claim-not-in-ledger", fused.chunk_id)

    with pytest.raises(ValueError, match="unknown atomic_claim_id"):
        service.record_nli_results([result])


def test_record_nli_results_rejects_chunk_with_no_fused_evidence() -> None:
    service, _ = _service_with_one_claim_and_fused_evidence()
    result = _nli_result("claim-1", "chunk-never-fused")

    with pytest.raises(ValueError, match="no corresponding entry in fused_evidence"):
        service.record_nli_results([result])


def test_record_nli_results_is_write_once() -> None:
    service, fused = _service_with_one_claim_and_fused_evidence()
    result = _nli_result("claim-1", fused.chunk_id)
    service.record_nli_results([result])

    with pytest.raises(ValueError, match="write-once"):
        service.record_nli_results([result])


def test_record_classification_populates_ledger() -> None:
    service, fused = _service_with_one_claim_and_fused_evidence()
    contradiction = Contradiction(atomic_claim_id="claim-1", evidence_chunk_id=fused.chunk_id, nli_score=0.9)

    service.record_classification([contradiction], [], [])

    assert service.ledger.contradictions == [contradiction]
    assert service.ledger.supporting_evidence == []
    assert service.ledger.unsupported_claims == []


def test_record_classification_rejects_unknown_atomic_claim_id_in_contradictions() -> None:
    service, fused = _service_with_one_claim_and_fused_evidence()
    contradiction = Contradiction(
        atomic_claim_id="claim-not-in-ledger", evidence_chunk_id=fused.chunk_id, nli_score=0.9
    )

    with pytest.raises(ValueError, match="unknown atomic_claim_id"):
        service.record_classification([contradiction], [], [])


def test_record_classification_rejects_unknown_atomic_claim_id_in_unsupported() -> None:
    service, _ = _service_with_one_claim_and_fused_evidence()

    with pytest.raises(ValueError, match="unknown atomic_claim_id"):
        service.record_classification([], [], ["claim-not-in-ledger"])


def test_record_classification_is_write_once() -> None:
    service, _ = _service_with_one_claim_and_fused_evidence()
    service.record_classification([], [], ["claim-1"])

    with pytest.raises(ValueError, match="write-once"):
        service.record_classification([], [], ["claim-1"])
