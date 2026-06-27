"""Integration: decomposition output -> LedgerService.record_atomic_claims."""

import pytest

from lncvs.ledger import LedgerService
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.schemas import EvidenceLedger
from tests.llm.fakes import FakeLLMClient

DUMMY_CLAIM = "John played a two-handed piano piece in London."
DUMMY_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'


def _decomposer() -> LLMClaimDecomposer:
    config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
    return LLMClaimDecomposer(FakeLLMClient(default_response=DUMMY_RESPONSE), config)


def test_record_atomic_claims_populates_ledger() -> None:
    ledger = EvidenceLedger(original_claim=DUMMY_CLAIM)
    service = LedgerService(ledger)
    decomposer = _decomposer()

    parent_id = make_source_claim_id(DUMMY_CLAIM)
    claims = decomposer.decompose(DUMMY_CLAIM)
    service.record_atomic_claims(parent_id, claims)

    assert len(service.ledger.atomic_claims) == 3
    assert service.ledger.original_claim_id == parent_id


def test_record_atomic_claims_appends_to_ledger_log() -> None:
    ledger = EvidenceLedger(original_claim=DUMMY_CLAIM)
    service = LedgerService(ledger)
    decomposer = _decomposer()

    parent_id = make_source_claim_id(DUMMY_CLAIM)
    claims = decomposer.decompose(DUMMY_CLAIM)
    service.record_atomic_claims(parent_id, claims)

    assert len(service.ledger.ledger_log) == 1
    assert "3 atomic claim" in service.ledger.ledger_log[0].message


def test_every_atomic_claim_traces_back_to_the_original_claim() -> None:
    ledger = EvidenceLedger(original_claim=DUMMY_CLAIM)
    service = LedgerService(ledger)
    decomposer = _decomposer()

    parent_id = make_source_claim_id(DUMMY_CLAIM)
    claims = decomposer.decompose(DUMMY_CLAIM)
    service.record_atomic_claims(parent_id, claims)

    for claim in service.ledger.atomic_claims:
        assert claim.parent_claim_id == service.ledger.original_claim_id
    assert service.ledger.original_claim_id == make_source_claim_id(service.ledger.original_claim)


def test_record_atomic_claims_is_write_once() -> None:
    ledger = EvidenceLedger(original_claim=DUMMY_CLAIM)
    service = LedgerService(ledger)
    decomposer = _decomposer()

    parent_id = make_source_claim_id(DUMMY_CLAIM)
    claims = decomposer.decompose(DUMMY_CLAIM)
    service.record_atomic_claims(parent_id, claims)

    with pytest.raises(ValueError, match="write-once"):
        service.record_atomic_claims(parent_id, claims)
