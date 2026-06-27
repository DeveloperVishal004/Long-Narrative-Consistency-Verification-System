"""Phase 2a acceptance test: the full decomposition slice, fully offline.

original_claim -> decompose (via FakeLLMClient) -> record in EvidenceLedger

Mirrors the role tests/acceptance/test_phase1_vertical_slice.py played for
Phase 1, but stays entirely offline: there is no real LLM client in Phase
2a, so this test has no network dependency and never skips.
"""

from lncvs.ledger import LedgerService
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.schemas import EvidenceLedger
from tests.llm.fakes import FakeLLMClient

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'


def test_phase2_decomposition_slice_populates_a_traceable_ledger() -> None:
    ledger = EvidenceLedger(original_claim=ORIGINAL_CLAIM)
    service = LedgerService(ledger)

    config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DECOMPOSITION_RESPONSE), config)

    parent_claim_id = make_source_claim_id(ORIGINAL_CLAIM)
    atomic_claims = decomposer.decompose(ORIGINAL_CLAIM)
    service.record_atomic_claims(parent_claim_id, atomic_claims)

    assert len(service.ledger.atomic_claims) == 3
    assert {c.text for c in service.ledger.atomic_claims} == {
        "John played piano",
        "John used both hands",
        "the event occurred in London",
    }
    assert service.ledger.original_claim_id == parent_claim_id
    for claim in service.ledger.atomic_claims:
        assert claim.parent_claim_id == service.ledger.original_claim_id


def test_phase2_decomposition_slice_is_deterministic_end_to_end() -> None:
    """Running the entire slice twice, from two fresh ledgers, must produce identical claim IDs."""

    def run_once() -> list[str]:
        ledger = EvidenceLedger(original_claim=ORIGINAL_CLAIM)
        service = LedgerService(ledger)
        config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
        decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DECOMPOSITION_RESPONSE), config)

        parent_claim_id = make_source_claim_id(ORIGINAL_CLAIM)
        atomic_claims = decomposer.decompose(ORIGINAL_CLAIM)
        service.record_atomic_claims(parent_claim_id, atomic_claims)
        return [c.claim_id for c in service.ledger.atomic_claims]

    assert run_once() == run_once()
