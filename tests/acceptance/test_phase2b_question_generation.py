"""Phase 2b acceptance test: decomposition + question generation, fully offline.

original_claim -> decompose -> generate questions per atomic claim -> record both in EvidenceLedger

Fully offline: no real LLM client in Phase 2b, so this test has no network
dependency and never skips.
"""

from lncvs.ledger import LedgerService
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig
from lncvs.schemas import EvidenceLedger
from tests.llm.fakes import FakeLLMClient

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'

# Only the "John used both hands" claim gets useful contradiction-seeking probes;
# the other two legitimately produce none.
QUESTIONS_BY_CLAIM_TEXT = {
    "John played piano": "[]",
    "John used both hands": '["Did John lose an arm?", "Did John suffer an injury?"]',
    "the event occurred in London": "[]",
}


def _run_phase2b_slice() -> tuple[EvidenceLedger, LedgerService]:
    ledger = EvidenceLedger(original_claim=ORIGINAL_CLAIM)
    service = LedgerService(ledger)

    decomp_config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DECOMPOSITION_RESPONSE), decomp_config)

    parent_id = make_source_claim_id(ORIGINAL_CLAIM)
    atomic_claims = decomposer.decompose(ORIGINAL_CLAIM)
    service.record_atomic_claims(parent_id, atomic_claims)

    question_config = QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model"))
    all_questions = []
    for claim in atomic_claims:
        scripted_response = QUESTIONS_BY_CLAIM_TEXT[claim.text]
        generator = LLMQuestionGenerator(FakeLLMClient(default_response=scripted_response), question_config)
        all_questions.extend(generator.generate(claim))
    service.record_probe_questions(all_questions)

    return ledger, service


def test_phase2b_slice_generates_questions_only_for_the_claim_that_warrants_them() -> None:
    _, service = _run_phase2b_slice()

    assert len(service.ledger.probe_questions) == 2
    assert {q.text for q in service.ledger.probe_questions} == {
        "Did John lose an arm?",
        "Did John suffer an injury?",
    }


def test_phase2b_slice_is_fully_traceable() -> None:
    ledger, service = _run_phase2b_slice()

    hands_claim = next(c for c in ledger.atomic_claims if c.text == "John used both hands")
    for question in service.ledger.probe_questions:
        assert question.atomic_claim_id == hands_claim.claim_id
        # claim -> original_claim_id -> original_claim
        assert hands_claim.parent_claim_id == ledger.original_claim_id
        assert ledger.original_claim_id == make_source_claim_id(ledger.original_claim)


def test_phase2b_slice_is_deterministic_end_to_end() -> None:
    def run_once() -> list[str]:
        _, service = _run_phase2b_slice()
        return [q.question_id for q in service.ledger.probe_questions]

    assert run_once() == run_once()
