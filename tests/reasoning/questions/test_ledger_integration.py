"""Integration: question generation output -> LedgerService.record_probe_questions."""

import pytest

from lncvs.ledger import LedgerService
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig
from lncvs.schemas import EvidenceLedger, ProbeQuestion
from tests.llm.fakes import FakeLLMClient

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'
QUESTIONS_RESPONSE = '["Did John lose an arm?", "Did John suffer an injury?"]'


def _ledger_with_decomposed_claims() -> tuple[EvidenceLedger, LedgerService]:
    ledger = EvidenceLedger(original_claim=ORIGINAL_CLAIM)
    service = LedgerService(ledger)
    decomp_config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DECOMPOSITION_RESPONSE), decomp_config)

    parent_id = make_source_claim_id(ORIGINAL_CLAIM)
    claims = decomposer.decompose(ORIGINAL_CLAIM)
    service.record_atomic_claims(parent_id, claims)
    return ledger, service


def test_record_probe_questions_populates_ledger() -> None:
    ledger, service = _ledger_with_decomposed_claims()
    generator = LLMQuestionGenerator(
        FakeLLMClient(default_response=QUESTIONS_RESPONSE),
        QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
    )

    all_questions = [q for claim in ledger.atomic_claims for q in generator.generate(claim)]
    service.record_probe_questions(all_questions)

    assert len(service.ledger.probe_questions) == len(all_questions)
    assert len(service.ledger.ledger_log) == 2  # one decomposition event, one question-generation event


def test_every_question_traces_to_a_known_atomic_claim() -> None:
    ledger, service = _ledger_with_decomposed_claims()
    generator = LLMQuestionGenerator(
        FakeLLMClient(default_response=QUESTIONS_RESPONSE),
        QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
    )

    all_questions = [q for claim in ledger.atomic_claims for q in generator.generate(claim)]
    service.record_probe_questions(all_questions)

    known_claim_ids = {c.claim_id for c in service.ledger.atomic_claims}
    for question in service.ledger.probe_questions:
        assert question.atomic_claim_id in known_claim_ids


def test_record_probe_questions_rejects_unknown_atomic_claim_id() -> None:
    _, service = _ledger_with_decomposed_claims()
    rogue_question = ProbeQuestion(
        question_id="q-rogue", atomic_claim_id="claim-not-in-ledger", text="Did John lose an arm?"
    )

    with pytest.raises(ValueError, match="unknown atomic_claim_id"):
        service.record_probe_questions([rogue_question])


def test_record_probe_questions_is_write_once() -> None:
    ledger, service = _ledger_with_decomposed_claims()
    generator = LLMQuestionGenerator(
        FakeLLMClient(default_response=QUESTIONS_RESPONSE),
        QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
    )

    all_questions = [q for claim in ledger.atomic_claims for q in generator.generate(claim)]
    service.record_probe_questions(all_questions)

    with pytest.raises(ValueError, match="write-once"):
        service.record_probe_questions(all_questions)


def test_record_probe_questions_accepts_empty_list() -> None:
    """Every claim legitimately yielding zero questions is a valid outcome."""
    _, service = _ledger_with_decomposed_claims()

    service.record_probe_questions([])

    assert service.ledger.probe_questions == []
