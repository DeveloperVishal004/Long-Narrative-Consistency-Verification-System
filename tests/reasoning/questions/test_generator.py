"""LLMQuestionGenerator tests — offline via FakeLLMClient."""

import pytest

from lncvs.llm import CachingLLMClient, InMemoryLLMCache, LLMConfig
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig, QuestionGenerator
from lncvs.schemas import AtomicClaim
from tests.llm.fakes import FakeLLMClient

ATOMIC_CLAIM = AtomicClaim(claim_id="claim-1", text="John used both hands", parent_claim_id="parent-1", index=1)
QUESTIONS_RESPONSE = '["Did John lose an arm?", "Did John suffer an injury?"]'


def _config(max_questions_per_claim: int = 10) -> QuestionGenerationConfig:
    return QuestionGenerationConfig(
        llm_config=LLMConfig(model_name="fake-model"), max_questions_per_claim=max_questions_per_claim
    )


def test_generator_satisfies_question_generator_protocol() -> None:
    generator = LLMQuestionGenerator(FakeLLMClient(default_response=QUESTIONS_RESPONSE), _config())
    assert isinstance(generator, QuestionGenerator)


def test_generate_returns_expected_questions_for_dummy_claim() -> None:
    generator = LLMQuestionGenerator(FakeLLMClient(default_response=QUESTIONS_RESPONSE), _config())

    questions = generator.generate(ATOMIC_CLAIM)

    assert [q.text for q in questions] == ["Did John lose an arm?", "Did John suffer an injury?"]


def test_generate_sets_atomic_claim_id_correctly() -> None:
    generator = LLMQuestionGenerator(FakeLLMClient(default_response=QUESTIONS_RESPONSE), _config())

    questions = generator.generate(ATOMIC_CLAIM)

    assert all(q.atomic_claim_id == "claim-1" for q in questions)


def test_generate_returns_empty_list_when_model_legitimately_produces_none() -> None:
    generator = LLMQuestionGenerator(FakeLLMClient(default_response="[]"), _config())

    questions = generator.generate(ATOMIC_CLAIM)

    assert questions == []


def test_generate_is_deterministic_across_fresh_generator_instances() -> None:
    first_generator = LLMQuestionGenerator(FakeLLMClient(default_response=QUESTIONS_RESPONSE), _config())
    second_generator = LLMQuestionGenerator(FakeLLMClient(default_response=QUESTIONS_RESPONSE), _config())

    first_questions = first_generator.generate(ATOMIC_CLAIM)
    second_questions = second_generator.generate(ATOMIC_CLAIM)

    assert [q.question_id for q in first_questions] == [q.question_id for q in second_questions]


def test_generate_with_caching_llm_client_invokes_wrapped_client_once_on_repeat() -> None:
    fake = FakeLLMClient(default_response=QUESTIONS_RESPONSE)
    config = _config()
    caching_client = CachingLLMClient(fake, InMemoryLLMCache(), config.llm_config)
    generator = LLMQuestionGenerator(caching_client, config)

    generator.generate(ATOMIC_CLAIM)
    generator.generate(ATOMIC_CLAIM)

    assert len(fake.calls) == 1


def test_generate_respects_max_questions_per_claim_guard() -> None:
    generator = LLMQuestionGenerator(
        FakeLLMClient(default_response=QUESTIONS_RESPONSE), _config(max_questions_per_claim=1)
    )
    with pytest.raises(ValueError, match="max_questions_per_claim"):
        generator.generate(ATOMIC_CLAIM)


def test_llm_client_error_propagates_without_silent_fallback() -> None:
    """No scripted/default response configured -> FakeLLMClient raises -> must propagate, not return []."""
    generator = LLMQuestionGenerator(FakeLLMClient(), _config())
    with pytest.raises(ValueError, match="no scripted response"):
        generator.generate(ATOMIC_CLAIM)


def test_generate_filters_declarative_statements_without_raising() -> None:
    """A claim where the model returns only declarative statements still succeeds with []
    (the empty-results-are-valid contract), not an error."""
    generator = LLMQuestionGenerator(
        FakeLLMClient(default_response='["John lost his left arm in 2010."]'), _config()
    )

    questions = generator.generate(ATOMIC_CLAIM)

    assert questions == []
