"""Question generation: the QuestionGenerator protocol and its LLM-backed implementation."""

from typing import Protocol, runtime_checkable

from lncvs.llm import LLMClient
from lncvs.reasoning.questions.config import QuestionGenerationConfig
from lncvs.reasoning.questions.parser import parse_question_response
from lncvs.reasoning.questions.prompts import render_question_generation_prompt
from lncvs.schemas import AtomicClaim, ProbeQuestion


@runtime_checkable
class QuestionGenerator(Protocol):
    """Contract for generating retrieval-oriented probe questions for a single atomic claim."""

    def generate(self, atomic_claim: AtomicClaim) -> list[ProbeQuestion]:
        """Return the probe questions generated for atomic_claim. May be empty."""
        ...


class LLMQuestionGenerator:
    """QuestionGenerator backed by an injected LLMClient.

    One call per atomic claim, so each call is independently cacheable by
    claim text via CachingLLMClient. Holds no state beyond its injected
    dependencies.
    """

    def __init__(self, llm_client: LLMClient, config: QuestionGenerationConfig) -> None:
        self._llm_client = llm_client
        self._config = config

    def generate(self, atomic_claim: AtomicClaim) -> list[ProbeQuestion]:
        prompt = render_question_generation_prompt(atomic_claim.text)
        completion = self._llm_client.complete(prompt)

        return parse_question_response(
            completion.text, atomic_claim.claim_id, self._config.max_questions_per_claim
        )
