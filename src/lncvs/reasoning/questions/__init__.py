"""Question Generation: produces retrieval-oriented probe questions from atomic claims."""

from lncvs.reasoning.questions.config import QuestionGenerationConfig
from lncvs.reasoning.questions.identity import make_question_id
from lncvs.reasoning.questions.parser import parse_question_response
from lncvs.reasoning.questions.prompts import PROMPT_VERSION, render_question_generation_prompt
from lncvs.reasoning.questions.service import LLMQuestionGenerator, QuestionGenerator

__all__ = [
    "LLMQuestionGenerator",
    "PROMPT_VERSION",
    "QuestionGenerationConfig",
    "QuestionGenerator",
    "make_question_id",
    "parse_question_response",
    "render_question_generation_prompt",
]
