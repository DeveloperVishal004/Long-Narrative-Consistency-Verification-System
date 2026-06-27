"""Question generation configuration."""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.llm import LLMConfig
from lncvs.reasoning.questions.prompts import PROMPT_VERSION


class QuestionGenerationConfig(BaseModel):
    """Configures a single LLMQuestionGenerator.

    prompt_version is provenance metadata, not a cache key — see
    lncvs.llm.cache and lncvs.reasoning.decomposition.config for the
    rationale (identical to decomposition's).
    """

    model_config = ConfigDict(frozen=True)

    llm_config: LLMConfig = Field(..., description="Configuration for the underlying LLMClient.")
    prompt_version: str = Field(
        default=PROMPT_VERSION, min_length=1, description="Hash of the prompt template in use, for provenance."
    )
    max_questions_per_claim: int = Field(
        default=5, gt=0, description="Maximum number of probe questions a single atomic claim may produce."
    )
