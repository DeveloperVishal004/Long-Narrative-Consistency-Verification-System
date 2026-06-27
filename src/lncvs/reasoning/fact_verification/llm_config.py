"""LLM fact-verification configuration. Mirrors
lncvs.graph.llm_extraction.config.ExtractionConfig exactly."""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.reasoning.fact_verification.llm_prompts import PROMPT_VERSION
from lncvs.reasoning.fact_verification.llm_schema import SCHEMA_VERSION


class FactVerificationConfig(BaseModel):
    """Configures a single LLMFactVerifier.

    prompt_version and schema_version are audit provenance, not cache keys
    themselves -- CachingStructuredLLMClient's cache key already includes
    schema_version directly and the rendered prompt text, which makes a
    template edit self-invalidating without separate version-bumping
    discipline. Mirrors ExtractionConfig's identical role.
    """

    model_config = ConfigDict(frozen=True)

    prompt_version: str = Field(default=PROMPT_VERSION, min_length=1, description="Hash of the prompt templates in use.")
    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1, description="Hash of the fact-verification JSON schema in use.")
