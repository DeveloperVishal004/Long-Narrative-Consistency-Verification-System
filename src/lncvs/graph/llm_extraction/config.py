"""LLM window-extraction configuration."""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.graph.llm_extraction.json_schema import SCHEMA_VERSION
from lncvs.graph.llm_extraction.prompts import PROMPT_VERSION


class ExtractionConfig(BaseModel):
    """Configures a single LLMWindowExtractor.

    prompt_version and schema_version are audit provenance (recorded for
    a later stage's build manifest), not cache keys themselves --
    CachingStructuredLLMClient's cache key already includes schema_version
    directly and the rendered prompt text, which makes a template edit
    self-invalidating without separate version-bumping discipline. Mirrors
    DecompositionConfig's identical prompt_version-as-provenance role.
    """

    model_config = ConfigDict(frozen=True)

    prompt_version: str = Field(default=PROMPT_VERSION, min_length=1, description="Hash of the prompt templates in use.")
    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1, description="Hash of the extraction JSON schema in use.")
