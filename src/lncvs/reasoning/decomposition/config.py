"""Claim decomposition configuration."""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition.prompts import PROMPT_VERSION


class DecompositionConfig(BaseModel):
    """Configures a single LLMClaimDecomposer.

    prompt_version is provenance metadata (recorded for audit/debugging),
    not a cache key — see lncvs.llm.cache and lncvs.reasoning.decomposition.prompts
    for why the rendered prompt text already makes cache invalidation
    automatic on template changes.
    """

    model_config = ConfigDict(frozen=True)

    llm_config: LLMConfig = Field(..., description="Configuration for the underlying LLMClient.")
    prompt_version: str = Field(
        default=PROMPT_VERSION, min_length=1, description="Hash of the prompt template in use, for provenance."
    )
    max_atomic_claims: int = Field(
        default=10, gt=0, description="Maximum number of atomic claims a single decomposition may produce."
    )
