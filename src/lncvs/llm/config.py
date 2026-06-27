"""LLM client configuration."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    """Configures which LLM an LLMClient implementation calls and how it runs.

    fingerprint() covers only model identity and sampling parameters.
    Prompt-template changes do not need to be folded in here: the cache key
    in CachingLLMClient is (fingerprint, rendered_prompt), and the rendered
    prompt already contains the template text — so a template edit changes
    the prompt string and therefore the cache key automatically.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(..., min_length=1, description="Identifier of the LLM to call.")
    temperature: float = Field(default=0.0, ge=0.0, description="Sampling temperature.")
    max_tokens: int = Field(default=1024, gt=0, description="Maximum tokens in the completion.")

    def fingerprint(self) -> str:
        """Deterministic identifier for this exact configuration.

        Used to namespace cached completions so a config change (different
        model, temperature, or max_tokens) can never silently serve a
        completion produced under a different configuration.
        """
        digest_input = f"{self.model_name}:{self.temperature}:{self.max_tokens}".encode("utf-8")
        return hashlib.sha256(digest_input).hexdigest()[:16]
