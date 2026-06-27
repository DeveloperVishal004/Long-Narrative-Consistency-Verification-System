"""LLMClient protocol and its completion DTO.

No vendor SDK types appear in this module. Concrete clients (added in a
later phase) implement this protocol in an isolated file, exactly as
chromadb is confined to indexing/chroma_index.py.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class LLMCompletion(BaseModel):
    """A single LLM completion, with provenance back to the model that produced it."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., min_length=1, description="The raw completion text.")
    model_fingerprint: str = Field(
        ..., min_length=1, description="Fingerprint of the LLMConfig used to produce this completion."
    )


@runtime_checkable
class LLMClient(Protocol):
    """Dependency-injection point for LLM calls.

    Any implementation (real provider, fake/scripted test double) must
    satisfy this shape. Decomposition (and later Question Generation)
    depend on this protocol, never on a concrete provider SDK. NLI does not
    use this protocol — it is a dedicated cross-encoder model with its own
    abstraction, added separately.
    """

    def complete(self, prompt: str) -> LLMCompletion:
        """Return a completion for prompt."""
        ...
