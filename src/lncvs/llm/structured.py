"""StructuredLLMClient protocol and its completion DTO (Phase 8 / G2).

Decision 1 of the G2 extraction-interface freeze: this is an *additive*
extension. LLMClient/LLMCompletion/complete() in lncvs.llm.base are
unmodified and continue serving Claim Decomposition and Question
Generation exactly as before. StructuredLLMClient exists solely for the
schema-enforced structured extraction Phase 8 introduces; nothing in
reasoning/decomposition or reasoning/questions depends on it.

StructuredCompletion.data is the raw, provider-validated-shape JSON
response as a plain dict -- the same intentional "untyped at this one
generic boundary" status LLMCompletion.text already has as a bare string.
lncvs.llm is a narrow leaf module with zero knowledge of graph-specific
schemas (it must not import from lncvs.graph or lncvs.schemas's graph
types, the same dependency-direction rule LLMCompletion already respects),
so it cannot return an already-typed WindowExtraction here. The contract
this DTO carries is the same one parse_decomposition_response() already
enforces for LLMCompletion.text: the *caller* must validate this payload
into a typed model immediately upon receipt and let the untyped dict go no
further -- see graph.llm_extraction.service, the only consumer.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class StructuredCompletion(BaseModel):
    """A single structured completion, with provenance back to the model
    and provider call that produced it."""

    model_config = ConfigDict(frozen=True)

    data: dict = Field(..., description="Raw provider JSON response, conforming to the requested schema.")
    model_fingerprint: str = Field(
        ..., min_length=1, description="Fingerprint of the LLMConfig used to produce this completion."
    )
    response_id: str = Field(
        ..., min_length=1, description="Provider's own call identifier, recorded for build-manifest audit."
    )


@runtime_checkable
class StructuredLLMClient(Protocol):
    """Dependency-injection point for schema-enforced structured completions.

    Any implementation (real provider, fake/scripted test double) must
    satisfy this shape. graph.llm_extraction depends on this protocol,
    never on a concrete provider SDK.
    """

    def complete_structured(self, prompt: str, response_schema: dict) -> StructuredCompletion:
        """Return a structured completion for prompt, conforming to response_schema."""
        ...
