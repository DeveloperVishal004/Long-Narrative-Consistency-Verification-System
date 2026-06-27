"""Raw LLM fact-verdict DTO and its pure parser (Phase H3).

RawFactVerdict is an intermediate type, distinct from the final
FactVerification domain type in schemas/ -- it exists only between the
raw structured-completion dict and llm_verifier.py's quote-verification
step, which consumes and discards it. Mirrors
lncvs.graph.llm_extraction.schema's RawEntityMention precedent: two
independent, hand-written artifacts (this DTO and llm_schema.py's literal
JSON schema) rather than one generated from the other, for the same
reason documented there.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class RawFactVerdict(BaseModel):
    """The literal shape of one fact-verification completion."""

    model_config = ConfigDict(frozen=True)

    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_MENTIONED"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    quotes: tuple[str, ...] = Field(default=())
    explanation: str = Field(..., min_length=1)


def parse_fact_verdict(raw: dict) -> RawFactVerdict:
    """Pure parser: raw structured-completion dict -> RawFactVerdict.

    No LLM calls happen here -- deterministic given identical input. Never
    lets a bare pydantic.ValidationError escape this module, mirroring
    lncvs.graph.llm_extraction.service.parse_window_extraction exactly.
    """
    try:
        return RawFactVerdict.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Fact verification response failed schema validation: {exc}") from exc
