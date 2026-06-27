"""FinalVerdict — the deterministic output of the rule engine."""

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas.enums import VerdictEnum


class FinalVerdict(BaseModel):
    """The deterministic verdict produced by the rule engine for a single claim.

    fired_rule must name exactly which rule produced the verdict (e.g.
    "rule_1_contradiction") so the decision is auditable. An LLM must never
    populate this model directly — only lncvs.rules.engine.RuleEngine
    implementations may construct one.
    """

    model_config = ConfigDict(frozen=True)

    verdict: VerdictEnum = Field(..., description="CONSISTENT, CONTRADICTORY, or INSUFFICIENT_EVIDENCE.")
    fired_rule: str = Field(..., min_length=1, description="Identifier of the rule that determined this verdict.")
    rationale: str = Field(..., min_length=1, description="Human-readable explanation of why this verdict fired.")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional aggregate confidence in the verdict."
    )
    contradicted_claim_ids: list[str] = Field(
        default_factory=list, description="Atomic claim IDs classified as CONTRADICTED."
    )
    unresolved_claim_ids: list[str] = Field(
        default_factory=list, description="Atomic claim IDs classified as UNRESOLVED (insufficient evidence)."
    )
