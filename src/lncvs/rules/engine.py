"""Deterministic Rule Engine — interface and contracts.

This module defines the contract only. The concrete rule logic (Rule 1: any
contradicted claim -> CONTRADICTORY; Rule 2: any unresolved claim ->
INSUFFICIENT_EVIDENCE; Rule 3: all claims supported -> CONSISTENT) ships in
lncvs.rules.threshold_engine.ThresholdRuleEngine (Phase 5), once NLI
verification exists to populate a ledger that can be meaningfully evaluated
against the PROJECT_SPEC.md Section 14 dummy case.

Per CLAUDE.md, an LLM must never produce a FinalVerdict. Every RuleEngine
implementation must be a pure, deterministic function of
(EvidenceLedger, RuleEngineConfig): the same inputs must always produce
the same FinalVerdict.
"""

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from lncvs.schemas import EvidenceLedger, FinalVerdict


class ClaimStatus(str, Enum):
    """Per-atomic-claim classification computed by a RuleEngine before verdict rules are applied.

    CONTRADICTED always dominates SUPPORTED for the same claim: a
    contradiction outweighs coexisting support. UNRESOLVED covers claims
    with no NLI result clearing either threshold, or no evidence at all —
    this is what drives INSUFFICIENT_EVIDENCE rather than a false
    CONTRADICTORY verdict.
    """

    CONTRADICTED = "CONTRADICTED"
    SUPPORTED = "SUPPORTED"
    UNRESOLVED = "UNRESOLVED"


class RuleEngineConfig(BaseModel):
    """Confidence thresholds gating the rule engine's hard-fail rules.

    These must be configuration, not hard-coded constants, so that a
    low-confidence NLI result cannot silently fire a hard-fail rule.
    """

    model_config = ConfigDict(frozen=True)

    contradiction_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum NLI contradiction score required to classify a claim as CONTRADICTED.",
    )
    entailment_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum NLI entailment score required to classify a claim as SUPPORTED.",
    )
    consistency_requires_entailment: bool = Field(
        default=True,
        description=(
            "Consistency policy. True (default, strict): a claim with no entailing AND "
            "no contradicting evidence is UNRESOLVED, routing the verdict to "
            "INSUFFICIENT_EVIDENCE -- the original three-verdict semantics. False "
            "(lenient): the absence of any contradiction is sufficient for CONSISTENT; "
            "unresolved (neutral-only) claims do not block a CONSISTENT verdict. The "
            "lenient policy fits datasets whose 'consistent' label means 'not "
            "contradicted by the source' rather than 'positively entailed by a single "
            "retrieved chunk' -- a setting where a cross-encoder almost never emits "
            "ENTAILMENT for a paraphrased claim against one chunk, making strict "
            "CONSISTENT structurally unreachable. Default preserves existing behavior."
        ),
    )


class RuleEngine(ABC):
    """Deterministic mapping from a completed EvidenceLedger to a FinalVerdict.

    Implementations must:
      - never call an LLM or any non-deterministic component;
      - be a pure function of the ledger's content and self.config;
      - record which rule fired in the returned FinalVerdict.fired_rule.

    This class defines the contract only. ThresholdRuleEngine is the
    concrete Phase 5 implementation.
    """

    def __init__(self, config: RuleEngineConfig) -> None:
        self._config = config

    @property
    def config(self) -> RuleEngineConfig:
        return self._config

    @abstractmethod
    def evaluate(self, ledger: EvidenceLedger) -> FinalVerdict:
        """Compute the deterministic final verdict for a completed ledger.

        Must raise, not guess, if the ledger is missing data the engine
        needs (e.g. no atomic claims recorded at all) rather than silently
        producing a verdict the data doesn't support.
        """
        raise NotImplementedError
