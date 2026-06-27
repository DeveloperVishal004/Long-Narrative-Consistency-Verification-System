"""NLI verification: the NLIVerifier protocol and its cross-encoder-backed implementation.

NLIVerifier is deliberately evidence-level only: it returns one NLIResult
per (AtomicClaim, FusedEvidence) pair, with no claim-level aggregation or
threshold application. Per the Phase 5 architecture review, claim-level
aggregation requires thresholds, and thresholds belong to exactly one place
-- lncvs.rules.classification.classify(), called from ThresholdRuleEngine.
Keeping this layer aggregation-free preserves the same invariant CLAUDE.md
already enforces for the LLM-backed agents: the model never decides.

The premise/hypothesis direction is fixed and must never be swapped:
premise = evidence text, hypothesis = atomic claim text.
"""

from lncvs.reasoning.nli.model import NLIModel
from lncvs.schemas import AtomicClaim, FusedEvidence, NLIResult


class CrossEncoderNLIVerifier:
    """NLIVerifier backed by an injected NLIModel.

    Holds no state beyond its injected dependency. Determinism rests
    entirely on the injected NLIModel (a real CrossEncoder is already
    deterministic in eval mode; FakeNLIModel is deterministic by
    construction; CachingNLIModel adds throughput on top of either).
    """

    def __init__(self, model: NLIModel) -> None:
        self._model = model

    def verify(self, claim: AtomicClaim, evidence: list[FusedEvidence]) -> list[NLIResult]:
        """Return one NLIResult per evidence record, fixed in the premise=evidence direction.

        An empty evidence list returns an empty list -- this is the correct
        input for a claim with no retrieved evidence, not an error. It is
        what ultimately routes that claim to UNRESOLVED rather than a false
        CONTRADICTORY.
        """
        results = []
        for record in evidence:
            prediction = self._model.predict(premise=record.text, hypothesis=claim.text)
            results.append(
                NLIResult(
                    atomic_claim_id=claim.claim_id,
                    evidence_chunk_id=record.chunk_id,
                    label=prediction.label,
                    score=prediction.score,
                    premise=record.text,
                    hypothesis=claim.text,
                )
            )
        return results
