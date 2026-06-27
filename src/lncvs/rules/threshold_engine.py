"""ThresholdRuleEngine — the Phase 5 concrete RuleEngine.

Per Design B (confirmed in the Phase 5 architecture review), threshold
application lives in lncvs.rules.classification.classify() and only there.
evaluate() reads ledger.nli_results and ledger.atomic_claims -- never the
derived contradictions/supporting_evidence/unsupported_claims ledger
fields, which are an explainability co-product written separately by the
Phase 5 driver via LedgerService.record_classification(), using this same
classify() helper. Because classify() is pure, both call sites agree by
construction; there is no double-threshold path and no risk of the engine
reading back its own derived output.
"""

from lncvs.rules.classification import classify
from lncvs.rules.engine import ClaimStatus, RuleEngine
from lncvs.schemas import EvidenceLedger, FinalVerdict, VerdictEnum


class ThresholdRuleEngine(RuleEngine):
    """Deterministic verdict construction over per-claim NLI classification.

    Rule 1: any CONTRADICTED claim -> CONTRADICTORY.
    Rule 2: any UNRESOLVED claim (including zero evidence) -> INSUFFICIENT_EVIDENCE.
    Rule 3: all claims SUPPORTED -> CONSISTENT.
    Rule 1 is checked before Rule 2: a claim set with both a contradiction
    and an unresolved claim is CONTRADICTORY, since a confirmed
    contradiction is a stronger, more specific signal than a coverage gap.
    """

    def evaluate(self, ledger: EvidenceLedger) -> FinalVerdict:
        if not ledger.atomic_claims:
            raise ValueError("Cannot evaluate a ledger with no atomic_claims recorded.")

        claim_ids = [claim.claim_id for claim in ledger.atomic_claims]
        outcome = classify(ledger.nli_results, claim_ids, self.config)

        contradicted_ids = [
            claim_id for claim_id, status in outcome.statuses.items() if status is ClaimStatus.CONTRADICTED
        ]
        unresolved_ids = [
            claim_id for claim_id, status in outcome.statuses.items() if status is ClaimStatus.UNRESOLVED
        ]

        if contradicted_ids:
            return FinalVerdict(
                verdict=VerdictEnum.CONTRADICTORY,
                fired_rule="rule_1_contradiction",
                rationale=f"{len(contradicted_ids)} atomic claim(s) contradicted by evidence.",
                contradicted_claim_ids=contradicted_ids,
                unresolved_claim_ids=unresolved_ids,
            )
        if unresolved_ids and self.config.consistency_requires_entailment:
            return FinalVerdict(
                verdict=VerdictEnum.INSUFFICIENT_EVIDENCE,
                fired_rule="rule_2_unresolved",
                rationale=f"{len(unresolved_ids)} atomic claim(s) have no entailing or contradicting evidence.",
                unresolved_claim_ids=unresolved_ids,
            )
        # Lenient consistency policy (consistency_requires_entailment=False):
        # no contradiction was found, so the claim set is CONSISTENT even if
        # some claims are merely unresolved (neutral). unresolved_claim_ids
        # is still recorded for auditability.
        return FinalVerdict(
            verdict=VerdictEnum.CONSISTENT,
            fired_rule="rule_3_all_supported" if not unresolved_ids else "rule_3_no_contradiction",
            rationale=(
                "All atomic claims are supported by entailing evidence."
                if not unresolved_ids
                else f"No contradiction found; {len(unresolved_ids)} claim(s) unresolved but not blocking under lenient policy."
            ),
            unresolved_claim_ids=unresolved_ids,
        )
