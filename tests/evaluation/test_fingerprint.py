"""ledger_fingerprint() tests: stability, timestamp-invariance, content-sensitivity."""

from datetime import datetime, timedelta, timezone

from lncvs.evaluation import ledger_fingerprint
from lncvs.ledger import LedgerService
from lncvs.schemas import AtomicClaim, EvidenceLedger, FinalVerdict, LedgerEvent, PipelineStage, VerdictEnum


def _ledger_with_one_claim() -> EvidenceLedger:
    ledger = EvidenceLedger(original_claim="A claim.")
    service = LedgerService(ledger)
    service.add_atomic_claim(AtomicClaim(claim_id="claim-1", text="part of the claim"))
    return ledger


def test_fingerprint_is_stable_across_identical_ledgers() -> None:
    ledger_a = _ledger_with_one_claim()
    ledger_b = _ledger_with_one_claim()
    assert ledger_fingerprint(ledger_a) == ledger_fingerprint(ledger_b)


def test_fingerprint_differs_for_different_atomic_claims() -> None:
    ledger_a = _ledger_with_one_claim()

    ledger_b = EvidenceLedger(original_claim="A claim.")
    LedgerService(ledger_b).add_atomic_claim(AtomicClaim(claim_id="claim-2", text="a different claim"))

    assert ledger_fingerprint(ledger_a) != ledger_fingerprint(ledger_b)


def test_fingerprint_is_invariant_to_ledger_log_timestamps() -> None:
    """The audit log's wall-clock timestamps must never affect the fingerprint --
    this is the accepted exception to full ledger reproducibility (CLAUDE.md)."""
    ledger_a = _ledger_with_one_claim()
    ledger_b = _ledger_with_one_claim()

    # Manually append a LedgerEvent with a very different timestamp to one ledger.
    ledger_b.ledger_log.append(
        LedgerEvent(
            event_id="manual-event",
            stage=PipelineStage.CLAIM_DECOMPOSITION,
            message="some other event",
            timestamp=datetime.now(timezone.utc) + timedelta(days=365),
        )
    )

    assert ledger_fingerprint(ledger_a) == ledger_fingerprint(ledger_b)


def test_fingerprint_includes_final_verdict() -> None:
    ledger_a = _ledger_with_one_claim()
    ledger_b = _ledger_with_one_claim()

    LedgerService(ledger_a).set_final_verdict(
        FinalVerdict(verdict=VerdictEnum.CONSISTENT, fired_rule="rule_3_all_supported", rationale="all supported")
    )
    LedgerService(ledger_b).set_final_verdict(
        FinalVerdict(verdict=VerdictEnum.CONTRADICTORY, fired_rule="rule_1_contradiction", rationale="contradicted")
    )

    assert ledger_fingerprint(ledger_a) != ledger_fingerprint(ledger_b)
