"""Pure helper: group claim-linked RetrievedEvidence by atomic_claim_id.

Produces a transient view for NLI prep — this grouping is never stored in
the ledger (which keeps a flat list, not a dict, per CLAUDE.md's "no dicts
in the ledger" rule). Callers needing per-claim evidence call this on
demand against the ledger's flat retrieved_evidence list.
"""

from lncvs.schemas import RetrievedEvidence


def group_evidence_by_claim(evidence: list[RetrievedEvidence]) -> dict[str, list[RetrievedEvidence]]:
    """Group evidence records by atomic_claim_id.

    Raises ValueError if any record has atomic_claim_id=None — such a
    record has not been stamped by RetrievalOrchestrator yet and should
    never reach this function (or the ledger).
    """
    grouped: dict[str, list[RetrievedEvidence]] = {}
    for item in evidence:
        if item.atomic_claim_id is None:
            raise ValueError(
                f"Cannot group evidence {item.evidence_id!r}: atomic_claim_id is not set. "
                "Evidence must be stamped by RetrievalOrchestrator before grouping."
            )
        grouped.setdefault(item.atomic_claim_id, []).append(item)
    return grouped
