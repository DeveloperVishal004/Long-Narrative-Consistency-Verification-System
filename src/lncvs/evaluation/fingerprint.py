"""Deterministic ledger fingerprinting for lightweight evaluation reporting.

Excludes wall-clock fields (ledger_log, reasoning_trace timestamps) per
CLAUDE.md's accepted exception to full ledger reproducibility -- the same
(narrative, claim, variant) run must fingerprint identically across repeated
executions even though its audit-log timestamps differ.
"""

import hashlib

from lncvs.schemas import EvidenceLedger


def ledger_fingerprint(ledger: EvidenceLedger) -> str:
    """Hash the determinism-relevant content of a ledger.

    Covers claims, probe questions, retrieved/fused evidence, NLI results,
    classification, and the final verdict. Never includes ledger_log or
    reasoning_trace, which carry wall-clock timestamps.
    """
    parts = [
        ledger.original_claim,
        ledger.original_claim_id or "",
        ",".join(sorted(claim.claim_id for claim in ledger.atomic_claims)),
        ",".join(sorted(question.question_id for question in ledger.probe_questions)),
        ",".join(sorted(evidence.evidence_id for evidence in ledger.retrieved_evidence)),
        ",".join(
            sorted(
                f"{fused.atomic_claim_id}:{fused.chunk_id}:{fused.rrf_score:.6f}"
                for fused in ledger.fused_evidence
            )
        ),
        ",".join(
            sorted(
                f"{result.atomic_claim_id}:{result.evidence_chunk_id}:{result.label.value}:{result.score:.6f}"
                for result in ledger.nli_results
            )
        ),
        ",".join(sorted(c.evidence_chunk_id for c in ledger.contradictions)),
        ",".join(sorted(s.evidence_chunk_id for s in ledger.supporting_evidence)),
        ",".join(sorted(ledger.unsupported_claims)),
        ledger.final_verdict.verdict.value if ledger.final_verdict else "",
        ledger.final_verdict.fired_rule if ledger.final_verdict else "",
    ]
    digest_input = "|".join(parts).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
