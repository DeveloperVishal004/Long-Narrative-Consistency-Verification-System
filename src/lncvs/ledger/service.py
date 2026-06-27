"""LedgerService — the only sanctioned interface for mutating an EvidenceLedger.

Node code must never assign directly to EvidenceLedger fields. Direct
assignment bypasses the audit log (ledger_log) and the invariants enforced
here (e.g. write-once final verdicts, deduplicated unsupported-claim
tracking). Every mutation method appends a LedgerEvent so the full history
of how a ledger reached its final state is reconstructable.
"""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from lncvs.schemas import (
    AtomicClaim,
    Contradiction,
    EvidenceLedger,
    FinalVerdict,
    FusedEvidence,
    LedgerEvent,
    NLIResult,
    PipelineStage,
    ProbeQuestion,
    ReasoningStep,
    RetrievalQuery,
    RetrievedEvidence,
    SupportingEvidence,
)

logger = logging.getLogger(__name__)


class LedgerService:
    """Typed mutation API wrapping a single EvidenceLedger instance."""

    def __init__(self, ledger: EvidenceLedger) -> None:
        self._ledger = ledger

    @property
    def ledger(self) -> EvidenceLedger:
        """Read-only access to the underlying ledger for downstream consumers (e.g. the rule engine)."""
        return self._ledger

    def add_atomic_claim(self, claim: AtomicClaim) -> None:
        """Record a decomposed atomic claim."""
        self._ledger.atomic_claims.append(claim)
        self._log_event(PipelineStage.CLAIM_DECOMPOSITION, f"Added atomic claim {claim.claim_id!r}")

    def record_atomic_claims(self, original_claim_id: str, claims: list[AtomicClaim]) -> None:
        """Record the full decomposition output for this ledger's original_claim.

        Write-once: decomposition is meant to run exactly once per ledger.
        Raises ValueError if atomic claims have already been recorded.
        """
        if self._ledger.original_claim_id is not None:
            raise ValueError(
                "Atomic claims have already been recorded on this ledger; decomposition is write-once."
            )
        self._ledger.original_claim_id = original_claim_id
        self._ledger.atomic_claims.extend(claims)
        self._log_event(
            PipelineStage.CLAIM_DECOMPOSITION,
            f"Recorded {len(claims)} atomic claim(s) for original_claim_id {original_claim_id!r}",
        )

    def add_probe_question(self, question: ProbeQuestion) -> None:
        """Record a generated probe question."""
        self._ledger.probe_questions.append(question)
        self._log_event(PipelineStage.QUESTION_GENERATION, f"Added probe question {question.question_id!r}")

    def record_probe_questions(self, questions: list[ProbeQuestion]) -> None:
        """Record the full question-generation output across all atomic claims.

        Write-once: question generation is meant to run exactly once per
        ledger. Raises ValueError if probe questions have already been
        recorded, or if any question references an atomic_claim_id not
        present in this ledger's atomic_claims (a sign decomposition has not
        run yet, or ran against a different ledger).

        Known limitation: write-once is detected via list non-emptiness.
        If every atomic claim legitimately yields zero probe questions, a
        second call also passing [] cannot be distinguished from the first
        and will not raise. Revisit with an explicit sentinel if this
        matters in practice.
        """
        if self._ledger.probe_questions:
            raise ValueError(
                "Probe questions have already been recorded on this ledger; question generation is write-once."
            )

        known_claim_ids = {claim.claim_id for claim in self._ledger.atomic_claims}
        for question in questions:
            if question.atomic_claim_id not in known_claim_ids:
                raise ValueError(
                    f"ProbeQuestion references unknown atomic_claim_id {question.atomic_claim_id!r}; "
                    "claim decomposition must be recorded on this ledger first."
                )

        self._ledger.probe_questions.extend(questions)
        self._log_event(
            PipelineStage.QUESTION_GENERATION,
            f"Recorded {len(questions)} probe question(s) across {len(known_claim_ids)} atomic claim(s)",
        )

    def record_retrieval_queries(self, queries: list[RetrievalQuery]) -> None:
        """Record the full set of retrieval queries built from this ledger's atomic claims and probe questions.

        Write-once: query-building is meant to run exactly once per ledger.
        Raises ValueError if queries have already been recorded, or if any
        query references an atomic_claim_id not present in this ledger.

        Known limitation: write-once is detected via list non-emptiness —
        see record_probe_questions for the same caveat (it cannot matter
        here in practice, since build_retrieval_queries always returns at
        least one CLAIM-origin query per atomic claim).
        """
        if self._ledger.retrieval_queries:
            raise ValueError(
                "Retrieval queries have already been recorded on this ledger; query-building is write-once."
            )

        known_claim_ids = {claim.claim_id for claim in self._ledger.atomic_claims}
        for query in queries:
            if query.atomic_claim_id not in known_claim_ids:
                raise ValueError(
                    f"RetrievalQuery references unknown atomic_claim_id {query.atomic_claim_id!r}; "
                    "claim decomposition must be recorded on this ledger first."
                )

        self._ledger.retrieval_queries.extend(queries)
        self._log_event(
            PipelineStage.RETRIEVAL,
            f"Recorded {len(queries)} retrieval quer(y/ies) across {len(known_claim_ids)} atomic claim(s)",
        )

    def record_retrieved_evidence(self, evidence_list: list[RetrievedEvidence]) -> None:
        """Record the full set of retrieved evidence produced by a RetrievalOrchestrator run.

        This is the ledger-boundary enforcement point for the linkage
        invariant: every record must already carry a non-None
        atomic_claim_id and query_id (stamped by RetrievalOrchestrator), and
        both must resolve against this ledger's recorded atomic_claims and
        retrieval_queries. Evidence constructed directly by a Retriever/
        Indexer (claim-agnostic, with these fields left None) must never
        reach this method without first passing through the orchestrator.

        Write-once: see record_probe_questions for the same non-emptiness caveat.
        """
        if self._ledger.retrieved_evidence:
            raise ValueError(
                "Retrieved evidence has already been recorded on this ledger; retrieval is write-once."
            )

        known_claim_ids = {claim.claim_id for claim in self._ledger.atomic_claims}
        known_query_ids = {query.query_id for query in self._ledger.retrieval_queries}
        for evidence in evidence_list:
            if evidence.atomic_claim_id is None or evidence.query_id is None:
                raise ValueError(
                    f"RetrievedEvidence {evidence.evidence_id!r} is missing atomic_claim_id or query_id; "
                    "it must be stamped by RetrievalOrchestrator before being recorded."
                )
            if evidence.atomic_claim_id not in known_claim_ids:
                raise ValueError(
                    f"RetrievedEvidence references unknown atomic_claim_id {evidence.atomic_claim_id!r}."
                )
            if evidence.query_id not in known_query_ids:
                raise ValueError(f"RetrievedEvidence references unknown query_id {evidence.query_id!r}.")

        self._ledger.retrieved_evidence.extend(evidence_list)
        self._log_event(
            PipelineStage.RETRIEVAL,
            f"Recorded {len(evidence_list)} retrieved evidence record(s) across {len(known_query_ids)} quer(y/ies)",
        )

    def add_retrieved_evidence(self, evidence: RetrievedEvidence) -> None:
        """Record a single piece of pre-fusion retrieved evidence."""
        self._ledger.retrieved_evidence.append(evidence)
        self._log_event(
            PipelineStage.RETRIEVAL,
            f"Added {evidence.source.value} evidence for chunk {evidence.chunk_id!r}",
        )

    def add_fused_evidence(self, evidence: FusedEvidence) -> None:
        """Record a single post-fusion evidence record."""
        self._ledger.fused_evidence.append(evidence)
        self._log_event(PipelineStage.FUSION, f"Added fused evidence for chunk {evidence.chunk_id!r}")

    def record_fused_evidence(self, fused_list: list[FusedEvidence]) -> None:
        """Record the full set of fused evidence produced by a fuse_evidence run.

        Ledger-boundary enforcement: every record's atomic_claim_id must
        resolve against this ledger's recorded atomic_claims, and every
        chunk_id must trace back to at least one already-recorded
        retrieved_evidence entry for that same claim — fusion must operate
        only on evidence already present in the ledger, never invent new
        chunks.

        Write-once: see record_probe_questions for the same non-emptiness caveat.
        """
        if self._ledger.fused_evidence:
            raise ValueError(
                "Fused evidence has already been recorded on this ledger; fusion is write-once."
            )

        known_claim_ids = {claim.claim_id for claim in self._ledger.atomic_claims}
        known_claim_chunk_pairs = {
            (record.atomic_claim_id, record.chunk_id) for record in self._ledger.retrieved_evidence
        }
        for fused in fused_list:
            if fused.atomic_claim_id not in known_claim_ids:
                raise ValueError(
                    f"FusedEvidence references unknown atomic_claim_id {fused.atomic_claim_id!r}."
                )
            if (fused.atomic_claim_id, fused.chunk_id) not in known_claim_chunk_pairs:
                raise ValueError(
                    f"FusedEvidence for claim {fused.atomic_claim_id!r} references chunk "
                    f"{fused.chunk_id!r} with no corresponding entry in retrieved_evidence."
                )

        self._ledger.fused_evidence.extend(fused_list)
        self._log_event(
            PipelineStage.FUSION,
            f"Recorded {len(fused_list)} fused evidence record(s) across {len(known_claim_ids)} atomic claim(s)",
        )

    def record_nli_results(self, results: list[NLIResult]) -> None:
        """Record the full set of NLI verification outcomes for this ledger.

        Ledger-boundary enforcement: every result's atomic_claim_id must
        resolve against this ledger's recorded atomic_claims, and the
        (atomic_claim_id, evidence_chunk_id) pair must trace back to an
        already-recorded fused_evidence entry — NLI must operate only on
        evidence already present in the ledger.

        Write-once: see record_probe_questions for the same non-emptiness caveat.
        """
        if self._ledger.nli_results:
            raise ValueError(
                "NLI results have already been recorded on this ledger; NLI verification is write-once."
            )

        known_claim_ids = {claim.claim_id for claim in self._ledger.atomic_claims}
        known_claim_chunk_pairs = {
            (record.atomic_claim_id, record.chunk_id) for record in self._ledger.fused_evidence
        }
        for result in results:
            if result.atomic_claim_id not in known_claim_ids:
                raise ValueError(f"NLIResult references unknown atomic_claim_id {result.atomic_claim_id!r}.")
            if (result.atomic_claim_id, result.evidence_chunk_id) not in known_claim_chunk_pairs:
                raise ValueError(
                    f"NLIResult for claim {result.atomic_claim_id!r} references chunk "
                    f"{result.evidence_chunk_id!r} with no corresponding entry in fused_evidence."
                )

        self._ledger.nli_results.extend(results)
        self._log_event(PipelineStage.NLI_VERIFICATION, f"Recorded {len(results)} NLI result(s)")

    def record_classification(
        self,
        contradictions: list[Contradiction],
        supporting_evidence: list[SupportingEvidence],
        unsupported_claim_ids: list[str],
    ) -> None:
        """Record the derived classification records produced by lncvs.rules.classification.classify().

        This is explainability bookkeeping, not verdict input:
        ThresholdRuleEngine independently recomputes the same classification
        from ledger.nli_results via the same pure classify() helper, so the
        recorded trace and the verdict are guaranteed to agree.

        Write-once: see record_probe_questions for the same non-emptiness caveat.
        """
        if self._ledger.contradictions or self._ledger.supporting_evidence or self._ledger.unsupported_claims:
            raise ValueError(
                "Classification has already been recorded on this ledger; classification is write-once."
            )

        known_claim_ids = {claim.claim_id for claim in self._ledger.atomic_claims}
        for record in (*contradictions, *supporting_evidence):
            if record.atomic_claim_id not in known_claim_ids:
                raise ValueError(
                    f"Classification record references unknown atomic_claim_id {record.atomic_claim_id!r}."
                )
        for claim_id in unsupported_claim_ids:
            if claim_id not in known_claim_ids:
                raise ValueError(f"Classification record references unknown atomic_claim_id {claim_id!r}.")

        self._ledger.contradictions.extend(contradictions)
        self._ledger.supporting_evidence.extend(supporting_evidence)
        self._ledger.unsupported_claims.extend(unsupported_claim_ids)
        self._log_event(
            PipelineStage.NLI_VERIFICATION,
            f"Recorded classification: {len(contradictions)} contradiction(s), "
            f"{len(supporting_evidence)} supporting, {len(unsupported_claim_ids)} unsupported",
        )

    def add_nli_result(self, result: NLIResult) -> None:
        """Record a single NLI verification outcome."""
        self._ledger.nli_results.append(result)
        self._log_event(
            PipelineStage.NLI_VERIFICATION,
            f"NLI result for claim {result.atomic_claim_id!r}: {result.label.value} ({result.score:.3f})",
        )

    def add_contradiction(self, contradiction: Contradiction) -> None:
        """Record a claim found to be contradicted by evidence."""
        self._ledger.contradictions.append(contradiction)
        self._log_event(
            PipelineStage.NLI_VERIFICATION,
            f"Recorded contradiction for claim {contradiction.atomic_claim_id!r}",
        )

    def add_supporting_evidence(self, evidence: SupportingEvidence) -> None:
        """Record a claim found to be entailed by evidence."""
        self._ledger.supporting_evidence.append(evidence)
        self._log_event(
            PipelineStage.NLI_VERIFICATION,
            f"Recorded supporting evidence for claim {evidence.atomic_claim_id!r}",
        )

    def mark_unsupported(self, atomic_claim_id: str) -> None:
        """Record that an atomic claim has neither entailing nor contradicting evidence."""
        if atomic_claim_id not in self._ledger.unsupported_claims:
            self._ledger.unsupported_claims.append(atomic_claim_id)
        self._log_event(PipelineStage.NLI_VERIFICATION, f"Marked claim {atomic_claim_id!r} as unsupported")

    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """Append a human-readable step to the reasoning trace."""
        self._ledger.reasoning_trace.append(step)

    def log_event(self, event: LedgerEvent) -> None:
        """Append a pre-built LedgerEvent directly to the audit log."""
        self._ledger.ledger_log.append(event)
        logger.debug("Ledger event [%s]: %s", event.stage.value, event.message)

    def set_final_verdict(self, verdict: FinalVerdict) -> None:
        """Set the ledger's final verdict.

        Raises ValueError if a verdict has already been set: verdicts are
        write-once, since a ledger is meant to be evaluated exactly one time.
        """
        if self._ledger.final_verdict is not None:
            raise ValueError(
                "Final verdict has already been set on this ledger; verdicts are write-once."
            )
        self._ledger.final_verdict = verdict
        self._log_event(PipelineStage.RULE_ENGINE, f"Final verdict set: {verdict.verdict.value}")

    def _log_event(self, stage: PipelineStage, message: str) -> None:
        event = LedgerEvent(
            event_id=str(uuid4()),
            stage=stage,
            message=message,
            timestamp=datetime.now(timezone.utc),
        )
        self.log_event(event)
