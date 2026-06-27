"""Pure Reciprocal Rank Fusion.

Imports schemas only — never lncvs.retrieval — so this module is testable
in complete isolation from the retrieval stack it consumes output from.
Grouping-by-claim logic is intentionally reimplemented here rather than
imported from lncvs.retrieval.grouping, to keep that independence real
(the same small-duplication-over-coupling tradeoff already used between
lncvs.reasoning.decomposition.identity and lncvs.reasoning.questions.identity).
"""

from collections import defaultdict

from lncvs.fusion.config import FusionConfig
from lncvs.schemas import FusedEvidence, RetrievedEvidence


def fuse_evidence(evidence: list[RetrievedEvidence], config: FusionConfig) -> list[FusedEvidence]:
    """Fuse claim-linked RetrievedEvidence into per-claim, deduplicated FusedEvidence.

    For each (atomic_claim_id, chunk_id) pair, rrf_score is the sum of
    1/(rrf_k + rank) over every (query, source) contribution that surfaced
    that chunk for that claim. Results are ranked best-first per claim,
    with ties broken deterministically by chunk_id, and capped at
    config.top_k_fused per claim.

    Raises ValueError if any evidence record's atomic_claim_id is None —
    such a record has not been stamped by RetrievalOrchestrator and must
    never reach fusion.
    """
    for record in evidence:
        if record.atomic_claim_id is None:
            raise ValueError(
                f"Cannot fuse evidence {record.evidence_id!r}: atomic_claim_id is not set. "
                "Evidence must be stamped by RetrievalOrchestrator before fusion."
            )

    groups: dict[tuple[str, str], list[RetrievedEvidence]] = defaultdict(list)
    for record in evidence:
        groups[(record.atomic_claim_id, record.chunk_id)].append(record)

    fused_by_claim: dict[str, list[FusedEvidence]] = defaultdict(list)
    for (claim_id, chunk_id), records in groups.items():
        rrf_score = sum(1.0 / (config.rrf_k + record.rank) for record in records)

        sources = []
        seen_sources = set()
        query_ids = []
        seen_query_ids = set()
        for record in records:
            if record.source not in seen_sources:
                seen_sources.add(record.source)
                sources.append(record.source)
            if record.query_id not in seen_query_ids:
                seen_query_ids.add(record.query_id)
                query_ids.append(record.query_id)

        fused_by_claim[claim_id].append(
            FusedEvidence(
                atomic_claim_id=claim_id,
                chunk_id=chunk_id,
                text=records[0].text,
                rrf_score=rrf_score,
                contributing_sources=sources,
                contributing_query_ids=query_ids,
            )
        )

    result: list[FusedEvidence] = []
    for claim_id in fused_by_claim:
        ranked = sorted(fused_by_claim[claim_id], key=lambda f: (-f.rrf_score, f.chunk_id))
        result.extend(ranked[: config.top_k_fused])

    return result
