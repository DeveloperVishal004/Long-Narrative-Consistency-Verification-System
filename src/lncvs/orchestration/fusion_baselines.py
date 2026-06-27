"""Evaluation-flavored fusion baseline: round-robin (best-rank) interleaving.

Moved here from evaluation/ in Phase 7: both PipelineRunner (evaluation/runner.py)
and the LangGraph fuse node (orchestration/nodes.py) must call the IDENTICAL
function for the equivalence guarantee between the two runners to hold, and
orchestration/ must never import from evaluation/ (the canonical dependency
chain is `... -> rules -> orchestration -> evaluation`). evaluation/ now
imports this from here instead of defining its own copy --
evaluation/fusion_baselines.py is a re-export shim for backward compatibility.

Never used by production fusion: lncvs.fusion remains RRF-only. This exists
solely so the RRF ablation has something concrete to compare against -- a
boolean "RRF off" is ill-defined once two retrieval sources exist, since
*some* method is still needed to rank the combined candidate set.

Structurally mirrors lncvs.fusion.rrf.fuse_evidence (same grouping and
per-claim ranking shape), differing only in the score formula: rrf_score is
repurposed here as a generic ranking score, 1/(1 + best_rank), never an
actual RRF score.
"""

from collections import defaultdict

from lncvs.schemas import FusedEvidence, RetrievedEvidence


def round_robin_fuse(evidence: list[RetrievedEvidence], top_k_fused: int) -> list[FusedEvidence]:
    """Fuse claim-linked RetrievedEvidence by best (lowest) rank across all
    (query, source) contributions, deduplicated by (atomic_claim_id, chunk_id).

    Raises ValueError if any evidence record's atomic_claim_id is None --
    such a record has not been stamped by RetrievalOrchestrator.
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
        best_rank = min(record.rank for record in records)

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
                rrf_score=1.0 / (1 + best_rank),
                contributing_sources=sources,
                contributing_query_ids=query_ids,
            )
        )

    result: list[FusedEvidence] = []
    for claim_id in fused_by_claim:
        ranked = sorted(fused_by_claim[claim_id], key=lambda f: (-f.rrf_score, f.chunk_id))
        result.extend(ranked[:top_k_fused])

    return result
