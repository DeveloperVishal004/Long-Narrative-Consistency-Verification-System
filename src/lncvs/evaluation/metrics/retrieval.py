"""Retrieval and fusion quality metrics: Recall@k, Precision@k, MRR.

Computed over the ledger's fused_evidence (already deduplicated by
chunk_id, ranked by rrf_score), pooled across all atomic claims for a
single example, against that example's gold-derived relevant chunk_ids.
"""

from lncvs.schemas import EvidenceLedger
from lncvs.schemas.evaluation import RankCutoffMetric, RetrievalMetrics


def compute_retrieval_metrics(
    ledger: EvidenceLedger, gold_chunk_ids: set[str], k_cutoffs: list[int]
) -> RetrievalMetrics | None:
    """Returns None if gold_chunk_ids is empty -- a metric must never be
    silently reported as 0 when its required gold input is absent."""
    if not gold_chunk_ids:
        return None

    ranked_chunk_ids: list[str] = []
    seen: set[str] = set()
    for fused in sorted(ledger.fused_evidence, key=lambda f: (-f.rrf_score, f.chunk_id)):
        if fused.chunk_id not in seen:
            seen.add(fused.chunk_id)
            ranked_chunk_ids.append(fused.chunk_id)

    mrr = 0.0
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in gold_chunk_ids:
            mrr = 1.0 / rank
            break

    cutoffs: list[RankCutoffMetric] = []
    for k in sorted(k_cutoffs):
        top_k = ranked_chunk_ids[:k]
        hits = sum(1 for chunk_id in top_k if chunk_id in gold_chunk_ids)
        recall = hits / len(gold_chunk_ids)
        precision = hits / len(top_k) if top_k else 0.0
        cutoffs.append(RankCutoffMetric(k=k, recall=recall, precision=precision))

    return RetrievalMetrics(mrr=mrr, cutoffs=cutoffs)
