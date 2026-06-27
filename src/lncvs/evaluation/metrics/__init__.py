"""Pure, read-only metric functions over (EvidenceLedger, gold) pairs."""

from lncvs.evaluation.metrics.citation import compute_citation_metrics
from lncvs.evaluation.metrics.latency import compute_latency_metrics
from lncvs.evaluation.metrics.retrieval import compute_retrieval_metrics
from lncvs.evaluation.metrics.verdict import compute_verdict_metrics

__all__ = [
    "compute_citation_metrics",
    "compute_latency_metrics",
    "compute_retrieval_metrics",
    "compute_verdict_metrics",
]
