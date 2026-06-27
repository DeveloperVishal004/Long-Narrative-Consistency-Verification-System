"""Citation-quality metrics: citation accuracy and hallucination rate.

Computed over the union of the ledger's contradictions and
supporting_evidence -- the only evidence the rule engine actually cited in
reaching its verdict.
"""

from lncvs.schemas import EvidenceLedger
from lncvs.schemas.evaluation import CitationMetrics


def compute_citation_metrics(ledger: EvidenceLedger, gold_chunk_ids: set[str]) -> CitationMetrics | None:
    """Returns None if nothing was cited -- not an error, and never reported as 0."""
    cited_chunk_ids = [c.evidence_chunk_id for c in ledger.contradictions] + [
        s.evidence_chunk_id for s in ledger.supporting_evidence
    ]
    if not cited_chunk_ids:
        return None

    grounded_count = sum(1 for chunk_id in cited_chunk_ids if chunk_id in gold_chunk_ids)
    cited_count = len(cited_chunk_ids)

    return CitationMetrics(
        citation_accuracy=grounded_count / cited_count,
        hallucination_rate=(cited_count - grounded_count) / cited_count,
        cited_count=cited_count,
        grounded_count=grounded_count,
    )
