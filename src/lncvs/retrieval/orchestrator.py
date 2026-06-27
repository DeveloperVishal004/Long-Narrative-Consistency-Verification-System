"""RetrievalOrchestrator — stamps claim/query provenance onto otherwise claim-agnostic retrieval results.

The injected Retrievers (and the Indexers they wrap) know nothing about
claims or queries — they only answer "given this text, which chunks?".
This orchestrator is the sole layer that knows about RetrievalQuery and is
responsible for linking each RetrievedEvidence record back to the claim,
query, and source that produced it, including re-deriving evidence_id so
that two claims sharing identical query text never collide, AND so that
two different retrieval backends returning the same chunk at the same rank
for the same query never collide either (see
lncvs.retrieval.identity.make_evidence_id).
"""

from lncvs.retrieval.base import Retriever
from lncvs.retrieval.config import RetrievalConfig
from lncvs.retrieval.identity import make_evidence_id
from lncvs.schemas import RetrievalQuery, RetrievedEvidence


class RetrievalOrchestrator:
    """Runs a set of RetrievalQuery objects through every injected Retriever and
    stamps every resulting RetrievedEvidence with claim/query/source provenance.

    Retrievers run in the fixed order given at construction, so output
    ordering is deterministic. A single-retriever list (e.g. semantic-only)
    behaves exactly as Phase 3's single-retriever orchestrator did.

    Holds no state beyond its injected dependencies.
    """

    def __init__(self, retrievers: list[Retriever], config: RetrievalConfig) -> None:
        self._retrievers = retrievers
        self._config = config

    def retrieve_for_queries(self, queries: list[RetrievalQuery]) -> list[RetrievedEvidence]:
        """Retrieve evidence for every query across every retriever, with full
        provenance stamped on each result.

        A query that retrieves zero evidence from a given retriever is not
        an error — it is the correct signal for a claim with no supporting
        or contradicting evidence from that source, which the rule engine
        later resolves to INSUFFICIENT_EVIDENCE if true across all sources.
        """
        all_evidence: list[RetrievedEvidence] = []

        for query in queries:
            for retriever in self._retrievers:
                results = retriever.retrieve(query.text, self._config.top_k)
                for result in results:
                    stamped = result.model_copy(
                        update={
                            "evidence_id": make_evidence_id(
                                query.query_id, result.source.value, result.chunk_id, result.rank
                            ),
                            "atomic_claim_id": query.atomic_claim_id,
                            "query_id": query.query_id,
                        }
                    )
                    all_evidence.append(stamped)

        return all_evidence
