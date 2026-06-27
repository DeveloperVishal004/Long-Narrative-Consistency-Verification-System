"""GraphRetriever -- a Retriever backed by an injected GraphIndex (Indexer).

Exact structural mirror of SemanticRetriever and BM25Retriever: holds no
state beyond the injected dependency, satisfies the unchanged Retriever
protocol unmodified. This is what makes graph retrieval "another
RetrievalSource" rather than a new pipeline stage -- RetrievalOrchestrator
takes a list[Retriever] and cannot tell GraphRetriever apart from the
others except via the source field already on each RetrievedEvidence.
"""

from lncvs.graph.index import GraphIndex
from lncvs.schemas import RetrievedEvidence


class GraphRetriever:
    def __init__(self, indexer: GraphIndex) -> None:
        self._indexer = indexer

    def retrieve(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return self._indexer.query(query, top_k)
