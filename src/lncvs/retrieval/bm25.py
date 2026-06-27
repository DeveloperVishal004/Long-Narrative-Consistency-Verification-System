"""BM25Retriever — a Retriever backed by an injected BM25Index (Indexer)."""

from lncvs.indexing.base import Indexer
from lncvs.schemas import RetrievedEvidence


class BM25Retriever:
    """Retriever that delegates entirely to an injected lexical Indexer (e.g. BM25Index).

    Exact structural mirror of SemanticRetriever — holds no state beyond
    the injected dependency, satisfies the unchanged Retriever protocol.
    """

    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer

    def retrieve(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return self._indexer.query(query, top_k)
