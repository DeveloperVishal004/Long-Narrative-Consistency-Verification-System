"""SemanticRetriever — a Retriever backed by an injected Indexer."""

from lncvs.indexing.base import Indexer
from lncvs.schemas import RetrievedEvidence


class SemanticRetriever:
    """Retriever that delegates entirely to an injected Indexer (e.g. ChromaIndex).

    Holds no state beyond the injected dependency itself.
    """

    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer

    def retrieve(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return self._indexer.query(query, top_k)
