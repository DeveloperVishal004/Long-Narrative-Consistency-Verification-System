"""GraphRetriever: pure delegation to an injected GraphIndex, Retriever
protocol conformance -- exact structural mirror of test_semantic_retriever.py."""

import pytest

from lncvs.graph import GraphRetriever
from lncvs.retrieval import Retriever
from lncvs.schemas import Provenance, RetrievalSource, RetrievedEvidence


class _StubGraphIndex:
    def __init__(self) -> None:
        self.last_query: tuple[str, int] | None = None

    def index(self, chunks: list) -> None:  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def query(self, query_text: str, top_k: int) -> list[RetrievedEvidence]:
        self.last_query = (query_text, top_k)
        return [
            RetrievedEvidence(
                evidence_id="ev-graph-1",
                chunk_id="chunk-london",
                text="John moved to London in 2012.",
                source=RetrievalSource.GRAPH,
                raw_score=1.5,
                rank=1,
                provenance=Provenance(chunk_id="chunk-london", char_start=0, char_end=30),
            )
        ]


def test_graph_retriever_satisfies_retriever_protocol() -> None:
    assert isinstance(GraphRetriever(_StubGraphIndex()), Retriever)


def test_graph_retriever_delegates_to_indexer() -> None:
    stub = _StubGraphIndex()
    retriever = GraphRetriever(stub)

    results = retriever.retrieve("Where did John move?", top_k=3)

    assert stub.last_query == ("Where did John move?", 3)
    assert len(results) == 1
    assert results[0].source is RetrievalSource.GRAPH


def test_graph_retriever_rejects_non_positive_top_k() -> None:
    retriever = GraphRetriever(_StubGraphIndex())
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=0)
