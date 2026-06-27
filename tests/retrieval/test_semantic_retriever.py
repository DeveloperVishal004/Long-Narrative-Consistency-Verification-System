"""SemanticRetriever tests: delegation to an injected Indexer, contract conformance."""

import pytest

from lncvs.retrieval import Retriever, SemanticRetriever
from lncvs.schemas import Provenance, RetrievalSource, RetrievedEvidence


class _StubIndexer:
    """A minimal Indexer test double that records calls and returns canned evidence."""

    def __init__(self) -> None:
        self.last_query: tuple[str, int] | None = None

    def index(self, chunks: list) -> None:  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def query(self, query_text: str, top_k: int) -> list[RetrievedEvidence]:
        self.last_query = (query_text, top_k)
        return [
            RetrievedEvidence(
                evidence_id="ev-1",
                chunk_id="chunk-arm",
                text="John lost his left arm in an accident in 2010.",
                source=RetrievalSource.SEMANTIC,
                raw_score=0.95,
                rank=1,
                provenance=Provenance(chunk_id="chunk-arm", char_start=0, char_end=48),
            )
        ]


def test_semantic_retriever_satisfies_retriever_protocol() -> None:
    assert isinstance(SemanticRetriever(_StubIndexer()), Retriever)


def test_semantic_retriever_delegates_to_indexer() -> None:
    stub = _StubIndexer()
    retriever = SemanticRetriever(stub)

    results = retriever.retrieve("What happened to John's arm?", top_k=3)

    assert stub.last_query == ("What happened to John's arm?", 3)
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-arm"


def test_semantic_retriever_rejects_non_positive_top_k() -> None:
    retriever = SemanticRetriever(_StubIndexer())
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=0)
