"""BM25Retriever tests: Retriever protocol conformance and delegation."""

import pytest

from lncvs.retrieval import BM25Retriever, Retriever


class _StubBM25Indexer:
    def __init__(self) -> None:
        self.last_query: tuple[str, int] | None = None

    def index(self, chunks: list) -> None:  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def query(self, query_text: str, top_k: int) -> list:
        self.last_query = (query_text, top_k)
        return []


def test_bm25_retriever_satisfies_retriever_protocol() -> None:
    assert isinstance(BM25Retriever(_StubBM25Indexer()), Retriever)


def test_bm25_retriever_delegates_to_indexer() -> None:
    stub = _StubBM25Indexer()
    retriever = BM25Retriever(stub)

    retriever.retrieve("Did John lose an arm?", top_k=3)

    assert stub.last_query == ("Did John lose an arm?", 3)


def test_bm25_retriever_rejects_non_positive_top_k() -> None:
    retriever = BM25Retriever(_StubBM25Indexer())
    with pytest.raises(ValueError):
        retriever.retrieve("query", top_k=0)
