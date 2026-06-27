"""Indexer protocol conformance tests."""

from lncvs.indexing import ChromaIndex, Indexer
from tests.indexing.fakes import FakeEmbedder


def test_chroma_index_satisfies_indexer_protocol() -> None:
    index = ChromaIndex(embedder=FakeEmbedder(), collection_name="contract-test")
    assert isinstance(index, Indexer)
