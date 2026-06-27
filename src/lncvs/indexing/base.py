"""Indexer protocol — the dependency-injection boundary between retrieval and any vector store."""

from typing import Protocol, runtime_checkable

from lncvs.schemas import DocumentChunk, RetrievedEvidence


@runtime_checkable
class Indexer(Protocol):
    """Contract for building and querying a semantic index.

    ChromaIndex implements this protocol; retrieval/ depends only on this
    interface, never on chromadb directly. A future backend could implement
    this same protocol without retrieval/ changing at all.
    """

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Embed and add chunks to the index."""
        ...

    def query(self, query_text: str, top_k: int) -> list[RetrievedEvidence]:
        """Return the top_k most semantically similar chunks to query_text, ranked best-first."""
        ...
