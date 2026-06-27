"""Retriever protocol — the shared contract every retrieval backend conforms to."""

from typing import Protocol, runtime_checkable

from lncvs.schemas import RetrievedEvidence


@runtime_checkable
class Retriever(Protocol):
    """Contract for retrieving evidence for a query.

    Phase 1 has a single implementation (SemanticRetriever). Future
    backends (e.g. lexical, in Phase 3) implement the same protocol so the
    hybrid orchestrator added then can treat them uniformly.
    """

    def retrieve(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        """Return the top_k most relevant evidence records for query, ranked best-first."""
        ...
