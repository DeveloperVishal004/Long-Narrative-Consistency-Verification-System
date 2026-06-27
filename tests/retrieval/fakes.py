"""Deterministic, offline test double for the Retriever protocol."""

from lncvs.schemas import Provenance, RetrievalSource, RetrievedEvidence


class FakeRetriever:
    """A scripted Retriever: returns a fixed, claim-agnostic result list for a
    given query text, regardless of which claim/query invoked it. Records
    every (query, top_k) call it received."""

    def __init__(self, responses: dict[str, list[RetrievedEvidence]] | None = None) -> None:
        self._responses = responses or {}
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedEvidence]:
        self.calls.append((query, top_k))
        if query not in self._responses:
            raise ValueError(f"FakeRetriever has no scripted response for query: {query!r}")
        return self._responses[query][:top_k]


def make_unstamped_evidence(
    chunk_id: str, text: str, rank: int = 1, source: RetrievalSource = RetrievalSource.SEMANTIC
) -> RetrievedEvidence:
    """Build a RetrievedEvidence exactly as a claim-agnostic Retriever would —
    evidence_id derived from raw query text, atomic_claim_id/query_id left None."""
    return RetrievedEvidence(
        evidence_id=f"unstamped-{source.value}-{chunk_id}-{rank}",
        chunk_id=chunk_id,
        text=text,
        source=source,
        raw_score=0.9,
        rank=rank,
        provenance=Provenance(chunk_id=chunk_id, char_start=0, char_end=len(text)),
    )
