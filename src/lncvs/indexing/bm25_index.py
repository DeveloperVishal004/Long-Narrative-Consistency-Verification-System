"""BM25Index — the only module in this codebase that imports rank_bm25.

Mirrors ChromaIndex's shape exactly: implements the Indexer protocol,
holds its corpus in memory (no persistence in Phase 1, same documented
limitation as ChromaIndex), and shares the same chunk-ID space as Chroma
by construction — both index identical DocumentChunk lists with
deterministic content-hash chunk_ids.
"""

import hashlib
import logging

from lncvs.indexing.tokenizer import tokenize
from lncvs.schemas import DocumentChunk, Provenance, RetrievalSource, RetrievedEvidence

logger = logging.getLogger(__name__)


class BM25Index:
    """Lexical index backed by an in-memory BM25Okapi model.

    Uses the single shared tokenize() function for both corpus and query
    text, so corpus/query tokenization can never silently diverge.
    """

    def __init__(self, collection_name: str = "lncvs_chunks_bm25") -> None:
        self._collection_name = collection_name
        self._chunks: list[DocumentChunk] = []
        self._bm25 = None

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Tokenize and build the BM25 model over chunks."""
        if not chunks:
            raise ValueError("Cannot index an empty list of chunks")

        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        tokenized_corpus = [tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Indexed %d chunks into BM25 collection %r", len(self._chunks), self._collection_name)

    def query(self, query_text: str, top_k: int) -> list[RetrievedEvidence]:
        """Return the top_k highest-BM25-score chunks, ranked best-first.

        Ties (including an all-zero-score query, e.g. a query with no
        tokens in the corpus vocabulary) are broken deterministically by
        chunk_id, so result ordering never depends on iteration order.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self._bm25 is None:
            raise ValueError("Index has not been built yet; call index() first.")

        tokenized_query = tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda i: (-scores[i], self._chunks[i].chunk_id),
        )

        evidence: list[RetrievedEvidence] = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            chunk = self._chunks[idx]
            provenance = Provenance(
                chunk_id=chunk.chunk_id, char_start=chunk.char_start, char_end=chunk.char_end
            )
            evidence.append(
                RetrievedEvidence(
                    evidence_id=_make_raw_evidence_id(query_text, chunk.chunk_id, rank),
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=RetrievalSource.LEXICAL,
                    raw_score=float(scores[idx]),
                    rank=rank,
                    provenance=provenance,
                )
            )
        return evidence


def _make_raw_evidence_id(query_text: str, chunk_id: str, rank: int) -> str:
    """Claim-agnostic raw evidence_id, re-derived by RetrievalOrchestrator
    once claim/query/source provenance is known (see retrieval.identity.make_evidence_id).
    Mirrors ChromaIndex's own private helper of the same shape."""
    digest_input = f"{query_text}:{chunk_id}:{rank}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
