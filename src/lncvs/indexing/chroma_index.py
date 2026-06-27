"""ChromaIndex — the only module in this codebase that imports chromadb directly.

Embeddings are always computed by the injected Embedder and passed to Chroma
explicitly; Chroma's own default embedding function is never invoked, so
indexing never triggers a hidden model download beyond the one the injected
Embedder itself performs.
"""

import hashlib
import logging

from lncvs.indexing.embedder import Embedder
from lncvs.schemas import DocumentChunk, Provenance, RetrievalSource, RetrievedEvidence

logger = logging.getLogger(__name__)


class ChromaIndex:
    """Semantic index backed by an in-memory (ephemeral) ChromaDB collection.

    No persistence in Phase 1: the index must be rebuilt (via index()) every
    time a new ChromaIndex is constructed. This is a deliberate, documented
    limitation, not an oversight — see CLAUDE.md Phase 1 risk notes.
    """

    def __init__(self, embedder: Embedder, collection_name: str = "lncvs_chunks") -> None:
        import chromadb

        self._embedder = embedder
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, chunks: list[DocumentChunk]) -> None:
        """Embed and upsert chunks into the collection."""
        if not chunks:
            raise ValueError("Cannot index an empty list of chunks")

        texts = [chunk.text for chunk in chunks]
        vectors = self._embedder.embed_texts(texts)

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "source_id": chunk.source_id,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "chapter": chunk.chapter or "",
                }
                for chunk in chunks
            ],
        )
        logger.info("Indexed %d chunks into collection %r", len(chunks), self._collection.name)

    def query(self, query_text: str, top_k: int) -> list[RetrievedEvidence]:
        """Return the top_k most semantically similar chunks, ranked best-first."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        query_vector = self._embedder.embed_query(query_text)
        results = self._collection.query(query_embeddings=[query_vector], n_results=top_k)

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        evidence: list[RetrievedEvidence] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            provenance = Provenance(
                chunk_id=chunk_id,
                char_start=int(metadata["char_start"]),
                char_end=int(metadata["char_end"]),
            )
            evidence.append(
                RetrievedEvidence(
                    evidence_id=_make_evidence_id(query_text, chunk_id, rank),
                    chunk_id=chunk_id,
                    text=text,
                    source=RetrievalSource.SEMANTIC,
                    raw_score=max(0.0, 1.0 - distance),
                    rank=rank,
                    provenance=provenance,
                )
            )
        return evidence


def _make_evidence_id(query_text: str, chunk_id: str, rank: int) -> str:
    """Deterministic evidence_id: the same query against the same index always
    produces the same IDs, rather than a fresh uuid4() on every call."""
    digest_input = f"{query_text}:{chunk_id}:{rank}".encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()[:16]
