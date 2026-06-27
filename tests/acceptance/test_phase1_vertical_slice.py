"""Phase 1 acceptance test: the full vertical slice against the dummy narrative.

Narrative -> Load -> Clean -> Chunk -> Embed -> Index -> Retrieve

Unlike the rest of the Phase 1 suite, this test uses the real
SentenceTransformerEmbedder (not the FakeEmbedder) because it is the only
way to prove genuinely *semantic* retrieval: the query below shares no
distinctive words with the target sentence, so a lexical match (and the
hash-based FakeEmbedder) would not reliably succeed here. If the model
cannot be loaded in this environment (no network access), the test skips
with a clear reason rather than failing or hanging.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig, chunk_document
from lncvs.indexing import ChromaIndex, EmbeddingConfig, SentenceTransformerEmbedder
from lncvs.ingestion import load_and_clean_narrative
from lncvs.retrieval import SemanticRetriever

SAMPLE_NARRATIVE = Path(__file__).resolve().parents[2] / "data" / "sample_narrative" / "john_test.txt"


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return SentenceTransformerEmbedder(config)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load embedding model in this environment: {exc}")


def test_vertical_slice_retrieves_arm_chunk_for_a_semantically_related_query(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    document = load_and_clean_narrative(SAMPLE_NARRATIVE, source_id="john_test")
    chunks = chunk_document(document, ChunkingConfig(chunk_size=60, overlap=10))
    assert len(chunks) >= 2, "expected the narrative to split into more than one chunk"

    index = ChromaIndex(embedder=real_embedder, collection_name="phase1-acceptance")
    index.index(chunks)

    retriever = SemanticRetriever(index)
    results = retriever.retrieve("Does John have any physical disability?", top_k=1)

    assert len(results) == 1
    assert "lost his left arm" in results[0].text


def test_vertical_slice_ranks_arm_chunk_above_london_chunk_for_injury_query(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    document = load_and_clean_narrative(SAMPLE_NARRATIVE, source_id="john_test")
    chunks = chunk_document(document, ChunkingConfig(chunk_size=60, overlap=10))

    index = ChromaIndex(embedder=real_embedder, collection_name="phase1-acceptance-ranked")
    index.index(chunks)

    retriever = SemanticRetriever(index)
    results = retriever.retrieve("Did John suffer an injury?", top_k=len(chunks))

    top_chunk_texts = [evidence.text for evidence in results]
    assert any("lost his left arm" in text for text in top_chunk_texts)
    assert "lost his left arm" in results[0].text
