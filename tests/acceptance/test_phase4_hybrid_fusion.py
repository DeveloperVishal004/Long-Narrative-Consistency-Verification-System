"""Phase 4 acceptance tests: Hybrid Retrieval (BM25 + Dense) and Reciprocal Rank Fusion.

Two variants:

1. An offline, fully deterministic slice (two FakeRetrievers standing in for
   semantic and lexical sources -> orchestrate -> fuse -> record) proving
   the fusion wiring, cross-source evidence_id collision-freedom, and full
   claim/query provenance on FusedEvidence.

2. A gated, real-index "hybrid" test: BM25Index + ChromaIndex both built
   over the dummy narrative, fused via RRF. Skips cleanly if the real
   embedding model cannot be loaded in this environment.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig, chunk_document
from lncvs.fusion import FusionConfig, fuse_evidence
from lncvs.indexing import BM25Index, ChromaIndex, EmbeddingConfig, SentenceTransformerEmbedder
from lncvs.ingestion import load_and_clean_narrative
from lncvs.ledger import LedgerService
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries
from lncvs.schemas import EvidenceLedger, RetrievalSource
from tests.llm.fakes import FakeLLMClient
from tests.retrieval.fakes import FakeRetriever, make_unstamped_evidence

SAMPLE_NARRATIVE = Path(__file__).resolve().parents[2] / "data" / "sample_narrative" / "john_test.txt"

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'
QUESTIONS_BY_CLAIM_TEXT = {
    "John played piano": "[]",
    "John used both hands": '["Did John lose an arm?", "Did John suffer an injury?"]',
    "the event occurred in London": "[]",
}


# --- Variant 1: offline, fully deterministic ---


def _build_offline_ledger_through_retrieval() -> tuple[EvidenceLedger, LedgerService, list]:
    ledger = EvidenceLedger(original_claim=ORIGINAL_CLAIM)
    service = LedgerService(ledger)

    decomp_config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DECOMPOSITION_RESPONSE), decomp_config)
    parent_id = make_source_claim_id(ORIGINAL_CLAIM)
    atomic_claims = decomposer.decompose(ORIGINAL_CLAIM)
    service.record_atomic_claims(parent_id, atomic_claims)

    question_config = QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model"))
    all_questions = []
    for claim in atomic_claims:
        scripted_response = QUESTIONS_BY_CLAIM_TEXT[claim.text]
        generator = LLMQuestionGenerator(FakeLLMClient(default_response=scripted_response), question_config)
        all_questions.extend(generator.generate(claim))
    service.record_probe_questions(all_questions)

    queries = build_retrieval_queries(atomic_claims, all_questions)
    service.record_retrieval_queries(queries)

    semantic_retriever = FakeRetriever(
        {
            "John played piano": [],
            "the event occurred in London": [],
            "John used both hands": [],
            "Did John lose an arm?": [
                make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.", source=RetrievalSource.SEMANTIC)
            ],
            "Did John suffer an injury?": [],
        }
    )
    lexical_retriever = FakeRetriever(
        {
            "John played piano": [],
            "the event occurred in London": [],
            "John used both hands": [],
            "Did John lose an arm?": [
                make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.", source=RetrievalSource.LEXICAL)
            ],
            "Did John suffer an injury?": [
                make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.", source=RetrievalSource.LEXICAL)
            ],
        }
    )
    orchestrator = RetrievalOrchestrator([semantic_retriever, lexical_retriever], RetrievalConfig(top_k=5))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    return ledger, service, atomic_claims


def test_offline_hybrid_fusion_produces_claim_linked_fused_evidence_with_full_provenance() -> None:
    ledger, service, atomic_claims = _build_offline_ledger_through_retrieval()

    fusion_config = FusionConfig(rrf_k=60, top_k_fused=5)
    fused = fuse_evidence(service.ledger.retrieved_evidence, fusion_config)
    service.record_fused_evidence(fused)

    hands_claim_id = next(c.claim_id for c in atomic_claims if c.text == "John used both hands")
    hands_fused = [f for f in service.ledger.fused_evidence if f.atomic_claim_id == hands_claim_id]

    assert len(hands_fused) == 1
    assert hands_fused[0].chunk_id == "chunk-arm"
    # The arm chunk was surfaced by both SEMANTIC (via "Did John lose an arm?")
    # and LEXICAL (via both probe questions) -- three contributing evidence
    # records collapse into one fused record with both sources represented.
    assert set(hands_fused[0].contributing_sources) == {RetrievalSource.SEMANTIC, RetrievalSource.LEXICAL}
    assert len(hands_fused[0].contributing_query_ids) == 2  # two distinct probe questions contributed

    # Full provenance: fused -> claim, and fused.contributing_query_ids -> recorded queries.
    for query_id in hands_fused[0].contributing_query_ids:
        matching_query = next(q for q in service.ledger.retrieval_queries if q.query_id == query_id)
        assert matching_query.atomic_claim_id == hands_claim_id


def test_offline_hybrid_fusion_is_deterministic_end_to_end() -> None:
    def run_once() -> list[tuple[str, float]]:
        _, service, _ = _build_offline_ledger_through_retrieval()
        fusion_config = FusionConfig(rrf_k=60, top_k_fused=5)
        fused = fuse_evidence(service.ledger.retrieved_evidence, fusion_config)
        service.record_fused_evidence(fused)
        return [(f.chunk_id, f.rrf_score) for f in service.ledger.fused_evidence]

    assert run_once() == run_once()


def test_evidence_ids_remain_collision_free_across_two_sources_before_fusion() -> None:
    """Sanity check that the Phase 4 orchestrator fix (source folded into
    evidence_id) holds even in this two-source acceptance scenario."""
    _, service, _ = _build_offline_ledger_through_retrieval()

    evidence_ids = [e.evidence_id for e in service.ledger.retrieved_evidence]
    assert len(evidence_ids) == len(set(evidence_ids))


# --- Variant 2: gated, real BM25 + Chroma hybrid index ---


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return SentenceTransformerEmbedder(config)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load embedding model in this environment: {exc}")


def test_hybrid_retrieval_and_fusion_surfaces_the_contradiction(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    document = load_and_clean_narrative(SAMPLE_NARRATIVE, source_id="john_test")
    chunks = chunk_document(document, ChunkingConfig(chunk_size=60, overlap=10))

    chroma_index = ChromaIndex(embedder=real_embedder, collection_name="phase4-hybrid")
    chroma_index.index(chunks)
    bm25_index = BM25Index(collection_name="phase4-hybrid-bm25")
    bm25_index.index(chunks)

    semantic_retriever = SemanticRetriever(chroma_index)
    lexical_retriever = BM25Retriever(bm25_index)

    ledger = EvidenceLedger(original_claim=ORIGINAL_CLAIM)
    service = LedgerService(ledger)

    decomp_config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
    decomposer = LLMClaimDecomposer(FakeLLMClient(default_response=DECOMPOSITION_RESPONSE), decomp_config)
    parent_id = make_source_claim_id(ORIGINAL_CLAIM)
    atomic_claims = decomposer.decompose(ORIGINAL_CLAIM)
    service.record_atomic_claims(parent_id, atomic_claims)

    question_config = QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model"))
    all_questions = []
    for claim in atomic_claims:
        scripted_response = QUESTIONS_BY_CLAIM_TEXT[claim.text]
        generator = LLMQuestionGenerator(FakeLLMClient(default_response=scripted_response), question_config)
        all_questions.extend(generator.generate(claim))
    service.record_probe_questions(all_questions)

    queries = build_retrieval_queries(atomic_claims, all_questions)
    service.record_retrieval_queries(queries)

    orchestrator = RetrievalOrchestrator([semantic_retriever, lexical_retriever], RetrievalConfig(top_k=len(chunks)))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    fusion_config = FusionConfig()
    fused = fuse_evidence(service.ledger.retrieved_evidence, fusion_config)
    service.record_fused_evidence(fused)

    hands_claim_id = next(c.claim_id for c in atomic_claims if c.text == "John used both hands")
    hands_fused = [f for f in service.ledger.fused_evidence if f.atomic_claim_id == hands_claim_id]

    arm_fused = [f for f in hands_fused if "lost his left arm" in f.text]
    assert arm_fused, "expected hybrid retrieval+fusion to surface the lost-arm chunk for the 'used both hands' claim"
    assert arm_fused[0].rrf_score > 0
    assert len(arm_fused[0].contributing_sources) >= 1
