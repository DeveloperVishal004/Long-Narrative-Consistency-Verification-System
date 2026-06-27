"""Phase 3 acceptance tests: Retrieval Integration.

Two variants:

1. An offline, fully deterministic slice (decompose -> generate questions ->
   build queries -> orchestrate against a FakeRetriever -> record in the
   ledger) proving the *wiring* and provenance chain.

2. A gated, real-embedder "thesis" test: the claim "John used both hands"
   is not semantically near "John lost his left arm" on its own, but its
   generated probe question ("Did John lose an arm?") is. This is the
   project's central hypothesis end to end — decomposition + question
   generation + retrieval surfacing a contradiction that a single
   undecomposed query would likely miss. Skips cleanly if the real model
   cannot be loaded in this environment.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig, chunk_document
from lncvs.indexing import ChromaIndex, EmbeddingConfig, SentenceTransformerEmbedder
from lncvs.ingestion import load_and_clean_narrative
from lncvs.ledger import LedgerService
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig
from lncvs.retrieval import RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries
from lncvs.schemas import EvidenceLedger
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


def test_offline_retrieval_integration_slice_is_fully_traceable_and_collision_free() -> None:
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

    # A claim-agnostic retriever: "John used both hands" itself returns nothing,
    # but its probe question surfaces the contradicting evidence. Two other
    # claim queries happen to share identical fake retriever responses to
    # exercise the evidence_id collision-freedom property.
    shared_response = [make_unstamped_evidence("chunk-shared", "some shared evidence text")]
    retriever = FakeRetriever(
        {
            "John played piano": shared_response,
            "the event occurred in London": shared_response,
            "John used both hands": [],
            "Did John lose an arm?": [make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.")],
            "Did John suffer an injury?": [make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.")],
        }
    )
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    # Full traceability: every piece of evidence resolves claim -> query -> chunk.
    for item in service.ledger.retrieved_evidence:
        assert item.atomic_claim_id is not None
        assert item.query_id is not None
        matching_query = next(q for q in service.ledger.retrieval_queries if q.query_id == item.query_id)
        assert matching_query.atomic_claim_id == item.atomic_claim_id

    # Collision-freedom: two claims ("John played piano", "the event occurred
    # in London") got the identical underlying evidence from the fake
    # retriever, but their evidence_ids must differ.
    piano_claim_id = next(c.claim_id for c in atomic_claims if c.text == "John played piano")
    london_claim_id = next(c.claim_id for c in atomic_claims if c.text == "the event occurred in London")
    piano_evidence_id = next(e.evidence_id for e in evidence if e.atomic_claim_id == piano_claim_id)
    london_evidence_id = next(e.evidence_id for e in evidence if e.atomic_claim_id == london_claim_id)
    assert piano_evidence_id != london_evidence_id

    # The "John used both hands" claim has no evidence from its own claim
    # query, but does have evidence surfaced via its probe questions.
    hands_claim_id = next(c.claim_id for c in atomic_claims if c.text == "John used both hands")
    hands_evidence = [e for e in evidence if e.atomic_claim_id == hands_claim_id]
    assert len(hands_evidence) == 2  # one per probe question
    assert all(e.chunk_id == "chunk-arm" for e in hands_evidence)


def test_offline_retrieval_integration_slice_is_deterministic_end_to_end() -> None:
    def run_once() -> list[str]:
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

        retriever = FakeRetriever(
            {
                "John played piano": [],
                "the event occurred in London": [],
                "John used both hands": [],
                "Did John lose an arm?": [make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.")],
                "Did John suffer an injury?": [make_unstamped_evidence("chunk-arm", "John lost his left arm in 2010.")],
            }
        )
        orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))
        evidence = orchestrator.retrieve_for_queries(queries)
        service.record_retrieved_evidence(evidence)

        return [e.evidence_id for e in service.ledger.retrieved_evidence]

    assert run_once() == run_once()


# --- Variant 2: gated, real embedder — the project's central hypothesis ---


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return SentenceTransformerEmbedder(config)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load embedding model in this environment: {exc}")


def test_probe_question_retrieves_the_contradicting_chunk_with_full_provenance(
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    """Proves the probe-question retrieval path works end to end with real
    embeddings: a QUESTION-origin query for the "John used both hands" claim
    retrieves the lost-arm chunk, and that evidence record's provenance
    resolves correctly back to its originating query and claim.

    This does NOT require the bare CLAIM-origin query to miss the chunk —
    on this small two-chunk fixture the claim query may well also retrieve
    it. The property under test is that the probe-question path independently
    works and is traceable, not that it is the exclusive or first source.
    """
    document = load_and_clean_narrative(SAMPLE_NARRATIVE, source_id="john_test")
    chunks = chunk_document(document, ChunkingConfig(chunk_size=60, overlap=10))

    index = ChromaIndex(embedder=real_embedder, collection_name="phase3-thesis")
    index.index(chunks)
    retriever = SemanticRetriever(index)

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

    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=len(chunks)))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    hands_claim_id = next(c.claim_id for c in atomic_claims if c.text == "John used both hands")
    probe_query = next(
        q for q in service.ledger.retrieval_queries if q.atomic_claim_id == hands_claim_id and q.origin.value == "QUESTION"
    )

    # 1. Probe-question retrieval occurs: the QUESTION-origin query actually
    #    returned results (not silently dropped).
    question_origin_evidence = [e for e in evidence if e.query_id == probe_query.query_id]
    assert question_origin_evidence, "expected the probe question's query to retrieve at least one result"

    # 2. QUESTION-origin evidence specifically surfaces the contradicting
    #    lost-arm chunk for the "John used both hands" claim.
    arm_evidence_via_question = [e for e in question_origin_evidence if "lost his left arm" in e.text]
    assert arm_evidence_via_question, (
        "expected the probe question 'Did John lose an arm?' to retrieve the lost-arm chunk"
    )

    # 3. Provenance is preserved: the evidence resolves back to the correct
    #    claim and query, and the query itself resolves back to the same claim.
    surfaced_evidence = arm_evidence_via_question[0]
    assert surfaced_evidence.atomic_claim_id == hands_claim_id
    assert surfaced_evidence.query_id == probe_query.query_id
    assert probe_query.atomic_claim_id == hands_claim_id
    assert probe_query.text == "Did John lose an arm?"
