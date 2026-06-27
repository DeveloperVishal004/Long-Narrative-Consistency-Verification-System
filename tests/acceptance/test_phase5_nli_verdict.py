"""Phase 5 acceptance tests: NLI Verification and Verdict Construction.

Two variants:

1. An offline, fully deterministic slice (FakeNLIModel standing in for the
   real cross-encoder) proving the full wiring -- fused evidence -> NLI ->
   classification -> ThresholdRuleEngine -- and that the PROJECT_SPEC.md
   Section 14 dummy case resolves to CONTRADICTORY.

2. A gated, real end-to-end pipeline test: ingestion through hybrid
   retrieval and fusion (Phase 1-4, real models) into NLI verification and
   verdict construction (Phase 5, real cross-encoder). Skips cleanly if
   either real model cannot be loaded in this environment.

Wiring is deliberately thin and local to this test, per the Phase 5
architecture review: no new orchestration module is introduced here.
LangGraph orchestration remains Phase 6 scope.
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
from lncvs.reasoning.nli import CrossEncoderNLIModel, CrossEncoderNLIVerifier, NLIConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig
from lncvs.retrieval import BM25Retriever, RetrievalConfig, RetrievalOrchestrator, SemanticRetriever, build_retrieval_queries
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine, classify
from lncvs.schemas import EvidenceLedger, NLILabel, VerdictEnum
from tests.llm.fakes import FakeLLMClient
from tests.reasoning.nli.fakes import FakeNLIModel
from tests.retrieval.fakes import FakeRetriever, make_unstamped_evidence

SAMPLE_NARRATIVE = Path(__file__).resolve().parents[2] / "data" / "sample_narrative" / "john_test.txt"

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'
QUESTIONS_BY_CLAIM_TEXT = {
    "John played piano": "[]",
    "John used both hands": '["Did John lose an arm?", "Did John suffer an injury?"]',
    "the event occurred in London": "[]",
}


def _run_nli_and_verdict(service: LedgerService, nli_model, rule_config: RuleEngineConfig) -> None:
    """Thin Phase 5 wiring: fused_evidence -> NLI -> classification -> verdict.

    Not a reusable orchestration module by design (see module docstring).
    """
    verifier = CrossEncoderNLIVerifier(nli_model)
    ledger = service.ledger

    fused_by_claim: dict[str, list] = {}
    for fused in ledger.fused_evidence:
        fused_by_claim.setdefault(fused.atomic_claim_id, []).append(fused)

    all_results = []
    for claim in ledger.atomic_claims:
        claim_evidence = fused_by_claim.get(claim.claim_id, [])
        all_results.extend(verifier.verify(claim, claim_evidence))
    service.record_nli_results(all_results)

    claim_ids = [claim.claim_id for claim in ledger.atomic_claims]
    outcome = classify(ledger.nli_results, claim_ids, rule_config)
    service.record_classification(outcome.contradictions, outcome.supporting_evidence, outcome.unsupported_claim_ids)

    engine = ThresholdRuleEngine(rule_config)
    verdict = engine.evaluate(ledger)
    service.set_final_verdict(verdict)


# --- Variant 1: offline, fully deterministic ---


def test_offline_dummy_case_resolves_to_contradictory() -> None:
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
            "the event occurred in London": [make_unstamped_evidence("chunk-london", "John moved to London in 2012.")],
            "John used both hands": [],
            "Did John lose an arm?": [
                make_unstamped_evidence("chunk-arm", "John lost his left arm in an accident in 2010.")
            ],
            "Did John suffer an injury?": [],
        }
    )
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
    service.record_fused_evidence(fused)

    hands_claim_id = next(c.claim_id for c in atomic_claims if c.text == "John used both hands")
    london_claim_id = next(c.claim_id for c in atomic_claims if c.text == "the event occurred in London")

    nli_model = FakeNLIModel(
        scripted={
            (
                "John lost his left arm in an accident in 2010.",
                "John used both hands",
            ): NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95),
            (
                "John moved to London in 2012.",
                "the event occurred in London",
            ): NLIPrediction(label=NLILabel.ENTAILMENT, score=0.9),
        },
        default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
    )
    rule_config = RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7)

    _run_nli_and_verdict(service, nli_model, rule_config)

    verdict = service.ledger.final_verdict
    assert verdict is not None
    assert verdict.verdict is VerdictEnum.CONTRADICTORY
    assert hands_claim_id in verdict.contradicted_claim_ids
    assert london_claim_id not in verdict.contradicted_claim_ids


def test_offline_pipeline_is_deterministic_end_to_end() -> None:
    def run_once() -> tuple[VerdictEnum, list[str]]:
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
                "the event occurred in London": [
                    make_unstamped_evidence("chunk-london", "John moved to London in 2012.")
                ],
                "John used both hands": [],
                "Did John lose an arm?": [
                    make_unstamped_evidence("chunk-arm", "John lost his left arm in an accident in 2010.")
                ],
                "Did John suffer an injury?": [],
            }
        )
        orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))
        evidence = orchestrator.retrieve_for_queries(queries)
        service.record_retrieved_evidence(evidence)

        fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
        service.record_fused_evidence(fused)

        nli_model = FakeNLIModel(
            scripted={
                (
                    "John lost his left arm in an accident in 2010.",
                    "John used both hands",
                ): NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95),
                (
                    "John moved to London in 2012.",
                    "the event occurred in London",
                ): NLIPrediction(label=NLILabel.ENTAILMENT, score=0.9),
            },
            default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
        )
        rule_config = RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7)
        _run_nli_and_verdict(service, nli_model, rule_config)

        verdict = service.ledger.final_verdict
        assert verdict is not None
        return verdict.verdict, sorted(verdict.contradicted_claim_ids)

    assert run_once() == run_once()


def test_claim_with_zero_retrieved_evidence_routes_to_insufficient_evidence_not_contradictory() -> None:
    """The cardinal invariant at the full-pipeline level: a coverage gap must
    never silently become a contradiction."""
    ledger = EvidenceLedger(original_claim="John traveled to Mars.")
    service = LedgerService(ledger)

    decomp_config = DecompositionConfig(llm_config=LLMConfig(model_name="fake-model"))
    decomposer = LLMClaimDecomposer(
        FakeLLMClient(default_response='["John traveled to Mars"]'), decomp_config
    )
    parent_id = make_source_claim_id("John traveled to Mars.")
    atomic_claims = decomposer.decompose("John traveled to Mars.")
    service.record_atomic_claims(parent_id, atomic_claims)

    question_config = QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model"))
    generator = LLMQuestionGenerator(FakeLLMClient(default_response="[]"), question_config)
    all_questions = generator.generate(atomic_claims[0])
    service.record_probe_questions(all_questions)

    queries = build_retrieval_queries(atomic_claims, all_questions)
    service.record_retrieval_queries(queries)

    retriever = FakeRetriever({"John traveled to Mars": []})
    orchestrator = RetrievalOrchestrator([retriever], RetrievalConfig(top_k=5))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
    service.record_fused_evidence(fused)

    nli_model = FakeNLIModel()
    rule_config = RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7)
    _run_nli_and_verdict(service, nli_model, rule_config)

    verdict = service.ledger.final_verdict
    assert verdict is not None
    assert verdict.verdict is VerdictEnum.INSUFFICIENT_EVIDENCE
    assert verdict.contradicted_claim_ids == []


# --- Variant 2: gated, real end-to-end pipeline ---


@pytest.fixture(scope="module")
def real_embedder() -> SentenceTransformerEmbedder:
    config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return SentenceTransformerEmbedder(config)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load embedding model in this environment: {exc}")


@pytest.fixture(scope="module")
def real_nli_model() -> CrossEncoderNLIModel:
    config = NLIConfig(model_name="cross-encoder/nli-deberta-v3-base")
    try:
        return CrossEncoderNLIModel(config)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Could not load NLI model in this environment: {exc}")


def test_real_pipeline_resolves_section_14_dummy_case_to_contradictory(
    real_embedder: SentenceTransformerEmbedder,
    real_nli_model: CrossEncoderNLIModel,
) -> None:
    document = load_and_clean_narrative(SAMPLE_NARRATIVE, source_id="john_test")
    chunks = chunk_document(document, ChunkingConfig(chunk_size=60, overlap=10))

    chroma_index = ChromaIndex(embedder=real_embedder, collection_name="phase5-e2e")
    chroma_index.index(chunks)
    bm25_index = BM25Index(collection_name="phase5-e2e-bm25")
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

    fused = fuse_evidence(service.ledger.retrieved_evidence, FusionConfig())
    service.record_fused_evidence(fused)

    rule_config = RuleEngineConfig(contradiction_threshold=0.5, entailment_threshold=0.5)
    _run_nli_and_verdict(service, real_nli_model, rule_config)

    verdict = service.ledger.final_verdict
    assert verdict is not None
    assert verdict.verdict is VerdictEnum.CONTRADICTORY, (
        f"expected CONTRADICTORY, got {verdict.verdict} -- rationale: {verdict.rationale}"
    )
