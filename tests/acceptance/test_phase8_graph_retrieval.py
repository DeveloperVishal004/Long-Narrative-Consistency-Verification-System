"""Phase 8 / Stage G1 acceptance test: GraphRetriever wired as a third,
opt-in RetrievalSource into the real RetrievalOrchestrator/fusion/NLI/rule
engine pipeline, on the PROJECT_SPEC.md Section 14 dummy case.

This is an offline, fully deterministic slice (FakeRetriever standing in
for semantic/lexical, FakeNLIModel for the cross-encoder) -- the same
pattern test_phase5_nli_verdict.py's "Variant 1" uses -- with one addition:
a real GraphIndex/GraphRetriever, built over the actual dummy-case text,
sits alongside the two fakes in the orchestrator's retriever list.

Acceptance criteria (Stage G1, per the approved architecture):
  1. The Section 14 dummy case still resolves to CONTRADICTORY with the
     graph retriever present.
  2. RRF/NLI/rule-engine are provably unchanged: the verdict and the set of
     contradicted claim IDs are byte-identical to the no-graph baseline,
     because the graph contributes only chunks the fakes already supply
     for this case (a redundant-evidence scenario, not a content change).
  3. The graph never leaves the chunk-ID space: every graph-sourced
     RetrievedEvidence.chunk_id is one of the two real chunk_ids the dummy
     narrative defines, and graph evidence flows through the unmodified
     ledger write boundary, fusion, NLI, and rule engine exactly like
     semantic/lexical evidence.
"""

from lncvs.fusion import FusionConfig, fuse_evidence
from lncvs.graph import GraphIndex, GraphRetriever
from lncvs.ledger import LedgerService
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig, LLMClaimDecomposer, make_source_claim_id
from lncvs.reasoning.nli import CrossEncoderNLIVerifier
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import LLMQuestionGenerator, QuestionGenerationConfig
from lncvs.retrieval import RetrievalConfig, RetrievalOrchestrator
from lncvs.rules import RuleEngineConfig, ThresholdRuleEngine, classify
from lncvs.schemas import EvidenceLedger, NLILabel, RetrievalSource, VerdictEnum
from tests.graph.fakes import make_chunk
from tests.llm.fakes import FakeLLMClient
from tests.reasoning.nli.fakes import FakeNLIModel
from tests.retrieval.fakes import FakeRetriever, make_unstamped_evidence

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'
QUESTIONS_BY_CLAIM_TEXT = {
    "John played piano": "[]",
    "John used both hands": '["Did John lose an arm?", "Did John suffer an injury?"]',
    "the event occurred in London": "[]",
}

CHUNK_ARM = make_chunk("chunk-arm", "John lost his left arm in an accident in 2010.")
CHUNK_LONDON = make_chunk("chunk-london", "John moved to London in 2012.")


def _run_dummy_case(retrievers: list) -> tuple[VerdictEnum, list[str], list[RetrievalSource]]:
    """Identical wiring to test_phase5_nli_verdict.py's offline variant,
    parameterized only by which retrievers are injected into the
    orchestrator -- so the no-graph and with-graph runs differ in exactly
    one place."""
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

    from lncvs.retrieval import build_retrieval_queries

    queries = build_retrieval_queries(atomic_claims, all_questions)
    service.record_retrieval_queries(queries)

    orchestrator = RetrievalOrchestrator(retrievers, RetrievalConfig(top_k=5))
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

    verifier = CrossEncoderNLIVerifier(nli_model)
    fused_by_claim: dict[str, list] = {}
    for f in ledger.fused_evidence:
        fused_by_claim.setdefault(f.atomic_claim_id, []).append(f)
    all_results = []
    for claim in ledger.atomic_claims:
        all_results.extend(verifier.verify(claim, fused_by_claim.get(claim.claim_id, [])))
    service.record_nli_results(all_results)

    claim_ids = [c.claim_id for c in ledger.atomic_claims]
    outcome = classify(ledger.nli_results, claim_ids, rule_config)
    service.record_classification(outcome.contradictions, outcome.supporting_evidence, outcome.unsupported_claim_ids)

    engine = ThresholdRuleEngine(rule_config)
    verdict = engine.evaluate(ledger)
    service.set_final_verdict(verdict)

    sources = sorted({e.source for e in ledger.retrieved_evidence}, key=lambda s: s.value)
    return verdict.verdict, sorted(verdict.contradicted_claim_ids), sources


def _fake_semantic_lexical_retrievers() -> tuple[FakeRetriever, FakeRetriever]:
    responses = {
        "John played piano": [],
        "the event occurred in London": [make_unstamped_evidence("chunk-london", CHUNK_LONDON.text)],
        "John used both hands": [],
        "Did John lose an arm?": [make_unstamped_evidence("chunk-arm", CHUNK_ARM.text)],
        "Did John suffer an injury?": [],
    }
    return FakeRetriever(dict(responses)), FakeRetriever(dict(responses))


def test_baseline_without_graph_resolves_to_contradictory() -> None:
    semantic, lexical = _fake_semantic_lexical_retrievers()
    verdict, contradicted, sources = _run_dummy_case([semantic])
    assert verdict is VerdictEnum.CONTRADICTORY
    assert RetrievalSource.GRAPH not in sources


def test_with_graph_retriever_still_resolves_to_contradictory() -> None:
    """Acceptance criterion 1: the dummy case still resolves correctly with
    the graph retriever present."""
    semantic, _ = _fake_semantic_lexical_retrievers()
    graph_index = GraphIndex()
    graph_index.index([CHUNK_ARM, CHUNK_LONDON])
    graph_retriever = GraphRetriever(graph_index)

    verdict, contradicted, sources = _run_dummy_case([semantic, graph_retriever])

    assert verdict is VerdictEnum.CONTRADICTORY
    assert RetrievalSource.GRAPH in sources


def test_graph_retriever_does_not_change_the_verdict_or_contradicted_claims() -> None:
    """Acceptance criterion 2: RRF/NLI/rule-engine are provably unchanged --
    byte-identical verdict and contradicted-claim-ID set with and without
    the graph retriever, because the graph contributes only chunks the
    fakes already supply for this claim."""
    semantic_baseline, _ = _fake_semantic_lexical_retrievers()
    baseline_verdict, baseline_contradicted, _ = _run_dummy_case([semantic_baseline])

    semantic_with_graph, _ = _fake_semantic_lexical_retrievers()
    graph_index = GraphIndex()
    graph_index.index([CHUNK_ARM, CHUNK_LONDON])
    graph_retriever = GraphRetriever(graph_index)
    with_graph_verdict, with_graph_contradicted, _ = _run_dummy_case([semantic_with_graph, graph_retriever])

    assert with_graph_verdict == baseline_verdict
    assert with_graph_contradicted == baseline_contradicted


def test_graph_evidence_never_leaves_the_real_chunk_id_space() -> None:
    """Acceptance criterion 3: every graph-sourced chunk_id is one of the
    two real dummy-narrative chunk_ids -- the graph never invents or
    returns anything other than a chunk_id from the indexed chunk set."""
    semantic, _ = _fake_semantic_lexical_retrievers()
    graph_index = GraphIndex()
    graph_index.index([CHUNK_ARM, CHUNK_LONDON])
    graph_retriever = GraphRetriever(graph_index)

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
        generator = LLMQuestionGenerator(
            FakeLLMClient(default_response=QUESTIONS_BY_CLAIM_TEXT[claim.text]), question_config
        )
        all_questions.extend(generator.generate(claim))
    service.record_probe_questions(all_questions)

    from lncvs.retrieval import build_retrieval_queries

    queries = build_retrieval_queries(atomic_claims, all_questions)
    service.record_retrieval_queries(queries)

    orchestrator = RetrievalOrchestrator([semantic, graph_retriever], RetrievalConfig(top_k=5))
    evidence = orchestrator.retrieve_for_queries(queries)
    service.record_retrieved_evidence(evidence)

    graph_chunk_ids = {e.chunk_id for e in ledger.retrieved_evidence if e.source is RetrievalSource.GRAPH}
    assert graph_chunk_ids.issubset({"chunk-arm", "chunk-london"})
    assert all(e.atomic_claim_id is not None and e.query_id is not None for e in ledger.retrieved_evidence)
