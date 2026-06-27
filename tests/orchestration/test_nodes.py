"""Per-node tests: each node, given a pre-state, produces the expected ledger
mutation and advances control.current_stage -- or, on failure, routes to the
error state without crashing.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.orchestration.nodes import (
    decompose_claim,
    error_sink,
    fuse,
    generate_questions,
    ingest_and_index,
    retrieve,
    route_after,
    verdict,
    verify_nli,
)
from lncvs.orchestration.resources import PipelineResources, RunContext
from lncvs.orchestration.state_channels import GraphChannels
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import (
    AblationVariant,
    ControlState,
    EvidenceLedger,
    FusionStrategy,
    NLILabel,
    PipelineStage,
    RetrievalSource,
)
from tests.evaluation.fakes import SubstringNLIModel
from tests.indexing.fakes import FakeEmbedder
from tests.llm.fakes import FakeLLMClient
from tests.retrieval.fakes import FakeRetriever, make_unstamped_evidence

NARRATIVE_TEXT = "John lost his left arm in an accident in 2010.\n\nJohn moved to London in 2012.\n"
CLAIM = "John played a two-handed piano piece in London."
DECOMP = '["John played piano", "John used both hands", "the event occurred in London"]'


@pytest.fixture()
def narrative_path(tmp_path: Path) -> Path:
    path = tmp_path / "narrative.txt"
    path.write_text(NARRATIVE_TEXT, encoding="utf-8")
    return path


def _resources(**overrides) -> PipelineResources:
    defaults = dict(
        embedder=FakeEmbedder(),
        nli_model=SubstringNLIModel(rules=[]),
        decomposition_llm=FakeLLMClient(default_response=DECOMP),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
    )
    defaults.update(overrides)
    return PipelineResources(**defaults)


def _state(stage: PipelineStage, ledger: EvidenceLedger | None = None) -> GraphChannels:
    return GraphChannels(
        ledger=ledger or EvidenceLedger(original_claim=CLAIM),
        control=ControlState(current_stage=stage, config_fingerprint="test-fp"),
    )


def test_ingest_and_index_writes_nothing_to_ledger_and_advances_stage(narrative_path: Path) -> None:
    resources = _resources()
    run_context = RunContext(variant=AblationVariant(name="full"))
    state = _state(PipelineStage.INGESTION)

    result = ingest_and_index(
        state, {"configurable": {"resources": resources, "run_context": run_context, "narrative_path": narrative_path}}
    )

    assert "ledger" not in result
    assert result["control"].current_stage is PipelineStage.CLAIM_DECOMPOSITION
    assert run_context.semantic_retriever is not None
    assert run_context.lexical_retriever is not None


def test_ingest_and_index_skips_bm25_when_variant_disables_it(narrative_path: Path) -> None:
    resources = _resources()
    run_context = RunContext(variant=AblationVariant(name="no_bm25", use_bm25=False))
    state = _state(PipelineStage.INGESTION)

    ingest_and_index(
        state, {"configurable": {"resources": resources, "run_context": run_context, "narrative_path": narrative_path}}
    )

    assert run_context.lexical_retriever is None


def test_ingest_and_index_routes_to_error_on_missing_narrative(tmp_path: Path) -> None:
    resources = _resources()
    run_context = RunContext(variant=AblationVariant(name="full"))
    state = _state(PipelineStage.INGESTION)
    missing_path = tmp_path / "does-not-exist.txt"

    result = ingest_and_index(
        state, {"configurable": {"resources": resources, "run_context": run_context, "narrative_path": missing_path}}
    )

    assert result["control"].current_stage is PipelineStage.ERROR
    assert len(result["control"].errors) == 1
    assert result["control"].errors[0].stage is PipelineStage.INGESTION


def test_decompose_claim_populates_atomic_claims() -> None:
    resources = _resources()
    state = _state(PipelineStage.CLAIM_DECOMPOSITION)

    result = decompose_claim(state, {"configurable": {"resources": resources}})

    assert len(result["ledger"].atomic_claims) == 3
    assert result["control"].current_stage is PipelineStage.QUESTION_GENERATION


def _ledger_after_decomposition() -> EvidenceLedger:
    state = _state(PipelineStage.CLAIM_DECOMPOSITION)
    result = decompose_claim(state, {"configurable": {"resources": _resources()}})
    return result["ledger"]


def test_generate_questions_yields_empty_list_when_variant_disables_qg() -> None:
    ledger = _ledger_after_decomposition()
    state = _state(PipelineStage.QUESTION_GENERATION, ledger)
    run_context = RunContext(variant=AblationVariant(name="no_qg", use_question_generation=False))

    result = generate_questions(state, {"configurable": {"resources": _resources(), "run_context": run_context}})

    assert result["ledger"].probe_questions == []
    assert result["control"].current_stage is PipelineStage.RETRIEVAL


def test_generate_questions_calls_generator_when_variant_enables_qg() -> None:
    ledger = _ledger_after_decomposition()
    state = _state(PipelineStage.QUESTION_GENERATION, ledger)
    run_context = RunContext(variant=AblationVariant(name="full"))
    resources = _resources(question_llm=FakeLLMClient(default_response='["Did John lose an arm?"]'))

    result = generate_questions(state, {"configurable": {"resources": resources, "run_context": run_context}})

    assert len(result["ledger"].probe_questions) == 3  # one per atomic claim


def _ledger_after_questions() -> EvidenceLedger:
    ledger = _ledger_after_decomposition()
    state = _state(PipelineStage.QUESTION_GENERATION, ledger)
    run_context = RunContext(variant=AblationVariant(name="no_qg", use_question_generation=False))
    result = generate_questions(state, {"configurable": {"resources": _resources(), "run_context": run_context}})
    return result["ledger"]


def test_retrieve_records_queries_and_evidence() -> None:
    ledger = _ledger_after_questions()
    state = _state(PipelineStage.RETRIEVAL, ledger)
    run_context = RunContext(variant=AblationVariant(name="full"))
    run_context.semantic_retriever = FakeRetriever(
        {
            "John played piano": [],
            "John used both hands": [make_unstamped_evidence("chunk-arm", "John lost his left arm.")],
            "the event occurred in London": [],
        }
    )
    run_context.lexical_retriever = FakeRetriever(
        {
            "John played piano": [],
            "John used both hands": [],
            "the event occurred in London": [],
        }
    )

    result = retrieve(state, {"configurable": {"resources": _resources(), "run_context": run_context}})

    assert len(result["ledger"].retrieval_queries) == 3
    assert len(result["ledger"].retrieved_evidence) == 1
    assert result["control"].current_stage is PipelineStage.FUSION


def test_retrieve_omits_lexical_source_when_bm25_disabled() -> None:
    ledger = _ledger_after_questions()
    state = _state(PipelineStage.RETRIEVAL, ledger)
    run_context = RunContext(variant=AblationVariant(name="no_bm25", use_bm25=False))
    run_context.semantic_retriever = FakeRetriever(
        {
            "John played piano": [],
            "John used both hands": [make_unstamped_evidence("chunk-arm", "John lost his left arm.")],
            "the event occurred in London": [],
        }
    )
    run_context.lexical_retriever = None  # never consulted

    result = retrieve(state, {"configurable": {"resources": _resources(), "run_context": run_context}})

    assert all(e.source is RetrievalSource.SEMANTIC for e in result["ledger"].retrieved_evidence)


def _ledger_with_retrieved_evidence() -> EvidenceLedger:
    ledger = _ledger_after_questions()
    state = _state(PipelineStage.RETRIEVAL, ledger)
    run_context = RunContext(variant=AblationVariant(name="full"))
    run_context.semantic_retriever = FakeRetriever(
        {
            "John played piano": [],
            "John used both hands": [make_unstamped_evidence("chunk-arm", "John lost his left arm.")],
            "the event occurred in London": [],
        }
    )
    run_context.lexical_retriever = FakeRetriever(
        {"John played piano": [], "John used both hands": [], "the event occurred in London": []}
    )
    result = retrieve(state, {"configurable": {"resources": _resources(), "run_context": run_context}})
    return result["ledger"]


def test_fuse_uses_rrf_by_default() -> None:
    ledger = _ledger_with_retrieved_evidence()
    state = _state(PipelineStage.FUSION, ledger)
    run_context = RunContext(variant=AblationVariant(name="full"))

    result = fuse(state, {"configurable": {"resources": _resources(), "run_context": run_context}})

    assert len(result["ledger"].fused_evidence) == 1
    assert result["control"].current_stage is PipelineStage.NLI_VERIFICATION


def test_fuse_uses_round_robin_when_variant_requests_it() -> None:
    ledger = _ledger_with_retrieved_evidence()
    state = _state(PipelineStage.FUSION, ledger)
    run_context = RunContext(variant=AblationVariant(name="no_rrf", fusion_strategy=FusionStrategy.ROUND_ROBIN))

    result = fuse(state, {"configurable": {"resources": _resources(), "run_context": run_context}})

    assert len(result["ledger"].fused_evidence) == 1
    # round-robin's score formula is 1/(1+best_rank) = 1/(1+1) = 0.5 here,
    # distinguishable from RRF's 1/(rrf_k+rank) (rrf_k=60 by default -> ~0.0164)
    assert result["ledger"].fused_evidence[0].rrf_score == pytest.approx(0.5)


def _ledger_with_fused_evidence() -> EvidenceLedger:
    ledger = _ledger_with_retrieved_evidence()
    state = _state(PipelineStage.FUSION, ledger)
    run_context = RunContext(variant=AblationVariant(name="full"))
    result = fuse(state, {"configurable": {"resources": _resources(), "run_context": run_context}})
    return result["ledger"]


def test_verify_nli_records_one_result_per_fused_evidence() -> None:
    ledger = _ledger_with_fused_evidence()
    state = _state(PipelineStage.NLI_VERIFICATION, ledger)
    resources = _resources(
        nli_model=SubstringNLIModel(
            rules=[("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95))]
        )
    )

    result = verify_nli(state, {"configurable": {"resources": resources}})

    assert len(result["ledger"].nli_results) == 1
    assert result["ledger"].nli_results[0].label is NLILabel.CONTRADICTION
    assert result["control"].current_stage is PipelineStage.RULE_ENGINE


def _ledger_with_nli_results() -> EvidenceLedger:
    ledger = _ledger_with_fused_evidence()
    state = _state(PipelineStage.NLI_VERIFICATION, ledger)
    resources = _resources(
        nli_model=SubstringNLIModel(
            rules=[("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95))]
        )
    )
    result = verify_nli(state, {"configurable": {"resources": resources}})
    return result["ledger"]


def test_verdict_node_only_reads_ledger_never_control() -> None:
    """The rule-engine node must be provably ledger-only: an unrelated control
    value (a different current_stage, pre-existing errors) must not change
    the verdict it produces for an identical ledger."""
    resources = _resources()

    normal_state = _state(PipelineStage.RULE_ENGINE, _ledger_with_nli_results())
    normal_result = verdict(normal_state, {"configurable": {"resources": resources}})

    from datetime import datetime, timezone

    from lncvs.schemas import StageError

    poisoned_control = ControlState(
        current_stage=PipelineStage.RETRIEVAL,  # deliberately "wrong" stage
        errors=[StageError(stage=PipelineStage.INGESTION, message="unrelated prior error", timestamp=datetime.now(timezone.utc))],
        retry_count=7,
        config_fingerprint="unrelated-fingerprint",
    )
    # A separate, freshly-built (but content-identical) ledger -- not the
    # same mutated object, since LedgerService's write-once methods would
    # reject a second classification/verdict on the same instance.
    poisoned_state = GraphChannels(ledger=_ledger_with_nli_results(), control=poisoned_control)
    poisoned_result = verdict(poisoned_state, {"configurable": {"resources": resources}})

    assert normal_result["ledger"].final_verdict.verdict == poisoned_result["ledger"].final_verdict.verdict
    assert normal_result["ledger"].final_verdict.fired_rule == poisoned_result["ledger"].final_verdict.fired_rule
    assert normal_result["ledger"].final_verdict.verdict.value == "CONTRADICTORY"
    assert normal_result["control"].current_stage is PipelineStage.COMPLETE


def test_verdict_node_raises_into_error_boundary_on_empty_atomic_claims() -> None:
    ledger = EvidenceLedger(original_claim=CLAIM)  # no atomic_claims recorded
    state = _state(PipelineStage.RULE_ENGINE, ledger)

    result = verdict(state, {"configurable": {"resources": _resources()}})

    assert result["control"].current_stage is PipelineStage.ERROR
    assert result["control"].errors[0].stage is PipelineStage.RULE_ENGINE


def test_error_sink_is_a_no_op() -> None:
    state = _state(PipelineStage.ERROR)
    assert error_sink(state, {"configurable": {}}) == {}


def test_route_after_routes_to_error_sink_on_error_stage() -> None:
    control = ControlState(current_stage=PipelineStage.ERROR, config_fingerprint="fp")
    state = GraphChannels(ledger=EvidenceLedger(original_claim=CLAIM), control=control)

    assert route_after("next_node")(state) == "error_sink"


def test_route_after_routes_to_next_node_on_success() -> None:
    control = ControlState(current_stage=PipelineStage.RETRIEVAL, config_fingerprint="fp")
    state = GraphChannels(ledger=EvidenceLedger(original_claim=CLAIM), control=control)

    assert route_after("next_node")(state) == "next_node"
