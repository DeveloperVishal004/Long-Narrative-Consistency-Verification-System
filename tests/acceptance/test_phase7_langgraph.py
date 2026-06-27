"""Phase 7 acceptance tests: LangGraph Integration.

Three variants:

1. An offline, fully deterministic run of the PROJECT_SPEC.md Section 14
   dummy case directly through LangGraphPipeline.

2. A gated, real-model run of the same case through the graph, using the
   real embedder and real cross-encoder NLI model. Skips cleanly if either
   real model cannot be loaded in this environment.

3. Evaluation parity: EvaluationHarness driving LangGraphPipeline over
   datasets/phase6_gold.jsonl must produce identical aggregated metrics to
   EvaluationHarness driving PipelineRunner over the same dataset -- proving
   Phase 6 evaluation functionality is fully preserved.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.evaluation import EvaluationConfig, EvaluationHarness, PipelineRunner, load_dataset
from lncvs.indexing import EmbeddingConfig, SentenceTransformerEmbedder
from lncvs.llm import LLMConfig
from lncvs.orchestration import LangGraphPipeline, PipelineResources
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.decomposition.prompts import render_decomposition_prompt
from lncvs.reasoning.nli import CrossEncoderNLIModel, NLIConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import AblationVariant, NLILabel, VerdictEnum
from tests.evaluation.fakes import SubstringNLIModel
from tests.indexing.fakes import FakeEmbedder
from tests.llm.fakes import FakeLLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DATASET_PATH = REPO_ROOT / "datasets" / "phase6_gold.jsonl"

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_BY_CLAIM = {
    "John played a two-handed piano piece in London.": (
        '["John played piano", "John used both hands", "the event occurred in London"]'
    ),
    "Mary is a doctor in Paris.": '["Mary is a doctor", "Mary works in Paris"]',
    "Tom traveled to Japan last year.": '["Tom traveled to Japan last year"]',
}


# --- Variant 1: offline, fully deterministic ---


def test_offline_dummy_case_resolves_to_contradictory_through_the_graph(tmp_path: Path) -> None:
    narrative_path = tmp_path / "narrative.txt"
    narrative_path.write_text(
        "John lost his left arm in an accident in 2010.\n\nJohn moved to London in 2012.\n", encoding="utf-8"
    )

    resources = PipelineResources(
        embedder=FakeEmbedder(),
        nli_model=SubstringNLIModel(
            rules=[("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95))],
            default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
        ),
        decomposition_llm=FakeLLMClient(default_response=DECOMPOSITION_BY_CLAIM[ORIGINAL_CLAIM]),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
    )
    pipeline = LangGraphPipeline(resources)

    ledger = pipeline.run(narrative_path, ORIGINAL_CLAIM, AblationVariant(name="full"))

    assert ledger.final_verdict is not None
    assert ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY


# --- Variant 2: gated, real-model ---


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


def test_real_pipeline_resolves_section_14_dummy_case_to_contradictory_through_the_graph(
    real_embedder: SentenceTransformerEmbedder,
    real_nli_model: CrossEncoderNLIModel,
) -> None:
    claim = ORIGINAL_CLAIM
    decomposition_scripts = {render_decomposition_prompt(claim): DECOMPOSITION_BY_CLAIM[claim]}

    resources = PipelineResources(
        embedder=real_embedder,
        nli_model=real_nli_model,
        decomposition_llm=FakeLLMClient(scripted=decomposition_scripts),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.5, entailment_threshold=0.5),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
        retrieval_top_k=10,
    )
    pipeline = LangGraphPipeline(resources)

    ledger = pipeline.run(
        REPO_ROOT / "data" / "sample_narrative" / "john_test.txt", claim, AblationVariant(name="full")
    )

    assert ledger.final_verdict is not None
    assert ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY, (
        f"expected CONTRADICTORY, got {ledger.final_verdict.verdict} -- rationale: {ledger.final_verdict.rationale}"
    )


# --- Variant 3: evaluation parity (graph vs runner, driven by EvaluationHarness) ---


def _decomposition_llm() -> FakeLLMClient:
    scripts = {render_decomposition_prompt(claim): response for claim, response in DECOMPOSITION_BY_CLAIM.items()}
    return FakeLLMClient(scripted=scripts)


def _nli_model() -> SubstringNLIModel:
    return SubstringNLIModel(
        rules=[
            ("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95)),
            ("doctor in Paris", "is a doctor", NLIPrediction(label=NLILabel.ENTAILMENT, score=0.9)),
            ("medicine in Paris", "works in Paris", NLIPrediction(label=NLILabel.ENTAILMENT, score=0.85)),
        ],
        default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
    )


def _common_kwargs() -> dict:
    return dict(
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
        retrieval_top_k=10,
    )


def test_evaluation_harness_produces_identical_metrics_for_runner_and_graph() -> None:
    dataset = load_dataset(GOLD_DATASET_PATH, dataset_id="phase7-parity")
    eval_config = EvaluationConfig()
    full_variant = AblationVariant(name="full")

    runner = PipelineRunner(
        embedder=FakeEmbedder(), nli_model=_nli_model(), decomposition_llm=_decomposition_llm(),
        question_llm=FakeLLMClient(default_response="[]"), **_common_kwargs(),
    )
    runner_report = EvaluationHarness(runner, eval_config).evaluate_variant(dataset, full_variant)

    resources = PipelineResources(
        embedder=FakeEmbedder(), nli_model=_nli_model(), decomposition_llm=_decomposition_llm(),
        question_llm=FakeLLMClient(default_response="[]"), **_common_kwargs(),
    )
    pipeline = LangGraphPipeline(resources)
    graph_report = EvaluationHarness(pipeline, eval_config).evaluate_variant(dataset, full_variant)

    assert runner_report.verdict.accuracy == graph_report.verdict.accuracy == 1.0
    assert runner_report.verdict.confusion.cells == graph_report.verdict.confusion.cells
    assert runner_report.example_count == graph_report.example_count == 3
    for runner_result, graph_result in zip(runner_report.example_results, graph_report.example_results):
        assert runner_result.predicted_verdict == graph_result.predicted_verdict
        assert runner_result.correct == graph_result.correct
