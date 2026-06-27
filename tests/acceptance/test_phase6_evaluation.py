"""Phase 6 acceptance tests: the Evaluation Framework.

Two variants:

1. An offline, fully deterministic full-ablation run over datasets/phase6_gold.jsonl
   (the PROJECT_SPEC.md Section 14 dummy case, a CONSISTENT case, and an
   INSUFFICIENT_EVIDENCE case) using FakeEmbedder/FakeLLMClient/SubstringNLIModel
   with real ChromaIndex/BM25Index, proving the full EvaluationHarness +
   ablation-matrix wiring and that verdict accuracy is 1.0 across all three
   verdict classes.

2. A gated, real-model single-example run reusing the Section 14 dummy case,
   proving PipelineRunner produces a correct ledger when driven by the real
   embedder and real cross-encoder NLI model. Skips cleanly if either real
   model cannot be loaded in this environment.
"""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.evaluation import (
    AblationVariant,
    EvaluationConfig,
    EvaluationHarness,
    PipelineRunner,
    load_dataset,
    standard_ablation_matrix,
)
from lncvs.indexing import EmbeddingConfig, SentenceTransformerEmbedder
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.decomposition.prompts import render_decomposition_prompt
from lncvs.reasoning.nli import CrossEncoderNLIModel, NLIConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import NLILabel, VerdictEnum
from tests.evaluation.fakes import SubstringNLIModel
from tests.indexing.fakes import FakeEmbedder
from tests.llm.fakes import FakeLLMClient

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DATASET_PATH = REPO_ROOT / "datasets" / "phase6_gold.jsonl"

DECOMPOSITION_BY_CLAIM = {
    "John played a two-handed piano piece in London.": (
        '["John played piano", "John used both hands", "the event occurred in London"]'
    ),
    "Mary is a doctor in Paris.": '["Mary is a doctor", "Mary works in Paris"]',
    "Tom traveled to Japan last year.": '["Tom traveled to Japan last year"]',
}


# --- Variant 1: offline, fully deterministic full ablation ---


def _make_offline_runner() -> PipelineRunner:
    decomposition_scripts = {
        render_decomposition_prompt(claim): response for claim, response in DECOMPOSITION_BY_CLAIM.items()
    }
    nli_model = SubstringNLIModel(
        rules=[
            ("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95)),
            ("doctor in Paris", "is a doctor", NLIPrediction(label=NLILabel.ENTAILMENT, score=0.9)),
            ("medicine in Paris", "works in Paris", NLIPrediction(label=NLILabel.ENTAILMENT, score=0.85)),
        ],
        default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
    )
    return PipelineRunner(
        embedder=FakeEmbedder(),
        nli_model=nli_model,
        decomposition_llm=FakeLLMClient(scripted=decomposition_scripts),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
        retrieval_top_k=10,
    )


def test_offline_full_ablation_resolves_all_three_gold_verdicts_correctly() -> None:
    dataset = load_dataset(GOLD_DATASET_PATH, dataset_id="phase6-gold")
    assert {example.expected_verdict for example in dataset.examples} == {
        VerdictEnum.CONTRADICTORY,
        VerdictEnum.CONSISTENT,
        VerdictEnum.INSUFFICIENT_EVIDENCE,
    }

    harness = EvaluationHarness(_make_offline_runner(), EvaluationConfig())
    full_report = harness.evaluate_variant(dataset, AblationVariant(name="full"))

    assert full_report.verdict.accuracy == 1.0
    assert full_report.example_count == 3
    for result in full_report.example_results:
        assert result.correct, f"{result.example_id}: expected {result.expected_verdict}, got {result.predicted_verdict}"


def test_offline_ablation_matrix_produces_contribution_deltas_for_all_three_components() -> None:
    dataset = load_dataset(GOLD_DATASET_PATH, dataset_id="phase6-gold")
    harness = EvaluationHarness(_make_offline_runner(), EvaluationConfig())

    ablation_report = harness.run_ablation(dataset, standard_ablation_matrix())

    assert len(ablation_report.reports) == 4
    assert len(ablation_report.deltas) == 3
    for delta in ablation_report.deltas:
        assert delta.delta == pytest.approx(delta.with_value - delta.without_value)


def test_offline_reports_contain_no_embedded_ledger() -> None:
    dataset = load_dataset(GOLD_DATASET_PATH, dataset_id="phase6-gold")
    harness = EvaluationHarness(_make_offline_runner(), EvaluationConfig())

    report = harness.evaluate_variant(dataset, AblationVariant(name="full"))

    assert not hasattr(report, "ledger")
    for result in report.example_results:
        assert not hasattr(result, "ledger")
        assert result.ledger_path is None


def test_offline_evaluation_is_deterministic_across_repeated_runs() -> None:
    dataset = load_dataset(GOLD_DATASET_PATH, dataset_id="phase6-gold")

    def run_once() -> tuple[float, str]:
        harness = EvaluationHarness(_make_offline_runner(), EvaluationConfig())
        report = harness.evaluate_variant(dataset, AblationVariant(name="full"))
        return report.verdict.accuracy, report.run_id

    assert run_once() == run_once()


# --- Variant 2: gated, real-model single-example run ---


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


def test_real_pipeline_runner_resolves_section_14_dummy_case_to_contradictory(
    real_embedder: SentenceTransformerEmbedder,
    real_nli_model: CrossEncoderNLIModel,
) -> None:
    claim = "John played a two-handed piano piece in London."
    decomposition_scripts = {render_decomposition_prompt(claim): DECOMPOSITION_BY_CLAIM[claim]}

    runner = PipelineRunner(
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

    ledger = runner.run(REPO_ROOT / "data" / "sample_narrative" / "john_test.txt", claim, AblationVariant(name="full"))

    assert ledger.final_verdict is not None
    assert ledger.final_verdict.verdict is VerdictEnum.CONTRADICTORY, (
        f"expected CONTRADICTORY, got {ledger.final_verdict.verdict} -- rationale: {ledger.final_verdict.rationale}"
    )
