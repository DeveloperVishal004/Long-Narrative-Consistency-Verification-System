"""EvaluationHarness tests: single-variant evaluation, determinism, and ablation contribution deltas."""

from pathlib import Path

import pytest

from lncvs.chunking import ChunkingConfig
from lncvs.evaluation import AblationVariant, EvaluationConfig, EvaluationHarness, PipelineRunner, standard_ablation_matrix
from lncvs.llm import LLMConfig
from lncvs.reasoning.decomposition import DecompositionConfig
from lncvs.reasoning.nli.model import NLIPrediction
from lncvs.reasoning.questions import QuestionGenerationConfig
from lncvs.rules import RuleEngineConfig
from lncvs.schemas import EvaluationDataset, GoldExample, GoldSpan, NLILabel, VerdictEnum
from tests.evaluation.fakes import SubstringNLIModel
from tests.indexing.fakes import FakeEmbedder
from tests.llm.fakes import FakeLLMClient

ORIGINAL_CLAIM = "John played a two-handed piano piece in London."
DECOMPOSITION_RESPONSE = '["John played piano", "John used both hands", "the event occurred in London"]'
NARRATIVE_TEXT = "John lost his left arm in an accident in 2010.\n\nJohn moved to London in 2012.\n"


@pytest.fixture()
def narrative_path(tmp_path: Path) -> Path:
    path = tmp_path / "narrative.txt"
    path.write_text(NARRATIVE_TEXT, encoding="utf-8")
    return path


def _make_runner() -> PipelineRunner:
    nli_model = SubstringNLIModel(
        rules=[("lost his left arm", "used both hands", NLIPrediction(label=NLILabel.CONTRADICTION, score=0.95))],
        default_prediction=NLIPrediction(label=NLILabel.NEUTRAL, score=0.5),
    )
    return PipelineRunner(
        embedder=FakeEmbedder(),
        nli_model=nli_model,
        decomposition_llm=FakeLLMClient(default_response=DECOMPOSITION_RESPONSE),
        question_llm=FakeLLMClient(default_response="[]"),
        decomposition_config=DecompositionConfig(llm_config=LLMConfig(model_name="fake-model")),
        question_config=QuestionGenerationConfig(llm_config=LLMConfig(model_name="fake-model")),
        rule_config=RuleEngineConfig(contradiction_threshold=0.7, entailment_threshold=0.7),
        chunking_config=ChunkingConfig(chunk_size=60, overlap=10),
    )


def _dataset(narrative_path: Path) -> EvaluationDataset:
    example = GoldExample(
        example_id="dummy-case",
        narrative_path=str(narrative_path),
        original_claim=ORIGINAL_CLAIM,
        expected_verdict=VerdictEnum.CONTRADICTORY,
        gold_evidence=[GoldSpan(char_start=0, char_end=47)],
    )
    return EvaluationDataset(dataset_id="phase6-test", examples=[example])


def test_evaluate_variant_produces_lightweight_report_with_no_embedded_ledger(narrative_path: Path) -> None:
    harness = EvaluationHarness(_make_runner(), EvaluationConfig())
    dataset = _dataset(narrative_path)
    variant = AblationVariant(name="full")

    report = harness.evaluate_variant(dataset, variant)

    assert report.example_count == 1
    assert report.verdict.accuracy == 1.0
    assert not hasattr(report, "ledger")
    assert report.example_results[0].ledger_path is None
    assert len(report.example_results[0].ledger_fingerprint) > 0


def test_evaluate_variant_run_id_is_deterministic(narrative_path: Path) -> None:
    dataset = _dataset(narrative_path)
    variant = AblationVariant(name="full")

    report_a = EvaluationHarness(_make_runner(), EvaluationConfig()).evaluate_variant(dataset, variant)
    report_b = EvaluationHarness(_make_runner(), EvaluationConfig()).evaluate_variant(dataset, variant)

    assert report_a.run_id == report_b.run_id
    assert report_a.verdict.accuracy == report_b.verdict.accuracy


def test_evaluate_variant_citation_metrics_reflect_correct_citation(narrative_path: Path) -> None:
    harness = EvaluationHarness(_make_runner(), EvaluationConfig())
    dataset = _dataset(narrative_path)
    variant = AblationVariant(name="full")

    report = harness.evaluate_variant(dataset, variant)

    assert report.citation is not None
    assert report.citation.citation_accuracy == 1.0


def test_run_ablation_requires_a_full_variant() -> None:
    harness = EvaluationHarness(_make_runner(), EvaluationConfig())
    dataset = EvaluationDataset(
        dataset_id="ds",
        examples=[
            GoldExample(
                example_id="ex-1", narrative_path="x.txt", original_claim="a claim", expected_verdict=VerdictEnum.CONSISTENT
            )
        ],
    )
    variants = [AblationVariant(name="no_bm25", use_bm25=False)]

    with pytest.raises(ValueError, match='variant named "full"'):
        harness.run_ablation(dataset, variants)


def test_run_ablation_produces_one_delta_per_ablated_component(narrative_path: Path) -> None:
    harness = EvaluationHarness(_make_runner(), EvaluationConfig())
    dataset = _dataset(narrative_path)

    report = harness.run_ablation(dataset, standard_ablation_matrix())

    assert len(report.reports) == 4
    components = {delta.component for delta in report.deltas}
    assert len(components) == 3
    for delta in report.deltas:
        assert delta.delta == pytest.approx(delta.with_value - delta.without_value)


def test_persist_ledgers_false_writes_nothing_to_disk(narrative_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    harness = EvaluationHarness(_make_runner(), EvaluationConfig(output_dir=str(output_dir), persist_ledgers=False))
    dataset = _dataset(narrative_path)

    report = harness.evaluate_variant(dataset, AblationVariant(name="full"))

    assert report.example_results[0].ledger_path is None
    assert not output_dir.exists()


def test_persist_ledgers_true_writes_ledger_and_report_under_output_dir(narrative_path: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    harness = EvaluationHarness(_make_runner(), EvaluationConfig(output_dir=str(output_dir), persist_ledgers=True))
    dataset = _dataset(narrative_path)

    report = harness.evaluate_variant(dataset, AblationVariant(name="full"))

    ledger_path = report.example_results[0].ledger_path
    assert ledger_path is not None
    assert Path(ledger_path).exists()
    assert Path(ledger_path).parent == output_dir / "ledgers"

    report_path = output_dir / f"{report.run_id}.json"
    assert report_path.exists()


def test_persisted_ledger_json_preserves_provenance(narrative_path: Path, tmp_path: Path) -> None:
    """The persisted ledger file must be the real EvidenceLedger content, not a stub --
    provenance must be recoverable from disk, matching the in-memory fingerprint."""
    output_dir = tmp_path / "runs"
    harness = EvaluationHarness(_make_runner(), EvaluationConfig(output_dir=str(output_dir), persist_ledgers=True))
    dataset = _dataset(narrative_path)

    report = harness.evaluate_variant(dataset, AblationVariant(name="full"))
    result = report.example_results[0]

    from lncvs.schemas import EvidenceLedger

    persisted_ledger = EvidenceLedger.model_validate_json(Path(result.ledger_path).read_text(encoding="utf-8"))
    assert persisted_ledger.final_verdict is not None
    assert persisted_ledger.final_verdict.verdict == result.predicted_verdict
    assert len(persisted_ledger.contradictions) > 0
