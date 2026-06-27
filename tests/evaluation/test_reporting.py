"""save_report() / save_ledger() / plot_ablation_deltas() / plot_confusion_matrix() tests."""

import json
from pathlib import Path

import pytest

from lncvs.evaluation.reporting import plot_ablation_deltas, plot_confusion_matrix, save_ledger, save_report
from lncvs.schemas import (
    AblatedComponent,
    AblationReport,
    ConfusionCell,
    ConfusionMatrix,
    ContributionDelta,
    EvaluationReport,
    EvidenceLedger,
    ExampleResult,
    LatencyMetrics,
    PerClassMetric,
    ProvenanceFingerprints,
    VerdictEnum,
    VerdictMetrics,
)


def _report() -> EvaluationReport:
    verdict_metrics = VerdictMetrics(
        accuracy=1.0,
        macro_precision=1.0,
        macro_recall=1.0,
        macro_f1=1.0,
        per_class=[
            PerClassMetric(verdict=v, precision=1.0, recall=1.0, f1=1.0, support=1) for v in VerdictEnum
        ],
        confusion=ConfusionMatrix(
            cells=[ConfusionCell(gold=VerdictEnum.CONTRADICTORY, predicted=VerdictEnum.CONTRADICTORY, count=1)]
        ),
        contradiction_detection_rate=1.0,
    )
    example_result = ExampleResult(
        example_id="ex-1",
        predicted_verdict=VerdictEnum.CONTRADICTORY,
        expected_verdict=VerdictEnum.CONTRADICTORY,
        fired_rule="rule_1_contradiction",
        correct=True,
        latency=LatencyMetrics(stages=[], end_to_end_ms=10.0),
        ledger_fingerprint="abc123",
    )
    return EvaluationReport(
        run_id="run-1",
        variant_name="full",
        variant_fingerprint="vf-1",
        dataset_id="ds-1",
        dataset_fingerprint="df-1",
        provenance=ProvenanceFingerprints(eval_config_fp="ec-1", seed=0),
        verdict=verdict_metrics,
        latency=LatencyMetrics(stages=[], end_to_end_ms=10.0),
        example_results=[example_result],
        example_count=1,
    )


def test_save_report_writes_json_keyed_by_run_id(tmp_path: Path) -> None:
    report = _report()
    path = save_report(report, tmp_path)

    assert path == tmp_path / "run-1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"


def test_save_ledger_writes_json_named_by_given_filename(tmp_path: Path) -> None:
    ledger = EvidenceLedger(original_claim="John played a two-handed piano piece in London.")
    path = save_ledger(ledger, tmp_path, "example-1_full_abc123")

    assert path == tmp_path / "example-1_full_abc123.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["original_claim"] == "John played a two-handed piano piece in London."


def test_save_ledger_round_trips_through_model_validate_json(tmp_path: Path) -> None:
    ledger = EvidenceLedger(original_claim="A claim.")
    path = save_ledger(ledger, tmp_path, "ledger-1")

    restored = EvidenceLedger.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored.original_claim == ledger.original_claim


def test_plot_ablation_deltas_writes_a_file(tmp_path: Path) -> None:
    ablation_report = AblationReport(
        reports=[_report()],
        deltas=[
            ContributionDelta(
                component=AblatedComponent.BM25, metric_name="verdict_accuracy", with_value=1.0, without_value=0.5, delta=0.5
            )
        ],
    )

    output_path = plot_ablation_deltas(ablation_report, tmp_path / "deltas.png")

    assert output_path.exists()


def test_plot_ablation_deltas_rejects_empty_deltas(tmp_path: Path) -> None:
    ablation_report = AblationReport(reports=[_report()], deltas=[])

    with pytest.raises(ValueError, match="no contribution deltas"):
        plot_ablation_deltas(ablation_report, tmp_path / "deltas.png")


def test_plot_confusion_matrix_writes_a_file(tmp_path: Path) -> None:
    output_path = plot_confusion_matrix(_report(), tmp_path / "confusion.png")

    assert output_path.exists()
