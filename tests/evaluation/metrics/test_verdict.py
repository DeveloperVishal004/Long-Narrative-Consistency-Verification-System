"""compute_verdict_metrics() tests, hand-computed."""

import pytest

from lncvs.evaluation.metrics.verdict import compute_verdict_metrics
from lncvs.schemas import VerdictEnum

C = VerdictEnum.CONTRADICTORY
S = VerdictEnum.CONSISTENT
INSUFF = VerdictEnum.INSUFFICIENT_EVIDENCE


def test_raises_on_empty_pairs() -> None:
    with pytest.raises(ValueError, match="zero examples"):
        compute_verdict_metrics([])


def test_perfect_predictions_yield_accuracy_one() -> None:
    pairs = [(C, C), (S, S), (INSUFF, INSUFF)]
    metrics = compute_verdict_metrics(pairs)

    assert metrics.accuracy == 1.0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_recall == 1.0
    assert metrics.macro_f1 == 1.0


def test_hand_computed_accuracy_and_per_class_metrics() -> None:
    """4 examples: gold = [C, C, S, I], predicted = [C, S, S, I].
    Accuracy = 3/4 = 0.75.
    CONTRADICTORY: TP=1 (idx0), FN=1 (idx1 gold=C pred=S), FP=0 -> precision=1.0, recall=0.5, f1=2*1*0.5/1.5=0.667
    CONSISTENT: TP=1 (idx2), FP=1 (idx1, gold=C pred=S), FN=0 -> precision=0.5, recall=1.0, f1=2*0.5*1/1.5=0.667
    INSUFFICIENT_EVIDENCE: TP=1, FP=0, FN=0 -> precision=1.0, recall=1.0, f1=1.0
    """
    pairs = [(C, C), (C, S), (S, S), (INSUFF, INSUFF)]
    metrics = compute_verdict_metrics(pairs)

    assert metrics.accuracy == pytest.approx(0.75)

    by_verdict = {m.verdict: m for m in metrics.per_class}
    assert by_verdict[C].precision == pytest.approx(1.0)
    assert by_verdict[C].recall == pytest.approx(0.5)
    assert by_verdict[C].f1 == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert by_verdict[C].support == 2

    assert by_verdict[S].precision == pytest.approx(0.5)
    assert by_verdict[S].recall == pytest.approx(1.0)
    assert by_verdict[S].support == 1

    assert by_verdict[INSUFF].precision == pytest.approx(1.0)
    assert by_verdict[INSUFF].recall == pytest.approx(1.0)
    assert by_verdict[INSUFF].support == 1


def test_per_class_metric_is_zero_when_class_never_predicted_or_gold() -> None:
    pairs = [(C, C), (C, C)]
    metrics = compute_verdict_metrics(pairs)

    by_verdict = {m.verdict: m for m in metrics.per_class}
    assert by_verdict[S].support == 0
    assert by_verdict[S].precision == 0.0
    assert by_verdict[S].recall == 0.0
    assert by_verdict[S].f1 == 0.0


def test_confusion_matrix_cells_match_hand_count() -> None:
    pairs = [(C, C), (C, S), (S, S), (INSUFF, INSUFF)]
    metrics = compute_verdict_metrics(pairs)

    cell_counts = {(cell.gold, cell.predicted): cell.count for cell in metrics.confusion.cells}
    assert cell_counts[(C, C)] == 1
    assert cell_counts[(C, S)] == 1
    assert cell_counts[(S, S)] == 1
    assert cell_counts[(INSUFF, INSUFF)] == 1
    assert (S, C) not in cell_counts  # zero-count cells are omitted, not stored as 0


def test_contradiction_detection_rate_is_recall_on_contradictory_class() -> None:
    """3 gold-CONTRADICTORY examples, 2 correctly predicted CONTRADICTORY -> recall = 2/3."""
    pairs = [(C, C), (C, C), (C, S), (S, S)]
    metrics = compute_verdict_metrics(pairs)

    assert metrics.contradiction_detection_rate == pytest.approx(2 / 3)


def test_contradiction_detection_rate_is_none_when_no_gold_contradictory_examples() -> None:
    """Must never silently report 0 when there is no gold-CONTRADICTORY data to measure against."""
    pairs = [(S, S), (INSUFF, INSUFF)]
    metrics = compute_verdict_metrics(pairs)

    assert metrics.contradiction_detection_rate is None


def test_verdict_metrics_are_deterministic_across_calls() -> None:
    pairs = [(C, C), (C, S), (S, S), (INSUFF, INSUFF)]
    first = compute_verdict_metrics(pairs)
    second = compute_verdict_metrics(pairs)

    assert first.accuracy == second.accuracy
    assert first.confusion.cells == second.confusion.cells
