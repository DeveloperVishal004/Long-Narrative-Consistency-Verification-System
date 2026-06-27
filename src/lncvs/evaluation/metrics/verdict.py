"""Verdict-quality metrics: accuracy, macro P/R/F1, confusion matrix, contradiction-detection rate.

All metrics are strictly 3-class over VerdictEnum -- INSUFFICIENT_EVIDENCE
is scored as its own class, never collapsed into a CONTRADICTORY/CONSISTENT
binary.
"""

from lncvs.schemas import VerdictEnum
from lncvs.schemas.evaluation import ConfusionCell, ConfusionMatrix, PerClassMetric, VerdictMetrics

_ALL_VERDICTS = list(VerdictEnum)


def compute_verdict_metrics(pairs: list[tuple[VerdictEnum, VerdictEnum]]) -> VerdictMetrics:
    """pairs is a list of (expected, predicted) verdicts, one per evaluated example."""
    if not pairs:
        raise ValueError("Cannot compute verdict metrics over zero examples")

    total = len(pairs)
    correct = sum(1 for expected, predicted in pairs if expected == predicted)
    accuracy = correct / total

    per_class: list[PerClassMetric] = []
    for verdict in _ALL_VERDICTS:
        true_positive = sum(1 for expected, predicted in pairs if expected == verdict and predicted == verdict)
        false_positive = sum(1 for expected, predicted in pairs if expected != verdict and predicted == verdict)
        false_negative = sum(1 for expected, predicted in pairs if expected == verdict and predicted != verdict)
        support = sum(1 for expected, _ in pairs if expected == verdict)

        precision = (
            true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        )
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class.append(
            PerClassMetric(verdict=verdict, precision=precision, recall=recall, f1=f1, support=support)
        )

    macro_precision = sum(metric.precision for metric in per_class) / len(per_class)
    macro_recall = sum(metric.recall for metric in per_class) / len(per_class)
    macro_f1 = sum(metric.f1 for metric in per_class) / len(per_class)

    cells: list[ConfusionCell] = []
    for gold in _ALL_VERDICTS:
        for predicted in _ALL_VERDICTS:
            count = sum(1 for expected, actual in pairs if expected == gold and actual == predicted)
            if count > 0:
                cells.append(ConfusionCell(gold=gold, predicted=predicted, count=count))

    contradiction_metric = next(metric for metric in per_class if metric.verdict is VerdictEnum.CONTRADICTORY)
    contradiction_detection_rate = contradiction_metric.recall if contradiction_metric.support > 0 else None

    return VerdictMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=ConfusionMatrix(cells=cells),
        contradiction_detection_rate=contradiction_detection_rate,
    )
